"""A small typed state graph.

LangGraph is supported (see :mod:`paperpilot.orchestration.langgraph_adapter`)
but not required. This engine is ~120 lines, has no dependencies, and gives the
three things the workflow actually needs: named nodes over a typed state,
conditional edges, and safe parallel execution of independent nodes.

Nodes return *partial* states which are merged by :func:`merge_state`, so a node
can never accidentally clobber another node's contribution.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..domain.state import ResearchState, merge_state
from ..errors import PaperPilotError
from ..observability import debug, error, info

#: Sentinel target meaning "stop here".
END = "__end__"

NodeFn = Callable[[ResearchState], ResearchState | None]
RouterFn = Callable[[ResearchState], str]


@dataclass
class StateGraph:
    """A directed graph of node functions over a shared typed state."""

    nodes: dict[str, NodeFn] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    routers: dict[str, RouterFn] = field(default_factory=dict)
    parallel_groups: dict[str, list[str]] = field(default_factory=dict)
    entry: str | None = None
    #: Hard ceiling on node executions, so a bad router cannot loop forever.
    max_steps: int = 64

    def add_node(self, name: str, fn: NodeFn) -> StateGraph:
        if name in self.nodes:
            raise ValueError(f"duplicate node: {name}")
        self.nodes[name] = fn
        return self

    def add_parallel_node(self, name: str, members: list[str]) -> StateGraph:
        """Register a node that runs several existing nodes concurrently."""
        missing = [m for m in members if m not in self.nodes]
        if missing:
            raise ValueError(f"parallel group '{name}' references unknown nodes: {missing}")
        self.parallel_groups[name] = list(members)
        self.nodes[name] = self._make_parallel_runner(name, members)
        return self

    def add_edge(self, source: str, target: str) -> StateGraph:
        self.edges[source] = target
        return self

    def add_conditional_edge(self, source: str, router: RouterFn) -> StateGraph:
        self.routers[source] = router
        return self

    def set_entry(self, name: str) -> StateGraph:
        self.entry = name
        return self

    # --- execution --------------------------------------------------------

    def _next(self, current: str, state: ResearchState) -> str:
        if current in self.routers:
            target = self.routers[current](state)
            debug("graph.route", source=current, target=target)
            return target
        return self.edges.get(current, END)

    def _make_parallel_runner(self, name: str, members: list[str]) -> NodeFn:
        def runner(state: ResearchState) -> ResearchState:
            # Each branch gets a shallow copy so concurrent nodes cannot observe
            # each other's half-written updates; results are merged in a fixed
            # order afterwards, which keeps runs reproducible.
            info("graph.parallel_started", group=name, nodes=members)
            updates: dict[str, ResearchState | None] = {}
            with ThreadPoolExecutor(max_workers=max(1, len(members))) as pool:
                futures = {
                    pool.submit(self._run_node, member, dict(state)): member  # type: ignore[arg-type]
                    for member in members
                }
                for future in futures:
                    member = futures[future]
                    updates[member] = future.result()
            merged: ResearchState = {}
            for member in members:  # deterministic merge order
                merge_state(merged, updates.get(member))
            info("graph.parallel_completed", group=name, nodes=members)
            return merged

        return runner

    def _run_node(self, name: str, state: ResearchState) -> ResearchState | None:
        fn = self.nodes[name]
        try:
            return fn(state)
        except PaperPilotError as exc:
            error("graph.node_failed", node=name, code=exc.code, error=exc.message)
            return ResearchState(errors=[f"{name}: {exc.message}"])
        except Exception as exc:  # noqa: BLE001 - one bad node must not lose the run
            error(
                "graph.node_crashed",
                node=name,
                error_type=type(exc).__name__,
                error=str(exc)[:300],
            )
            return ResearchState(
                errors=[f"{name}: unexpected {type(exc).__name__}: {exc}"]
            )

    def run(self, state: ResearchState) -> ResearchState:
        if not self.entry:
            raise ValueError("graph has no entry node")
        current = self.entry
        steps = 0

        while current != END:
            if current not in self.nodes:
                raise ValueError(f"unknown node: {current}")
            steps += 1
            if steps > self.max_steps:
                error("graph.step_limit", node=current, max_steps=self.max_steps)
                merge_state(
                    state,
                    ResearchState(errors=[f"graph exceeded {self.max_steps} steps; stopped"]),
                )
                break
            debug("graph.node_started", node=current, step=steps)
            update = self._run_node(current, state)
            merge_state(state, update)
            current = self._next(current, state)

        info("graph.completed", steps=steps)
        return state
