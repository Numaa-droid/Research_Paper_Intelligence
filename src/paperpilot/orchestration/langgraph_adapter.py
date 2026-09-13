"""Optional LangGraph adapter.

The workflow is defined against the in-house :class:`StateGraph` so that the
system has no orchestration dependency. This adapter rebuilds the *same* node
functions as a LangGraph ``StateGraph`` for anyone who wants LangGraph's
checkpointing, streaming or tracing.

It is intentionally thin: the nodes are identical objects, so behaviour cannot
drift between the two engines.
"""

from __future__ import annotations

from typing import Any

from ..domain.state import ResearchState
from ..errors import ConfigurationError
from .engine import END
from .graph import AnalysisWorkflow


def langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True


def build_langgraph(workflow: AnalysisWorkflow) -> Any:  # pragma: no cover - optional dependency
    """Compile the same workflow as a LangGraph graph."""
    try:
        from langgraph.graph import END as LG_END
        from langgraph.graph import StateGraph as LangStateGraph
    except ImportError as exc:
        raise ConfigurationError(
            "langgraph is not installed; install the 'langgraph' extra or use the "
            "built-in state graph (the default)"
        ) from exc

    source = workflow.graph
    graph = LangStateGraph(ResearchState)

    for name, fn in source.nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point(source.entry)

    for node, target in source.edges.items():
        graph.add_edge(node, LG_END if target == END else target)

    for node, router in source.routers.items():
        targets = {
            name: (LG_END if name == END else name)
            for name in {*source.nodes, END}
        }

        def make_router(fn=router):
            def route(state: ResearchState) -> str:
                return fn(state)

            return route

        graph.add_conditional_edges(node, make_router(), targets)

    return graph.compile()
