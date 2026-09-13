"""The PaperPilot analysis graph.

    ingest -> index -> plan -> specialists -> aggregate -> extract claims
      -> verify -> (retry?) -> synthesize

Retry is a real cycle in the graph, bounded by the plan's ``max_retries``. Only
the agents that failed to ground their claims re-run, and they re-run with
feedback describing exactly which claims were rejected.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from ..agents import AGENT_REGISTRY, AgentContext, VerificationAgent
from ..agents.verification import apply_verification, extract_high_impact_claims
from ..config import Settings
from ..domain.enums import AnalysisStatus
from ..domain.models import (
    AgentResult,
    AgentUsage,
    AnalysisPlan,
    Finding,
    VerificationResult,
)
from ..domain.state import ResearchState
from ..errors import IngestionError, PaperPilotError
from ..ingestion.service import IngestionService
from ..observability import info, warning
from ..providers.embeddings import EmbeddingProvider
from ..providers.llm import LLMProvider
from ..reporting.synthesizer import ReportSynthesizer
from ..retrieval.index import PaperIndex
from ..retrieval.retriever import build_retriever
from .engine import END, StateGraph
from .events import EventRecorder
from .planner import create_plan
from .retry import decide_retry, merge_retry_findings


@dataclass
class WorkflowResources:
    """Everything the graph nodes need, resolved once per run."""

    settings: Settings
    llm: LLMProvider
    embedder: EmbeddingProvider
    recorder: EventRecorder
    ingestion: IngestionService
    synthesizer: ReportSynthesizer
    #: Populated by the ingest node, read by everything downstream.
    index: PaperIndex | None = None
    retriever: object | None = None
    plan: AnalysisPlan | None = None
    verification_agent: VerificationAgent | None = None
    usage: list[AgentUsage] = field(default_factory=list)


def _merge_verifications(
    previous: list[VerificationResult], latest: list[VerificationResult]
) -> list[VerificationResult]:
    """Keep one verdict per claim: the most recent one.

    A retry re-verifies the same claims, so appending blindly would make the
    report count every claim twice and show stale verdicts alongside fresh ones.
    """
    merged = {r.claim.strip().lower(): r for r in previous}
    for result in latest:
        merged[result.claim.strip().lower()] = result
    return list(merged.values())


def _finding_id(agent: str, position: int) -> str:
    return f"{agent}-{position:02d}-{uuid.uuid4().hex[:6]}"


class AnalysisWorkflow:
    """Builds and runs the analysis graph for one paper."""

    def __init__(self, resources: WorkflowResources) -> None:
        self.res = resources
        self.graph = self._build()

    # --- graph construction ----------------------------------------------

    def _build(self) -> StateGraph:
        graph = StateGraph()
        graph.add_node("ingest", self.node_ingest)
        graph.add_node("plan", self.node_plan)

        for name in AGENT_REGISTRY:
            graph.add_node(f"agent:{name}", self._make_agent_node(name))

        if self.res.settings.parallel_agents:
            # Specialist agents are independent: none of them reads another's
            # output, so they are safe to run concurrently.
            graph.add_parallel_node("specialists", [f"agent:{n}" for n in AGENT_REGISTRY])
        else:
            graph.add_node("specialists", self._make_sequential_specialists())

        graph.add_node("aggregate", self.node_aggregate)
        graph.add_node("verify", self.node_verify)
        graph.add_node("retry", self.node_retry)
        graph.add_node("synthesize", self.node_synthesize)

        graph.set_entry("ingest")
        graph.add_conditional_edge("ingest", self._after_ingest)
        graph.add_edge("plan", "specialists")
        graph.add_edge("specialists", "aggregate")
        graph.add_edge("aggregate", "verify")
        graph.add_conditional_edge("verify", self._after_verify)
        graph.add_edge("retry", "aggregate")
        graph.add_edge("synthesize", END)
        return graph

    def _after_ingest(self, state: ResearchState) -> str:
        return END if state.get("status") is AnalysisStatus.FAILED else "plan"

    def _after_verify(self, state: ResearchState) -> str:
        return "retry" if state.get("retry_targets") else "synthesize"

    # --- nodes ------------------------------------------------------------

    def node_ingest(self, state: ResearchState) -> ResearchState:
        recorder = self.res.recorder
        recorder.started("ingestion", "Reading and indexing the PDF")
        try:
            result = self.res.ingestion.ingest(
                state["file_path"], document_id=state.get("document_id")
            )
        except PaperPilotError as exc:
            recorder.failed("ingestion", exc.message, code=exc.code)
            return ResearchState(
                status=AnalysisStatus.FAILED,
                errors=[f"ingestion: {exc.message}"],
                index_status="failed",
                events=recorder.drain(),
            )

        self.res.index = result.index
        self.res.retriever = build_retriever(
            result.index,
            top_k=self.res.settings.retrieval_top_k,
            rerank_enabled=self.res.settings.rerank_enabled,
        )
        metadata = result.metadata

        recorder.completed(
            "ingestion",
            f"Parsed {metadata.page_count} pages into {metadata.chunk_count} page-aware chunks",
            pages=metadata.page_count,
            chunks=metadata.chunk_count,
            ocr_pages=metadata.ocr_page_count,
            sections=len(metadata.sections),
            tables=metadata.table_count,
            figures=metadata.figure_count,
            equations=metadata.equation_count,
        )
        recorder.completed(
            "indexing",
            f"Indexed {metadata.chunk_count} chunks for hybrid retrieval",
            lexical=result.index.bm25.size,
            dense=result.index.vectors.size if result.index.vectors else 0,
        )
        return ResearchState(
            document_id=result.document_id,
            paper_metadata=metadata,
            index_status="ready",
            status=AnalysisStatus.PLANNING,
            events=recorder.drain(),
        )

    def node_plan(self, state: ResearchState) -> ResearchState:
        recorder = self.res.recorder
        recorder.started("planning", "Deciding which specialists this paper needs")
        assert self.res.index is not None

        plan = create_plan(self.res.index, self.res.settings, self.res.llm)
        self.res.plan = plan

        recorder.completed(
            "planning",
            "Plan: " + ", ".join(plan.enabled_agents()),
            agents=plan.enabled_agents(),
            skipped=[
                name
                for name in AGENT_REGISTRY
                if name not in plan.enabled_agents()
            ],
            priority_sections=plan.priority_sections,
            rationale=plan.rationale,
            created_by=plan.created_by,
        )
        return ResearchState(
            analysis_plan=plan, status=AnalysisStatus.ANALYZING, events=recorder.drain()
        )

    def _agent_context(self, state: ResearchState, *, attempt: int, feedback: str | None):
        assert self.res.index is not None and self.res.retriever is not None
        plan = self.res.plan
        return AgentContext(
            document_id=state["document_id"],
            metadata=state["paper_metadata"],
            index=self.res.index,
            retriever=self.res.retriever,  # type: ignore[arg-type]
            priority_sections=plan.priority_sections if plan else [],
            attempt=attempt,
            retry_feedback=feedback,
        )

    def _instantiate(self, name: str):
        cls = AGENT_REGISTRY[name]
        if name == "math":
            return cls(self.res.llm, min_equations=self.res.settings.math_min_equations)
        return cls(self.res.llm)

    def _run_agent(self, name: str, state: ResearchState) -> ResearchState:
        recorder = self.res.recorder
        plan = self.res.plan
        enabled = plan.enabled_agents() if plan else list(AGENT_REGISTRY)

        if name not in enabled:
            reason = (plan.rationale.get(f"run_{name}") if plan else None) or (
                "the plan did not select this agent"
            )
            recorder.skipped(f"agent:{name}", reason, agent=name)
            return ResearchState(
                agent_results={
                    name: AgentResult(agent=name, status="skipped", skip_reason=reason)
                },
                events=recorder.drain(),
            )

        attempt = int(state.get("retry_count", 0)) + 1
        feedback = (state.get("retry_feedback") or {}).get(name) if isinstance(
            state.get("retry_feedback"), dict
        ) else None

        recorder.started(f"agent:{name}", f"Running the {name} agent", agent=name, attempt=attempt)
        agent = self._instantiate(name)
        result = agent.run(self._agent_context(state, attempt=attempt, feedback=feedback))

        if result.status == "skipped":
            recorder.skipped(f"agent:{name}", result.skip_reason or "", agent=name)
        elif result.status == "failed":
            recorder.failed(f"agent:{name}", result.error or "", agent=name)
        else:
            recorder.completed(
                f"agent:{name}",
                f"{len(result.findings)} findings, "
                f"{sum(1 for f in result.findings if f.has_evidence)} with verified evidence",
                agent=name,
                findings=len(result.findings),
                attempt=attempt,
            )

        usage = [result.usage] if result.usage else []
        return ResearchState(
            agent_results={name: result}, usage=usage, events=recorder.drain()
        )

    def _make_agent_node(self, name: str):
        def node(state: ResearchState) -> ResearchState:
            return self._run_agent(name, state)

        return node

    def _make_sequential_specialists(self):
        def node(state: ResearchState) -> ResearchState:
            merged: ResearchState = ResearchState(agent_results={}, usage=[], events=[])
            for name in AGENT_REGISTRY:
                update = self._run_agent(name, state)
                merged["agent_results"].update(update.get("agent_results", {}))  # type: ignore[union-attr]
                merged["usage"] = [*merged.get("usage", []), *update.get("usage", [])]
                merged["events"] = [*merged.get("events", []), *update.get("events", [])]
            return merged

        return node

    def node_aggregate(self, state: ResearchState) -> ResearchState:
        recorder = self.res.recorder
        recorder.started("aggregation", "Collecting findings from every agent")

        findings: list[Finding] = []
        for name, result in state.get("agent_results", {}).items():
            for position, finding in enumerate(result.findings, start=1):
                if not finding.finding_id:
                    finding.finding_id = _finding_id(name, position)
                findings.append(finding)

        if state.get("retry_count", 0) and state.get("retry_targets"):
            findings = merge_retry_findings(
                state.get("findings", []), findings, state["retry_targets"]
            )

        grounded = sum(1 for f in findings if f.has_evidence)
        recorder.completed(
            "aggregation",
            f"{len(findings)} findings collected, {grounded} carry verified evidence",
            findings=len(findings),
            grounded=grounded,
        )
        return ResearchState(
            findings=findings, status=AnalysisStatus.VERIFYING, events=recorder.drain()
        )

    def node_verify(self, state: ResearchState) -> ResearchState:
        recorder = self.res.recorder
        plan = self.res.plan
        threshold = plan.verification_threshold if plan else 0.85
        findings = state.get("findings", [])

        candidates = extract_high_impact_claims(findings)
        recorder.started(
            "verification",
            f"Verifying {len(candidates)} high-impact claims against the paper",
            claims=len(candidates),
        )

        if self.res.verification_agent is None:
            self.res.verification_agent = VerificationAgent(self.res.llm)
        verifier = self.res.verification_agent

        context = self._agent_context(
            state, attempt=int(state.get("retry_count", 0)) + 1, feedback=None
        )
        results = verifier.verify_all(candidates, context)
        apply_verification(findings, results)

        counts: dict[str, int] = {}
        for result in results:
            counts[result.status.value] = counts.get(result.status.value, 0) + 1

        recorder.completed(
            "verification",
            "Verified "
            + ", ".join(f"{count} {status}" for status, count in sorted(counts.items())),
            **counts,
        )

        decision = decide_retry(
            agent_results=state.get("agent_results", {}),
            findings=findings,
            verification_results=results,
            retry_count=int(state.get("retry_count", 0)),
            max_retries=plan.max_retries if plan else 0,
            threshold=threshold,
            already_retried=state.get("retry_targets_history", []),  # type: ignore[arg-type]
        )

        if decision.should_retry:
            recorder.emit(
                "retry",
                "scheduled",
                "Re-running agents whose claims could not be grounded: "
                + ", ".join(decision.agents),
                agents=decision.agents,
                reasons=decision.reasons,
            )
        else:
            recorder.skipped("retry", decision.stop_reason or "no retry needed")

        update = ResearchState(
            findings=findings,
            verification_results=_merge_verifications(
                state.get("verification_results", []), results
            ),
            usage=list(verifier.usage),
            retry_targets=decision.agents if decision.should_retry else [],
            events=recorder.drain(),
        )
        verifier.usage = []
        if decision.should_retry:
            update["retry_feedback"] = decision.reasons  # type: ignore[typeddict-unknown-key]
        return update

    def node_retry(self, state: ResearchState) -> ResearchState:
        recorder = self.res.recorder
        targets = state.get("retry_targets", [])
        retry_count = int(state.get("retry_count", 0)) + 1
        recorder.started(
            "retry",
            f"Attempt {retry_count}: re-running {', '.join(targets)}",
            agents=targets,
            attempt=retry_count,
        )

        results: dict[str, AgentResult] = {}
        usage: list[AgentUsage] = []
        feedback = state.get("retry_feedback") or {}

        for name in targets:
            agent = self._instantiate(name)
            context = self._agent_context(
                state, attempt=retry_count + 1, feedback=feedback.get(name)  # type: ignore[union-attr]
            )
            result = agent.run(context)
            results[name] = result
            if result.usage:
                usage.append(result.usage)
            recorder.completed(
                f"agent:{name}",
                f"Retry produced {len(result.findings)} findings",
                agent=name,
                attempt=retry_count + 1,
                findings=len(result.findings),
            )

        history = [*state.get("retry_targets_history", []), *targets]  # type: ignore[arg-type]
        update = ResearchState(
            agent_results=results,
            usage=usage,
            retry_count=retry_count,
            status=AnalysisStatus.RETRYING,
            events=recorder.drain(),
        )
        update["retry_targets_history"] = history  # type: ignore[typeddict-unknown-key]
        return update

    def node_synthesize(self, state: ResearchState) -> ResearchState:
        recorder = self.res.recorder
        recorder.started("report", "Writing the report")
        assert self.res.index is not None

        report = self.res.synthesizer.synthesize(
            metadata=state["paper_metadata"],
            index=self.res.index,
            plan=state.get("analysis_plan"),
            agent_results=state.get("agent_results", {}),
            findings=state.get("findings", []),
            verification_results=state.get("verification_results", []),
            events=state.get("events", []),
            usage=state.get("usage", []),
            retry_count=int(state.get("retry_count", 0)),
        )
        recorder.completed("report", f"Report written ({len(report)} characters)")
        return ResearchState(
            final_report=report, status=AnalysisStatus.COMPLETED, events=recorder.drain()
        )

    # --- entry point ------------------------------------------------------

    def run(self, state: ResearchState) -> ResearchState:
        started = time.perf_counter()
        try:
            final = self.graph.run(state)
        except IngestionError as exc:
            warning("workflow.failed", error=exc.message)
            state["status"] = AnalysisStatus.FAILED
            state.setdefault("errors", []).append(exc.message)
            return state
        info(
            "workflow.completed",
            document_id=final.get("document_id"),
            status=str(final.get("status")),
            findings=len(final.get("findings", [])),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return final
