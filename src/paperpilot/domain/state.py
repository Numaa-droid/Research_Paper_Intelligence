"""The orchestrator's typed state object.

``ResearchState`` is a plain ``TypedDict`` so it can be handed to LangGraph
unchanged, while the in-house graph engine treats it as a mutable mapping. Node
functions return *partial* states which are merged by the engine.
"""

from __future__ import annotations

from typing import TypedDict

from .enums import AnalysisStatus
from .models import (
    AgentResult,
    AgentUsage,
    AnalysisPlan,
    ClaimCandidate,
    Evidence,
    Finding,
    PaperMetadata,
    ProgressEvent,
    VerificationResult,
)


class ResearchState(TypedDict, total=False):
    """State threaded through the analysis graph."""

    # Identity and inputs
    analysis_id: str
    document_id: str
    file_path: str

    # Ingestion / indexing
    paper_metadata: PaperMetadata
    index_status: str

    # Planning and execution
    analysis_plan: AnalysisPlan
    retrieved_context: dict[str, list[Evidence]]
    agent_results: dict[str, AgentResult]
    findings: list[Finding]

    # Verification
    claim_candidates: list[ClaimCandidate]
    verification_results: list[VerificationResult]

    # Control
    retry_count: int
    retry_targets: list[str]
    status: AnalysisStatus

    # Outputs and observability
    final_report: str
    #: The two-page brief read before the full report.
    final_brief: str
    events: list[ProgressEvent]
    usage: list[AgentUsage]
    errors: list[str]


def new_state(analysis_id: str, document_id: str, file_path: str) -> ResearchState:
    """Create an empty state with all collection fields initialised."""
    return ResearchState(
        analysis_id=analysis_id,
        document_id=document_id,
        file_path=file_path,
        index_status="pending",
        retrieved_context={},
        agent_results={},
        findings=[],
        claim_candidates=[],
        verification_results=[],
        retry_count=0,
        retry_targets=[],
        status=AnalysisStatus.PENDING,
        final_report="",
        final_brief="",
        events=[],
        usage=[],
        errors=[],
    )


#: Keys whose values are lists that should be *appended* rather than replaced
#: when a node returns a partial state.
APPEND_KEYS = frozenset({"events", "usage", "errors"})

#: Keys whose values are dicts that should be *merged* rather than replaced.
MERGE_KEYS = frozenset({"agent_results", "retrieved_context"})


def merge_state(state: ResearchState, update: ResearchState | None) -> ResearchState:
    """Merge a node's partial state into the running state.

    Append-only keys accumulate so that progress events and token usage from
    every node survive; dict keys merge key-wise; everything else is replaced.
    """
    if not update:
        return state
    for key, value in update.items():
        if key in APPEND_KEYS and isinstance(value, list):
            state.setdefault(key, [])  # type: ignore[arg-type]
            state[key] = [*state[key], *value]  # type: ignore[literal-required]
        elif key in MERGE_KEYS and isinstance(value, dict):
            merged = dict(state.get(key, {}))  # type: ignore[arg-type]
            merged.update(value)
            state[key] = merged  # type: ignore[literal-required]
        else:
            state[key] = value  # type: ignore[literal-required]
    return state
