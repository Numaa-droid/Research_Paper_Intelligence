"""Typed contracts for every boundary in the system.

Nothing unstructured crosses a node boundary: agents return validated Pydantic
models, and the report synthesizer only ever renders citations built from
validated :class:`Evidence` objects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import (
    AnalysisStatus,
    CitationQuality,
    ElementType,
    ExtractionMethod,
    FindingStatus,
)

#: Whose statement a quote is; see :mod:`paperpilot.discourse.attribution`.
Attribution = Literal["own", "cited", "mixed", "unknown"]
#: What kind of paper is being analysed; see :mod:`paperpilot.discourse.profile`.
PaperKind = Literal["empirical", "review", "theoretical", "conceptual"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    """Base model: unknown fields are rejected so provider drift fails loudly."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# Paper representation
# ---------------------------------------------------------------------------


class PageText(StrictModel):
    """One physical PDF page after extraction.

    ``page`` is a 1-based **PDF page index**, not the printed page label. When
    the PDF declares page labels the printed label is preserved separately in
    ``printed_label`` so citations can be disambiguated by a human reader.
    """

    page: int = Field(ge=1)
    text: str
    source: ExtractionMethod = ExtractionMethod.NATIVE_TEXT
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    printed_label: str | None = None
    char_count: int = Field(default=0, ge=0)
    width: float | None = None
    height: float | None = None

    @field_validator("char_count")
    @classmethod
    def _default_char_count(cls, v: int, info) -> int:
        if v == 0:
            return len(info.data.get("text", "") or "")
        return v


class PaperChunk(StrictModel):
    """A page-aware, retrievable unit of the paper.

    Page metadata travels with the text at all times; a chunk never spans pages.
    """

    chunk_id: str
    document_id: str
    page: int = Field(ge=1)
    section: str | None = None
    subsection: str | None = None
    element_type: ElementType = ElementType.PARAGRAPH
    element_id: str | None = None
    text: str
    source: ExtractionMethod = ExtractionMethod.NATIVE_TEXT
    char_start: int | None = None
    char_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    printed_label: str | None = None

    @property
    def citation_section(self) -> str | None:
        """The most specific heading known for this chunk.

        Citations should read ``§4.2 Main Results`` rather than the parent
        ``§4 Experiments`` whenever the subsection was detected.
        """
        return self.subsection or self.section

    @property
    def location(self) -> str:
        parts = [f"p. {self.page}"]
        if self.citation_section:
            parts.append(f"§{self.citation_section}")
        if self.element_id:
            parts.append(self.element_id)
        return ", ".join(parts)


class PaperMetadata(StrictModel):
    """Bibliographic and structural facts about the ingested document."""

    document_id: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    doi: str | None = None
    keywords: list[str] = Field(default_factory=list)

    filename: str | None = None
    page_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    ocr_page_count: int = Field(default=0, ge=0)
    page_numbering: Literal["pdf_index", "printed_label"] = "pdf_index"

    sections: list[str] = Field(default_factory=list)
    element_counts: dict[str, int] = Field(default_factory=dict)
    has_equations: bool = False
    has_tables: bool = False
    has_figures: bool = False
    equation_count: int = Field(default=0, ge=0)
    table_count: int = Field(default=0, ge=0)
    figure_count: int = Field(default=0, ge=0)

    file_size_bytes: int = Field(default=0, ge=0)
    sha256: str | None = None
    ingested_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Evidence and findings
# ---------------------------------------------------------------------------


class Evidence(StrictModel):
    """A pointer back into the paper supporting exactly one statement.

    ``quote`` must be copied from retrieved paper text. Deterministic validation
    (see :mod:`paperpilot.validation`) re-checks every quote against the index
    and downgrades ``citation_quality`` / ``confidence`` when it cannot be found
    verbatim, rather than accepting the citation silently.
    """

    page: int = Field(ge=1)
    section: str | None = None
    source_type: ElementType = ElementType.UNKNOWN
    element_id: str | None = None
    quote: str
    chunk_id: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    citation_quality: CitationQuality = CitationQuality.UNVERIFIABLE
    validation_notes: list[str] = Field(default_factory=list)
    #: Whose statement the quoted sentences are: the paper's ``own``, a ``cited``
    #: work's (the paper reporting another work), ``mixed`` or ``unknown``. Set
    #: deterministically by :mod:`paperpilot.discourse`; ``None`` until then.
    attribution: Attribution | None = None
    #: The cited works a ``cited``/``mixed`` quote reports, e.g. ``["[20]"]``.
    cited_sources: list[str] = Field(default_factory=list)

    @field_validator("quote")
    @classmethod
    def _quote_not_empty(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if not cleaned:
            raise ValueError("evidence quote must not be empty")
        return cleaned

    @property
    def is_verified(self) -> bool:
        return self.citation_quality in (
            CitationQuality.EXACT,
            CitationQuality.APPROXIMATE,
            CitationQuality.NORMALIZED,
        )


class Finding(StrictModel):
    """One factual statement produced by an agent, with its evidence."""

    claim: str
    explanation: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: FindingStatus = FindingStatus.UNKNOWN
    category: str | None = None
    agent: str | None = None
    finding_id: str | None = None
    impact: float = Field(default=0.5, ge=0.0, le=1.0)
    verification_note: str | None = None
    #: Attribution of the finding's verified evidence taken together; see
    #: :attr:`Evidence.attribution`.
    attribution: Attribution | None = None
    cited_sources: list[str] = Field(default_factory=list)

    @field_validator("claim")
    @classmethod
    def _claim_not_empty(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if not cleaned:
            raise ValueError("finding claim must not be empty")
        return cleaned

    @property
    def has_evidence(self) -> bool:
        return any(e.is_verified for e in self.evidence)


# ---------------------------------------------------------------------------
# Agent results
# ---------------------------------------------------------------------------


class AgentUsage(StrictModel):
    """Execution metadata recorded in traces for every agent invocation."""

    agent: str
    prompt_version: str = "v0"
    model: str = "unknown"
    provider: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    attempt: int = 1
    estimated_cost_usd: float | None = None


class AgentResult(StrictModel):
    """Uniform envelope returned by every specialist agent."""

    agent: str
    status: Literal["ok", "skipped", "failed"] = "ok"
    findings: list[Finding] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list)
    usage: AgentUsage | None = None
    skip_reason: str | None = None
    error: str | None = None
    attempt: int = 1
    #: The model's answer exactly as written (Markdown).
    response_markdown: str | None = None
    #: That answer parsed into a ``paperpilot.response.ResponseDocument``.
    response_document: dict[str, Any] | None = None
    #: Every chunk shown to the model, in prompt order. The coverage sweep reads
    #: whatever no agent was shown.
    context_chunk_ids: list[str] = Field(default_factory=list)

    @property
    def ran(self) -> bool:
        return self.status == "ok"


class SummaryResult(StrictModel):
    """Structured payload of the Summary Agent."""

    research_problem: str = ""
    motivation: str = ""
    main_contribution: str = ""
    dataset: str = ""
    proposed_method: str = ""
    main_results: str = ""
    limitations: str = ""
    assumptions: str = ""
    concise_summary: str = ""
    unresolved_questions: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)


class MethodologyStep(StrictModel):
    order: int = Field(ge=1)
    name: str
    description: str
    evidence: list[Evidence] = Field(default_factory=list)
    is_inferred: bool = False


class MethodologyResult(StrictModel):
    """Structured payload of the Methodology Agent."""

    inputs: str = ""
    outputs: str = ""
    architecture: str = ""
    training_procedure: str = ""
    inference_procedure: str = ""
    design_decisions: list[str] = Field(default_factory=list)
    comparison_with_prior_work: str = ""
    pipeline_steps: list[MethodologyStep] = Field(default_factory=list)
    pseudocode: str | None = None
    findings: list[Finding] = Field(default_factory=list)


class MetricResult(StrictModel):
    """One reported number, always carrying its own evidence."""

    metric: str
    value: str
    system: str | None = None
    dataset: str | None = None
    split: str | None = None
    is_best: bool = False
    evidence: list[Evidence] = Field(default_factory=list)


class ExperimentResult(StrictModel):
    """Structured payload of the Experiment Agent."""

    datasets: list[str] = Field(default_factory=list)
    splits: str = ""
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    reported_results: list[MetricResult] = Field(default_factory=list)
    ablations: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, str] = Field(default_factory=dict)
    error_analysis: str = ""
    statistical_significance: str = ""
    conclusions_supported: FindingStatus = FindingStatus.UNKNOWN
    conclusions_assessment: str = ""
    findings: list[Finding] = Field(default_factory=list)


class CoverageDetail(StrictModel):
    """A setting or value stated in a part of the paper no specialist read."""

    name: str
    value: str
    evidence: list[Evidence] = Field(default_factory=list)
    value_found: bool = False


class CoverageResult(StrictModel):
    """Structured payload of the coverage sweep."""

    details: list[CoverageDetail] = Field(default_factory=list)
    #: Paper sections the sweep read, in document order.
    sections_read: list[str] = Field(default_factory=list)
    content_chunks: int = 0
    read_by_specialists: int = 0
    read_by_sweep: int = 0
    #: Sections with content that no model read, because the sweep budget ran out.
    unread_sections: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)


class EquationExplanation(StrictModel):
    """One equation, preserved in a safe representation.

    ``representation`` records how faithfully the source equation survived
    extraction. An unreadable equation is marked ``unavailable`` and is never
    silently reconstructed.
    """

    equation_id: str | None = None
    page: int = Field(ge=1)
    source_text: str
    representation: Literal["latex", "extracted_text", "unavailable"] = "extracted_text"
    purpose: str = ""
    variables: dict[str, str] = Field(default_factory=dict)
    shapes: dict[str, str] = Field(default_factory=dict)
    derivation_stated_by_authors: str = ""
    agent_interpretation: str = ""
    interpretation_differs: bool = False
    intuition: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class MathResult(StrictModel):
    """Structured payload of the conditional Math Agent."""

    equations: list[EquationExplanation] = Field(default_factory=list)
    loss_functions: list[str] = Field(default_factory=list)
    optimization_objective: str = ""
    notation: dict[str, str] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)


class VerificationResult(StrictModel):
    """Verdict of the Verification Agent for a single high-impact claim."""

    claim: str
    finding_id: str | None = None
    source_agent: str | None = None
    status: FindingStatus = FindingStatus.UNKNOWN
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    contradicting_evidence: list[Evidence] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    citation_quality: CitationQuality = CitationQuality.UNVERIFIABLE
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = ""
    numeric_checks: list[NumericCheck] = Field(default_factory=list)
    deterministic_notes: list[str] = Field(default_factory=list)
    attempt: int = 1
    response_markdown: str | None = None
    response_document: dict[str, Any] | None = None

    @property
    def is_weak(self) -> bool:
        return self.status in (
            FindingStatus.UNSUPPORTED,
            FindingStatus.CONTRADICTED,
            FindingStatus.UNKNOWN,
        )


class NumericCheck(StrictModel):
    """Deterministic comparison of a number in a claim against the paper text."""

    value: str
    found_in_evidence: bool
    matched_text: str | None = None
    page: int | None = None
    note: str | None = None


class ClaimCandidate(StrictModel):
    """A finding selected for verification, with the reason it was selected."""

    claim: str
    finding_id: str | None = None
    source_agent: str
    impact: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    existing_evidence: list[Evidence] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class AnalysisPlan(StrictModel):
    """The Orchestrator's conditional plan for one paper."""

    run_summary: bool = True
    run_methodology: bool = True
    run_experiments: bool = True
    run_math: bool = False
    run_table_analysis: bool = False
    priority_sections: list[str] = Field(default_factory=list)
    verification_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_retries: int = Field(default=1, ge=0, le=3)
    rationale: dict[str, str] = Field(default_factory=dict)
    created_by: Literal["heuristic", "llm", "hybrid"] = "heuristic"
    #: What kind of paper this is, decided from its text (see
    #: :mod:`paperpilot.discourse.profile`), and the signals that decided it.
    paper_kind: PaperKind = "empirical"
    paper_kind_signals: list[str] = Field(default_factory=list)

    def enabled_agents(self) -> list[str]:
        mapping = {
            "summary": self.run_summary,
            "methodology": self.run_methodology,
            "experiments": self.run_experiments,
            "math": self.run_math,
        }
        return [name for name, enabled in mapping.items() if enabled]


# ---------------------------------------------------------------------------
# Analysis lifecycle
# ---------------------------------------------------------------------------


class ProgressEvent(StrictModel):
    """A user-facing progress record; the UI renders these verbatim."""

    stage: str
    status: str
    message: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=_utcnow)


class AnalysisRecord(StrictModel):
    """Everything known about one analysis run, persisted between polls."""

    analysis_id: str
    document_id: str
    status: AnalysisStatus = AnalysisStatus.PENDING
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None

    plan: AnalysisPlan | None = None
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    events: list[ProgressEvent] = Field(default_factory=list)
    retry_count: int = 0
    retried_agents: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    report_path: str | None = None
    report_markdown: str | None = None
    #: The two-page brief; absent on analyses made before briefs existed.
    brief_markdown: str | None = None
    usage: list[AgentUsage] = Field(default_factory=list)
    duration_seconds: float | None = None

    def stats(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for finding in self.findings:
            statuses[finding.status.value] = statuses.get(finding.status.value, 0) + 1
        return {
            "findings": len(self.findings),
            "verified_claims": len(self.verification_results),
            "status_counts": statuses,
            "agents_run": [a for a, r in self.agent_results.items() if r.ran],
            "agents_skipped": [a for a, r in self.agent_results.items() if r.status == "skipped"],
            "retry_count": self.retry_count,
            "total_tokens": sum(u.total_tokens for u in self.usage),
        }


# Resolve the forward reference from VerificationResult -> NumericCheck.
VerificationResult.model_rebuild()
