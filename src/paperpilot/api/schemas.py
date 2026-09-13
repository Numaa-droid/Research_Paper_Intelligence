"""API request and response schemas.

Kept separate from the domain models so the HTTP contract can stay stable while
the internal models evolve.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..domain.enums import AnalysisStatus
from ..domain.models import AnalysisRecord, PaperMetadata


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    llm_provider: str
    llm_model: str
    embedding_provider: str
    ocr_available: bool
    warnings: list[str] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    document_id: str
    filename: str | None = None
    title: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    ocr_page_count: int = 0
    sections: list[str] = Field(default_factory=list)
    element_counts: dict[str, int] = Field(default_factory=dict)
    file_size_bytes: int = 0
    page_numbering: str = "pdf_index"
    ingested_at: datetime | None = None

    @classmethod
    def from_metadata(cls, metadata: PaperMetadata) -> DocumentResponse:
        return cls(
            document_id=metadata.document_id,
            filename=metadata.filename,
            title=metadata.title,
            page_count=metadata.page_count,
            chunk_count=metadata.chunk_count,
            ocr_page_count=metadata.ocr_page_count,
            sections=metadata.sections,
            element_counts=metadata.element_counts,
            file_size_bytes=metadata.file_size_bytes,
            page_numbering=metadata.page_numbering,
            ingested_at=metadata.ingested_at,
        )


class ProgressEventResponse(BaseModel):
    stage: str
    status: str
    message: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
    at: datetime


class AnalysisResponse(BaseModel):
    """Poll-friendly analysis status."""

    analysis_id: str
    document_id: str
    status: AnalysisStatus
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    #: Coarse progress so a client can render a bar without parsing events.
    stage: str = "pending"
    progress: float = 0.0

    plan: dict[str, Any] | None = None
    agents: dict[str, dict[str, Any]] = Field(default_factory=dict)
    finding_counts: dict[str, int] = Field(default_factory=dict)
    verified_claims: int = 0
    retry_count: int = 0
    errors: list[str] = Field(default_factory=list)
    events: list[ProgressEventResponse] = Field(default_factory=list)
    report_url: str | None = None

    @classmethod
    def from_record(cls, record: AnalysisRecord) -> AnalysisResponse:
        stats = record.stats()
        last_event = record.events[-1] if record.events else None
        return cls(
            analysis_id=record.analysis_id,
            document_id=record.document_id,
            status=record.status,
            created_at=record.created_at,
            updated_at=record.updated_at,
            completed_at=record.completed_at,
            duration_seconds=record.duration_seconds,
            stage=last_event.stage if last_event else "pending",
            progress=_progress(record),
            plan=record.plan.model_dump(mode="json") if record.plan else None,
            agents={
                name: {
                    "status": result.status,
                    "findings": len(result.findings),
                    "attempt": result.attempt,
                    "skip_reason": result.skip_reason,
                    "error": result.error,
                }
                for name, result in record.agent_results.items()
            },
            finding_counts=stats["status_counts"],
            verified_claims=stats["verified_claims"],
            retry_count=record.retry_count,
            errors=record.errors,
            events=[
                ProgressEventResponse(
                    stage=e.stage,
                    status=e.status,
                    message=e.message,
                    detail=e.detail,
                    at=e.at,
                )
                for e in record.events
            ],
            report_url=(
                f"/analyses/{record.analysis_id}/report"
                if record.status is AnalysisStatus.COMPLETED
                else None
            ),
        )


#: Rough fraction of the workflow complete when each stage finishes.
_STAGE_PROGRESS = {
    "ingestion": 0.15,
    "indexing": 0.25,
    "planning": 0.3,
    "agent:summary": 0.45,
    "agent:methodology": 0.5,
    "agent:experiments": 0.55,
    "agent:math": 0.6,
    "aggregation": 0.65,
    "verification": 0.85,
    "retry": 0.9,
    "report": 1.0,
}


def _progress(record: AnalysisRecord) -> float:
    if record.status is AnalysisStatus.COMPLETED:
        return 1.0
    if record.status is AnalysisStatus.FAILED:
        return 0.0
    best = 0.0
    for event in record.events:
        if event.status in ("completed", "skipped"):
            best = max(best, _STAGE_PROGRESS.get(event.stage, 0.0))
    return round(best, 3)


class UploadResponse(BaseModel):
    document_id: str
    filename: str | None
    file_size_bytes: int
    message: str = "Upload accepted. Parsing happens when analysis starts."


class AnalyzeRequest(BaseModel):
    """Options for starting an analysis."""

    wait: bool = Field(
        default=False,
        description=(
            "Run synchronously and return the completed analysis. Use only for small "
            "papers; prefer the default background mode with polling."
        ),
    )


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
