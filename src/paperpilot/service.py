"""Application service.

The single entry point used by both the API and the CLI. It owns the lifecycle
(upload -> analyse -> report), wires the adapters together, and persists the
analysis record so a long-running run stays pollable.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from .accounts import AccountService
from .config import Settings, get_settings
from .domain.enums import TERMINAL_STATUSES, AnalysisStatus
from .domain.models import AnalysisRecord, PaperMetadata, ProgressEvent
from .domain.state import ResearchState, new_state
from .errors import NotFoundError, ValidationError
from .ingestion.pdf_parser import validate_pdf_file
from .ingestion.service import IngestionService, new_document_id
from .observability import configure_logging, info, log_context
from .orchestration.events import EventRecorder, EventSink
from .orchestration.graph import AnalysisWorkflow, WorkflowResources
from .providers.embeddings import build_embedding_provider
from .providers.llm import build_llm_provider
from .reporting.synthesizer import ReportSynthesizer
from .storage.factory import build_repositories
from .storage.files import LocalFileStorage
from .storage.repository import AnalysisRepository, DocumentRepository, FileStorage, UserRepository
from .storage.sqlite_repo import InMemoryUserRepository
from .tracing import Tracer
from .tracing.store import SQLiteTraceStore


def new_analysis_id() -> str:
    return f"analysis-{uuid.uuid4().hex[:12]}"


class PaperPilotService:
    """Orchestrates ingestion, analysis and report storage."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        documents: DocumentRepository | None = None,
        analyses: AnalysisRepository | None = None,
        files: FileStorage | None = None,
        users: UserRepository | None = None,
        llm_provider=None,
        embedding_provider=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        configure_logging(
            self.settings.log_level, self.settings.log_format, http=self.settings.log_http
        )

        if documents is None or analyses is None:
            built_documents, built_analyses, built_users = build_repositories(self.settings)
            documents = documents or built_documents
            analyses = analyses or built_analyses
            users = users or built_users
        users = users or InMemoryUserRepository()

        self.documents = documents
        self.analyses = analyses
        self.accounts = AccountService(users, documents, settings=self.settings)
        self.files = files or LocalFileStorage(
            upload_dir=self.settings.upload_dir, report_dir=self.settings.report_dir
        )
        self.tracer = Tracer(
            SQLiteTraceStore(self.settings.trace_db_path, retention=self.settings.tracing_retention)
            if self.settings.tracing_enabled
            else None,
            enabled=self.settings.tracing_enabled,
            capture_content=self.settings.tracing_capture_content,
        )
        self.llm = llm_provider or build_llm_provider(self.settings)
        self.embedder = embedding_provider or build_embedding_provider(self.settings)
        self._lock = threading.RLock()
        #: Paths of files analysed in place (CLI), which are not copied into
        #: the upload directory.
        self._local_paths: dict[str, Path] = {}
        #: Events of runs still in flight. The record is only persisted when a
        #: run finishes, so polling merges these in to show live progress.
        self._live_events: dict[str, list[ProgressEvent]] = {}

    # --- documents --------------------------------------------------------

    def register_upload(
        self, stream: BinaryIO, *, filename: str, owner_id: str | None = None
    ) -> PaperMetadata:
        """Persist an uploaded PDF and return a stub metadata record.

        Parsing happens during analysis, not upload, so a large PDF does not block
        the HTTP request. Only the cheap deterministic checks run here.
        """
        document_id = new_document_id()
        path = self.files.save_upload(
            stream,
            filename=filename,
            document_id=document_id,
            max_bytes=self.settings.max_upload_bytes,
        )
        try:
            size = validate_pdf_file(path, max_bytes=self.settings.max_upload_bytes)
        except ValidationError:
            path.unlink(missing_ok=True)
            raise

        metadata = PaperMetadata(
            document_id=document_id,
            filename=filename,
            file_size_bytes=size,
        )
        self.documents.save(metadata, owner_id=owner_id)
        info("service.upload_registered", document_id=document_id, bytes=size)
        return metadata

    def register_local_file(self, file_path: str | Path) -> PaperMetadata:
        """Register a file already on disk (the CLI path); the file is not copied."""
        path = Path(file_path).expanduser().resolve()
        size = validate_pdf_file(path, max_bytes=self.settings.max_upload_bytes)
        document_id = new_document_id()
        metadata = PaperMetadata(
            document_id=document_id, filename=path.name, file_size_bytes=size
        )
        self.documents.save(metadata)
        self._local_paths[document_id] = path
        return metadata

    def get_document(self, document_id: str, *, owner_id: str | None = None) -> PaperMetadata:
        """The paper's metadata; another account's paper is reported as unknown."""
        metadata = self.documents.get(document_id)
        if metadata is None or (
            owner_id is not None and self.documents.owner_of(document_id) != owner_id
        ):
            raise NotFoundError(f"unknown document: {document_id}")
        return metadata

    def list_documents(self, *, limit: int = 50, owner_id: str | None = None) -> list[PaperMetadata]:
        return self.documents.list(limit=limit, owner_id=owner_id)

    def list_analyses(self, *, owner_id: str | None = None, limit: int = 50) -> list[AnalysisRecord]:
        """Recent analyses of the account's papers, newest first."""
        papers = self.documents.list(limit=1000, owner_id=owner_id)
        records = self.analyses.list_for_documents([p.document_id for p in papers], limit=limit)
        return [self._with_live_status(r) for r in records]

    def resolve_path(self, document_id: str) -> Path:
        path = self._local_paths.get(document_id) or self.files.path_for(document_id)
        if path is None or not Path(path).exists():
            raise NotFoundError(f"no stored file for document '{document_id}'")
        return Path(path)

    # --- analyses ---------------------------------------------------------

    def create_analysis(self, document_id: str, *, owner_id: str | None = None) -> AnalysisRecord:
        self.get_document(document_id, owner_id=owner_id)
        # Refuse now rather than in the background task, where a missing file
        # would leave the run "pending" with nothing to say why.
        try:
            self.resolve_path(document_id)
        except NotFoundError:
            raise NotFoundError(
                f"the PDF for document '{document_id}' is no longer stored on this server; "
                "upload the paper again to analyse it"
            ) from None
        record = AnalysisRecord(
            analysis_id=new_analysis_id(),
            document_id=document_id,
            status=AnalysisStatus.PENDING,
        )
        self.analyses.save(record)
        return record

    def get_analysis(self, analysis_id: str, *, owner_id: str | None = None) -> AnalysisRecord:
        record = self.analyses.get(analysis_id)
        if record is None or (
            owner_id is not None and self.documents.owner_of(record.document_id) != owner_id
        ):
            raise NotFoundError(f"unknown analysis: {analysis_id}")
        return self._with_live_status(record)

    def _with_live_status(self, record: AnalysisRecord) -> AnalysisRecord:
        analysis_id = record.analysis_id
        with self._lock:
            live = list(self._live_events.get(analysis_id, ()))
        if live and record.status not in TERMINAL_STATUSES:
            record.events = live
            record.status = _status_for_stage(live[-1].stage, record.status)
        return record

    def get_report(self, analysis_id: str, *, owner_id: str | None = None) -> str:
        record = self.get_analysis(analysis_id, owner_id=owner_id)
        if record.report_markdown:
            return record.report_markdown
        markdown = self.files.read_report(analysis_id)
        if markdown is None:
            raise NotFoundError(
                f"analysis '{analysis_id}' has no report yet (status: {record.status.value})"
            )
        return markdown

    def run_analysis(
        self, analysis_id: str, *, event_sink: EventSink | None = None
    ) -> AnalysisRecord:
        """Execute the workflow for an existing analysis record."""
        record = self.get_analysis(analysis_id)
        try:
            file_path = self.resolve_path(record.document_id)
        except NotFoundError as exc:
            self._mark_failed(record, str(exc))
            raise
        started = time.perf_counter()

        with self._lock:
            live = self._live_events[analysis_id] = []

        def sink(event: ProgressEvent) -> None:
            # Specialist agents emit from worker threads.
            with self._lock:
                live.append(event)
            if event_sink is not None:
                event_sink(event)

        recorder = EventRecorder(sink=sink)
        resources = WorkflowResources(
            settings=self.settings,
            llm=self.llm,
            embedder=self.embedder,
            recorder=recorder,
            ingestion=IngestionService(self.settings, embedder=self.embedder),
            synthesizer=ReportSynthesizer(),
        )
        workflow = AnalysisWorkflow(resources)

        record.status = AnalysisStatus.INGESTING
        record.updated_at = datetime.now(UTC)
        self.analyses.save(record)

        state: ResearchState = new_state(
            analysis_id=analysis_id,
            document_id=record.document_id,
            file_path=str(file_path),
        )

        paper = file_path.name.split("__", 1)[-1]
        llm_model = None if self.settings.llm_provider == "mock" else self.settings.llm_model
        try:
            with log_context(analysis_id=analysis_id, document_id=record.document_id), \
                    self.tracer.trace(
                        "analysis",
                        analysis_id=analysis_id,
                        document_id=record.document_id,
                        inputs={"paper": paper, "document_id": record.document_id},
                        metadata={
                            "llm_provider": self.settings.llm_provider,
                            "llm_model": self.llm.model,
                            "embedding_provider": self.settings.embedding_provider,
                            "parallel_agents": self.settings.parallel_agents,
                            "max_retries": self.settings.max_retries,
                        },
                    ) as root:
                info(
                    "service.analysis_started",
                    paper=paper,
                    llm_provider=self.settings.llm_provider,
                    llm_model=llm_model,
                )
                final = workflow.run(state)
                status = final.get("status", AnalysisStatus.FAILED)
                root.set_outputs(
                    status=getattr(status, "value", status),
                    findings=len(final.get("findings", [])),
                    verified_claims=len(final.get("verification_results", [])),
                    retries=int(final.get("retry_count", 0)),
                    errors=final.get("errors", []),
                    report_chars=len(final.get("final_report") or ""),
                )
                if status is AnalysisStatus.FAILED:
                    root.fail("; ".join(final.get("errors", [])) or "the analysis failed")
            return self._persist(record, final, started)
        except Exception as exc:
            # A crash outside the workflow's own error handling must still end the
            # run, or clients poll a "pending" record forever.
            self._mark_failed(record, f"the analysis crashed: {exc}")
            raise
        finally:
            with self._lock:
                self._live_events.pop(analysis_id, None)

    def _mark_failed(self, record: AnalysisRecord, message: str) -> None:
        if record.status in TERMINAL_STATUSES:
            return
        now = datetime.now(UTC)
        record.status = AnalysisStatus.FAILED
        record.errors = [*record.errors, message]
        record.updated_at = now
        record.completed_at = now
        self.analyses.save(record)

    def analyze_file(
        self, file_path: str | Path, *, event_sink: EventSink | None = None
    ) -> AnalysisRecord:
        """Convenience path used by the CLI: register, analyse, report."""
        metadata = self.register_local_file(file_path)
        record = self.create_analysis(metadata.document_id)
        return self.run_analysis(record.analysis_id, event_sink=event_sink)

    # --- traces -----------------------------------------------------------

    def list_traces(self, *, owner_id: str | None = None, limit: int = 50) -> list[dict]:
        """Run traces of the account's analyses, running ones included, newest first."""
        analysis_ids = (
            None
            if owner_id is None
            else [r.analysis_id for r in self.list_analyses(owner_id=owner_id, limit=1000)]
        )
        return self.tracer.list(analysis_ids=analysis_ids, limit=limit)

    def get_trace(self, trace_id: str, *, owner_id: str | None = None) -> dict:
        """One trace with every span; another account's trace is reported as unknown."""
        trace = self.tracer.get(trace_id)
        if trace is None:
            raise NotFoundError(f"unknown trace: {trace_id}")
        if owner_id is not None:
            if not trace.get("analysis_id"):
                raise NotFoundError(f"unknown trace: {trace_id}")
            try:
                self.get_analysis(trace["analysis_id"], owner_id=owner_id)
            except NotFoundError:
                raise NotFoundError(f"unknown trace: {trace_id}") from None
        return trace

    def trace_for_analysis(self, analysis_id: str, *, owner_id: str | None = None) -> dict:
        """The most recent trace of an analysis."""
        self.get_analysis(analysis_id, owner_id=owner_id)
        trace = self.tracer.latest_for_analysis(analysis_id)
        if trace is None:
            raise NotFoundError(f"analysis '{analysis_id}' has no trace")
        return trace

    # --- internals --------------------------------------------------------

    def _persist(
        self, record: AnalysisRecord, state: ResearchState, started: float
    ) -> AnalysisRecord:
        with self._lock:
            record.status = state.get("status", AnalysisStatus.FAILED)
            record.plan = state.get("analysis_plan")
            record.agent_results = state.get("agent_results", {})
            record.findings = state.get("findings", [])
            record.verification_results = state.get("verification_results", [])
            record.events = state.get("events", [])
            record.usage = state.get("usage", [])
            record.errors = state.get("errors", [])
            record.retry_count = int(state.get("retry_count", 0))
            record.retried_agents = list(state.get("retry_targets_history", []))  # type: ignore[arg-type]
            record.duration_seconds = round(time.perf_counter() - started, 3)
            record.updated_at = datetime.now(UTC)

            report = state.get("final_report")
            if report:
                path = self.files.save_report(record.analysis_id, report)
                info("report.saved", analysis_id=record.analysis_id, path=str(path), chars=len(report))
                record.report_markdown = report
                record.brief_markdown = state.get("final_brief") or None
                record.report_path = str(path)
                record.completed_at = datetime.now(UTC)
                record.status = AnalysisStatus.COMPLETED
            elif record.status is not AnalysisStatus.FAILED:
                record.status = AnalysisStatus.FAILED
                record.errors.append("the workflow finished without producing a report")

            metadata = state.get("paper_metadata")
            if metadata is not None:
                self.documents.save(metadata)
            self.analyses.save(record)

        info(
            "service.analysis_finished",
            analysis_id=record.analysis_id,
            status=record.status.value,
            findings=len(record.findings),
            verified_claims=len(record.verification_results),
            tokens=sum(u.total_tokens for u in record.usage),
            cost_usd=round(sum(u.estimated_cost_usd or 0.0 for u in record.usage), 6) or None,
            retries=record.retry_count,
            duration_s=record.duration_seconds,
        )
        return record


_STAGE_STATUS = {
    "ingestion": AnalysisStatus.INGESTING,
    "indexing": AnalysisStatus.INDEXING,
    "planning": AnalysisStatus.PLANNING,
    "aggregation": AnalysisStatus.ANALYZING,
    "verification": AnalysisStatus.VERIFYING,
    "retry": AnalysisStatus.RETRYING,
    "report": AnalysisStatus.SYNTHESIZING,
}


def _status_for_stage(stage: str, default: AnalysisStatus) -> AnalysisStatus:
    """Coarse run status implied by the most recent progress event."""
    if stage.startswith("agent:"):
        return AnalysisStatus.ANALYZING
    return _STAGE_STATUS.get(stage, default)
