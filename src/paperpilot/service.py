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

from .config import Settings, get_settings
from .domain.enums import AnalysisStatus
from .domain.models import AnalysisRecord, PaperMetadata
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
from .storage.files import LocalFileStorage
from .storage.repository import AnalysisRepository, DocumentRepository, FileStorage
from .storage.sqlite_repo import (
    SQLiteAnalysisRepository,
    SQLiteConnection,
    SQLiteDocumentRepository,
)


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
        llm_provider=None,
        embedding_provider=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        configure_logging(self.settings.log_level, self.settings.log_format)

        if documents is None or analyses is None:
            connection = SQLiteConnection(self.settings.sqlite_path)
            documents = documents or SQLiteDocumentRepository(connection)
            analyses = analyses or SQLiteAnalysisRepository(connection)

        self.documents = documents
        self.analyses = analyses
        self.files = files or LocalFileStorage(
            upload_dir=self.settings.upload_dir, report_dir=self.settings.report_dir
        )
        self.llm = llm_provider or build_llm_provider(self.settings)
        self.embedder = embedding_provider or build_embedding_provider(self.settings)
        self._lock = threading.RLock()
        #: Paths of files analysed in place (CLI), which are not copied into
        #: the upload directory.
        self._local_paths: dict[str, Path] = {}

    # --- documents --------------------------------------------------------

    def register_upload(self, stream: BinaryIO, *, filename: str) -> PaperMetadata:
        """Persist an uploaded PDF and return a stub metadata record.

        Parsing happens during analysis, not upload, so a large PDF does not block
        the HTTP request. Only the cheap deterministic checks run here.
        """
        document_id = new_document_id()
        path = self.files.save_upload(stream, filename=filename, document_id=document_id)
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
        self.documents.save(metadata)
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

    def get_document(self, document_id: str) -> PaperMetadata:
        metadata = self.documents.get(document_id)
        if metadata is None:
            raise NotFoundError(f"unknown document: {document_id}")
        return metadata

    def list_documents(self, *, limit: int = 50) -> list[PaperMetadata]:
        return self.documents.list(limit=limit)

    def resolve_path(self, document_id: str) -> Path:
        path = self._local_paths.get(document_id) or self.files.path_for(document_id)
        if path is None or not Path(path).exists():
            raise NotFoundError(f"no stored file for document '{document_id}'")
        return Path(path)

    # --- analyses ---------------------------------------------------------

    def create_analysis(self, document_id: str) -> AnalysisRecord:
        self.get_document(document_id)
        record = AnalysisRecord(
            analysis_id=new_analysis_id(),
            document_id=document_id,
            status=AnalysisStatus.PENDING,
        )
        self.analyses.save(record)
        return record

    def get_analysis(self, analysis_id: str) -> AnalysisRecord:
        record = self.analyses.get(analysis_id)
        if record is None:
            raise NotFoundError(f"unknown analysis: {analysis_id}")
        return record

    def get_report(self, analysis_id: str) -> str:
        record = self.get_analysis(analysis_id)
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
        file_path = self.resolve_path(record.document_id)
        started = time.perf_counter()

        recorder = EventRecorder(sink=event_sink)
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

        with log_context(analysis_id=analysis_id, document_id=record.document_id):
            final = workflow.run(state)

        return self._persist(record, final, started)

    def analyze_file(
        self, file_path: str | Path, *, event_sink: EventSink | None = None
    ) -> AnalysisRecord:
        """Convenience path used by the CLI: register, analyse, report."""
        metadata = self.register_local_file(file_path)
        record = self.create_analysis(metadata.document_id)
        return self.run_analysis(record.analysis_id, event_sink=event_sink)

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
                record.report_markdown = report
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
            duration_s=record.duration_seconds,
        )
        return record
