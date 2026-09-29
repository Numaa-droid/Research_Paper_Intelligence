"""Repository interfaces.

The MVP persists to SQLite and the local filesystem, but nothing above this layer
knows that. Swapping in PostgreSQL or object storage means implementing these
protocols, not editing the workflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from ..domain.models import AnalysisRecord, PaperMetadata


class DocumentRepository(ABC):
    """Stores paper metadata."""

    @abstractmethod
    def save(self, metadata: PaperMetadata) -> None: ...

    @abstractmethod
    def get(self, document_id: str) -> PaperMetadata | None: ...

    @abstractmethod
    def list(self, *, limit: int = 50, offset: int = 0) -> list[PaperMetadata]: ...

    @abstractmethod
    def find_by_hash(self, sha256: str) -> PaperMetadata | None: ...


class AnalysisRepository(ABC):
    """Stores analysis runs and their reports."""

    @abstractmethod
    def save(self, record: AnalysisRecord) -> None: ...

    @abstractmethod
    def get(self, analysis_id: str) -> AnalysisRecord | None: ...

    @abstractmethod
    def list_for_document(self, document_id: str, *, limit: int = 20) -> list[AnalysisRecord]: ...


class FileStorage(ABC):
    """Stores the uploaded PDFs and generated reports."""

    @abstractmethod
    def save_upload(self, stream: BinaryIO, *, filename: str, document_id: str) -> Path: ...

    @abstractmethod
    def path_for(self, document_id: str) -> Path | None: ...

    @abstractmethod
    def save_report(self, analysis_id: str, markdown: str) -> Path: ...

    @abstractmethod
    def read_report(self, analysis_id: str) -> str | None: ...
