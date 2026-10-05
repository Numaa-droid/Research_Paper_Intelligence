"""Repository interfaces.

The MVP persists to SQLite and the local filesystem, but nothing above this layer
knows that. Swapping in PostgreSQL or object storage means implementing these
protocols, not editing the workflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from ..domain.models import AnalysisRecord, PaperMetadata


class DocumentRepository(ABC):
    """Stores paper metadata and which account owns each paper.

    Ownership lives beside the metadata, not in it: an analysis re-saves the
    metadata it parsed, and that must never reset who the paper belongs to.
    ``owner_id=None`` on a read means "any owner" (the CLI, which has no users).
    """

    @abstractmethod
    def save(self, metadata: PaperMetadata, *, owner_id: str | None = None) -> None: ...

    @abstractmethod
    def get(self, document_id: str) -> PaperMetadata | None: ...

    @abstractmethod
    def list(
        self, *, limit: int = 50, offset: int = 0, owner_id: str | None = None
    ) -> list[PaperMetadata]: ...

    @abstractmethod
    def find_by_hash(self, sha256: str) -> PaperMetadata | None: ...

    @abstractmethod
    def owner_of(self, document_id: str) -> str | None: ...

    @abstractmethod
    def claim_unowned(self, owner_id: str) -> int:
        """Give every paper without an owner to ``owner_id``; return how many."""


@dataclass(slots=True)
class User:
    user_id: str
    email: str
    name: str
    password_hash: str
    created_at: datetime


class UserRepository(ABC):
    """Accounts and their login sessions. Sessions are stored by token hash only."""

    @abstractmethod
    def create(self, user: User) -> None: ...

    @abstractmethod
    def get(self, user_id: str) -> User | None: ...

    @abstractmethod
    def get_by_email(self, email: str) -> User | None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def update_password(self, user_id: str, password_hash: str) -> None: ...

    @abstractmethod
    def create_session(self, token_hash: str, user_id: str, expires_at: datetime) -> None: ...

    @abstractmethod
    def session_user(self, token_hash: str, now: datetime) -> User | None: ...

    @abstractmethod
    def delete_session(self, token_hash: str) -> None: ...

    @abstractmethod
    def delete_sessions(self, user_id: str, *, keep: str | None = None) -> None:
        """End every session of ``user_id`` except the one hashed as ``keep``."""


class AnalysisRepository(ABC):
    """Stores analysis runs and their reports."""

    @abstractmethod
    def save(self, record: AnalysisRecord) -> None: ...

    @abstractmethod
    def get(self, analysis_id: str) -> AnalysisRecord | None: ...

    @abstractmethod
    def list_for_document(self, document_id: str, *, limit: int = 20) -> list[AnalysisRecord]: ...

    @abstractmethod
    def list_for_documents(self, document_ids: list[str], *, limit: int = 50) -> list[AnalysisRecord]:
        """Most recent analyses of any of ``document_ids``, newest first."""


class FileStorage(ABC):
    """Stores the uploaded PDFs and generated reports."""

    @abstractmethod
    def save_upload(
        self, stream: BinaryIO, *, filename: str, document_id: str, max_bytes: int | None = None
    ) -> Path: ...

    @abstractmethod
    def path_for(self, document_id: str) -> Path | None: ...

    @abstractmethod
    def save_report(self, analysis_id: str, markdown: str) -> Path: ...

    @abstractmethod
    def read_report(self, analysis_id: str) -> str | None: ...
