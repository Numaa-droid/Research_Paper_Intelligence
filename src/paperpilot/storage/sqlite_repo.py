"""SQLite-backed repositories.

Records are stored as JSON blobs beside a few indexed columns. For a
single-paper workflow that is the right trade: it keeps the schema stable while
the domain models evolve, and the queries that matter (by id, by hash, recent
first) are still index-backed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ..domain.models import AnalysisRecord, PaperMetadata
from ..observability import debug
from .repository import AnalysisRepository, DocumentRepository

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title       TEXT,
    filename    TEXT,
    sha256      TEXT,
    page_count  INTEGER NOT NULL DEFAULT 0,
    ingested_at TEXT    NOT NULL,
    payload     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_documents_ingested ON documents(ingested_at DESC);

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id  TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    report_path  TEXT,
    payload      TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);
CREATE INDEX IF NOT EXISTS idx_analyses_document ON analyses(document_id, created_at DESC);
"""


class SQLiteConnection:
    """Thread-safe connection holder.

    The API runs analyses in a background thread pool, so the connection is
    guarded by a lock and opened with ``check_same_thread=False``.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        debug("storage.sqlite_ready", path=str(path))

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SQLiteDocumentRepository(DocumentRepository):
    def __init__(self, connection: SQLiteConnection) -> None:
        self.db = connection

    def save(self, metadata: PaperMetadata) -> None:
        self.db.execute(
            """
            INSERT INTO documents
                (document_id, title, filename, sha256, page_count, ingested_at, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                title=excluded.title, filename=excluded.filename, sha256=excluded.sha256,
                page_count=excluded.page_count, payload=excluded.payload
            """,
            (
                metadata.document_id,
                metadata.title,
                metadata.filename,
                metadata.sha256,
                metadata.page_count,
                metadata.ingested_at.isoformat(),
                metadata.model_dump_json(),
            ),
        )

    def get(self, document_id: str) -> PaperMetadata | None:
        rows = self.db.query(
            "SELECT payload FROM documents WHERE document_id = ?", (document_id,)
        )
        return PaperMetadata.model_validate_json(rows[0]["payload"]) if rows else None

    def list(self, *, limit: int = 50, offset: int = 0) -> list[PaperMetadata]:
        rows = self.db.query(
            "SELECT payload FROM documents ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [PaperMetadata.model_validate_json(r["payload"]) for r in rows]

    def find_by_hash(self, sha256: str) -> PaperMetadata | None:
        rows = self.db.query(
            "SELECT payload FROM documents WHERE sha256 = ? ORDER BY ingested_at DESC LIMIT 1",
            (sha256,),
        )
        return PaperMetadata.model_validate_json(rows[0]["payload"]) if rows else None


class SQLiteAnalysisRepository(AnalysisRepository):
    def __init__(self, connection: SQLiteConnection) -> None:
        self.db = connection

    def save(self, record: AnalysisRecord) -> None:
        self.db.execute(
            """
            INSERT INTO analyses
                (analysis_id, document_id, status, created_at, updated_at, report_path, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(analysis_id) DO UPDATE SET
                status=excluded.status, updated_at=excluded.updated_at,
                report_path=excluded.report_path, payload=excluded.payload
            """,
            (
                record.analysis_id,
                record.document_id,
                record.status.value,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.report_path,
                record.model_dump_json(),
            ),
        )

    def get(self, analysis_id: str) -> AnalysisRecord | None:
        rows = self.db.query(
            "SELECT payload FROM analyses WHERE analysis_id = ?", (analysis_id,)
        )
        return AnalysisRecord.model_validate_json(rows[0]["payload"]) if rows else None

    def list_for_document(self, document_id: str, *, limit: int = 20) -> list[AnalysisRecord]:
        rows = self.db.query(
            "SELECT payload FROM analyses WHERE document_id = ? ORDER BY created_at DESC LIMIT ?",
            (document_id, limit),
        )
        return [AnalysisRecord.model_validate_json(r["payload"]) for r in rows]


class InMemoryDocumentRepository(DocumentRepository):
    """Used by tests and by the CLI when persistence is not wanted."""

    def __init__(self) -> None:
        self._items: dict[str, PaperMetadata] = {}

    def save(self, metadata: PaperMetadata) -> None:
        self._items[metadata.document_id] = metadata

    def get(self, document_id: str) -> PaperMetadata | None:
        return self._items.get(document_id)

    def list(self, *, limit: int = 50, offset: int = 0) -> list[PaperMetadata]:
        items = sorted(self._items.values(), key=lambda m: m.ingested_at, reverse=True)
        return items[offset : offset + limit]

    def find_by_hash(self, sha256: str) -> PaperMetadata | None:
        return next((m for m in self._items.values() if m.sha256 == sha256), None)


class InMemoryAnalysisRepository(AnalysisRepository):
    def __init__(self) -> None:
        self._items: dict[str, AnalysisRecord] = {}

    def save(self, record: AnalysisRecord) -> None:
        self._items[record.analysis_id] = record

    def get(self, analysis_id: str) -> AnalysisRecord | None:
        return self._items.get(analysis_id)

    def list_for_document(self, document_id: str, *, limit: int = 20) -> list[AnalysisRecord]:
        items = [r for r in self._items.values() if r.document_id == document_id]
        items.sort(key=lambda r: r.created_at, reverse=True)
        return items[:limit]


def json_dumps(payload: dict) -> str:
    return json.dumps(payload, default=str)
