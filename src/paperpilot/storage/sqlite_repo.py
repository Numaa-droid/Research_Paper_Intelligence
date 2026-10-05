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
from datetime import datetime
from pathlib import Path

from ..domain.models import AnalysisRecord, PaperMetadata
from ..errors import ConflictError
from ..observability import debug
from .repository import AnalysisRepository, DocumentRepository, User, UserRepository

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

CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""

#: Columns added after the first release, applied to existing databases.
_MIGRATIONS = (
    ("documents", "owner_id", "ALTER TABLE documents ADD COLUMN owner_id TEXT"),
)


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
            for table, column, ddl in _MIGRATIONS:
                columns = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
                if column not in columns:
                    self._conn.execute(ddl)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id, ingested_at DESC)"
            )
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

    def save(self, metadata: PaperMetadata, *, owner_id: str | None = None) -> None:
        self.db.execute(
            """
            INSERT INTO documents
                (document_id, title, filename, sha256, page_count, ingested_at, payload, owner_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                title=excluded.title, filename=excluded.filename, sha256=excluded.sha256,
                page_count=excluded.page_count, payload=excluded.payload,
                owner_id=COALESCE(documents.owner_id, excluded.owner_id)
            """,
            (
                metadata.document_id,
                metadata.title,
                metadata.filename,
                metadata.sha256,
                metadata.page_count,
                metadata.ingested_at.isoformat(),
                metadata.model_dump_json(),
                owner_id,
            ),
        )

    def get(self, document_id: str) -> PaperMetadata | None:
        rows = self.db.query(
            "SELECT payload FROM documents WHERE document_id = ?", (document_id,)
        )
        return PaperMetadata.model_validate_json(rows[0]["payload"]) if rows else None

    def list(
        self, *, limit: int = 50, offset: int = 0, owner_id: str | None = None
    ) -> list[PaperMetadata]:
        if owner_id is None:
            rows = self.db.query(
                "SELECT payload FROM documents ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        else:
            rows = self.db.query(
                "SELECT payload FROM documents WHERE owner_id = ? "
                "ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
                (owner_id, limit, offset),
            )
        return [PaperMetadata.model_validate_json(r["payload"]) for r in rows]

    def owner_of(self, document_id: str) -> str | None:
        rows = self.db.query("SELECT owner_id FROM documents WHERE document_id = ?", (document_id,))
        return rows[0]["owner_id"] if rows else None

    def claim_unowned(self, owner_id: str) -> int:
        return self.db.execute(
            "UPDATE documents SET owner_id = ? WHERE owner_id IS NULL", (owner_id,)
        ).rowcount

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

    def list_for_documents(self, document_ids: list[str], *, limit: int = 50) -> list[AnalysisRecord]:
        if not document_ids:
            return []
        marks = ",".join("?" * len(document_ids))
        rows = self.db.query(
            f"SELECT payload FROM analyses WHERE document_id IN ({marks}) "
            "ORDER BY created_at DESC LIMIT ?",
            (*document_ids, limit),
        )
        return [AnalysisRecord.model_validate_json(r["payload"]) for r in rows]


def _user(row: sqlite3.Row) -> User:
    return User(
        user_id=row["user_id"],
        email=row["email"],
        name=row["name"],
        password_hash=row["password_hash"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class SQLiteUserRepository(UserRepository):
    def __init__(self, connection: SQLiteConnection) -> None:
        self.db = connection

    def create(self, user: User) -> None:
        try:
            self.db.execute(
                "INSERT INTO users (user_id, email, name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (user.user_id, user.email, user.name, user.password_hash, user.created_at.isoformat()),
            )
        except sqlite3.IntegrityError as exc:
            raise ConflictError("an account with this email already exists") from exc

    def get(self, user_id: str) -> User | None:
        rows = self.db.query("SELECT * FROM users WHERE user_id = ?", (user_id,))
        return _user(rows[0]) if rows else None

    def get_by_email(self, email: str) -> User | None:
        rows = self.db.query("SELECT * FROM users WHERE email = ?", (email,))
        return _user(rows[0]) if rows else None

    def count(self) -> int:
        return self.db.query("SELECT COUNT(*) AS n FROM users")[0]["n"]

    def update_password(self, user_id: str, password_hash: str) -> None:
        self.db.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (password_hash, user_id))

    def create_session(self, token_hash: str, user_id: str, expires_at: datetime) -> None:
        self.db.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, expires_at.isoformat()),
        )

    def session_user(self, token_hash: str, now: datetime) -> User | None:
        rows = self.db.query(
            "SELECT u.* FROM sessions s JOIN users u ON u.user_id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (token_hash, now.isoformat()),
        )
        return _user(rows[0]) if rows else None

    def delete_session(self, token_hash: str) -> None:
        self.db.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

    def delete_sessions(self, user_id: str, *, keep: str | None = None) -> None:
        self.db.execute(
            "DELETE FROM sessions WHERE user_id = ? AND token_hash != ?", (user_id, keep or "")
        )


class InMemoryUserRepository(UserRepository):
    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._sessions: dict[str, tuple[str, datetime]] = {}

    def create(self, user: User) -> None:
        if self.get_by_email(user.email):
            raise ConflictError("an account with this email already exists")
        self._users[user.user_id] = user

    def get(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def get_by_email(self, email: str) -> User | None:
        return next((u for u in self._users.values() if u.email == email), None)

    def count(self) -> int:
        return len(self._users)

    def update_password(self, user_id: str, password_hash: str) -> None:
        self._users[user_id].password_hash = password_hash

    def create_session(self, token_hash: str, user_id: str, expires_at: datetime) -> None:
        self._sessions[token_hash] = (user_id, expires_at)

    def session_user(self, token_hash: str, now: datetime) -> User | None:
        user_id, expires_at = self._sessions.get(token_hash, (None, now))
        return self._users.get(user_id) if user_id and expires_at > now else None

    def delete_session(self, token_hash: str) -> None:
        self._sessions.pop(token_hash, None)

    def delete_sessions(self, user_id: str, *, keep: str | None = None) -> None:
        self._sessions = {
            k: v for k, v in self._sessions.items() if v[0] != user_id or k == keep
        }


class InMemoryDocumentRepository(DocumentRepository):
    """Used by tests and by the CLI when persistence is not wanted."""

    def __init__(self) -> None:
        self._items: dict[str, PaperMetadata] = {}
        self._owners: dict[str, str] = {}

    def save(self, metadata: PaperMetadata, *, owner_id: str | None = None) -> None:
        self._items[metadata.document_id] = metadata
        if owner_id and metadata.document_id not in self._owners:
            self._owners[metadata.document_id] = owner_id

    def get(self, document_id: str) -> PaperMetadata | None:
        return self._items.get(document_id)

    def list(
        self, *, limit: int = 50, offset: int = 0, owner_id: str | None = None
    ) -> list[PaperMetadata]:
        items = [
            m for m in self._items.values()
            if owner_id is None or self._owners.get(m.document_id) == owner_id
        ]
        items.sort(key=lambda m: m.ingested_at, reverse=True)
        return items[offset : offset + limit]

    def owner_of(self, document_id: str) -> str | None:
        return self._owners.get(document_id)

    def claim_unowned(self, owner_id: str) -> int:
        unowned = [d for d in self._items if d not in self._owners]
        for document_id in unowned:
            self._owners[document_id] = owner_id
        return len(unowned)

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

    def list_for_documents(self, document_ids: list[str], *, limit: int = 50) -> list[AnalysisRecord]:
        wanted = set(document_ids)
        items = [r for r in self._items.values() if r.document_id in wanted]
        items.sort(key=lambda r: r.created_at, reverse=True)
        return items[:limit]


def json_dumps(payload: dict) -> str:
    return json.dumps(payload, default=str)
