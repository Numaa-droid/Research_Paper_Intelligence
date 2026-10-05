"""PostgreSQL-backed repositories (Supabase, or any Postgres 13+).

Same contract and record format as the SQLite backend: JSON payloads beside a
few indexed columns. Selected when ``PAPERPILOT_DATABASE_URL`` is a
``postgresql://`` URL.

Tables live in their own schema (``paperpilot`` by default) with row-level
security enabled and no policies. On Supabase that keeps them out of the
auto-generated Data API, which exposes ``public`` to anyone holding the
project's anon key; this server connects as the table owner, which RLS does
not restrict. Every statement names the schema explicitly instead of relying
on ``search_path``, which Supabase's connection pooler does not pass through.
"""

from __future__ import annotations

import atexit
import re
from datetime import datetime
from urllib.parse import urlsplit

from ..domain.models import AnalysisRecord, PaperMetadata
from ..errors import ConfigurationError, ConflictError
from ..observability import info
from .repository import AnalysisRepository, DocumentRepository, User, UserRepository

_SCHEMA_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

_DDL = """
CREATE SCHEMA IF NOT EXISTS {s};

CREATE TABLE IF NOT EXISTS {s}.documents (
    document_id TEXT PRIMARY KEY,
    title       TEXT,
    filename    TEXT,
    sha256      TEXT,
    page_count  INTEGER NOT NULL DEFAULT 0,
    ingested_at TIMESTAMPTZ NOT NULL,
    payload     TEXT NOT NULL,
    owner_id    TEXT
);
CREATE INDEX IF NOT EXISTS documents_sha256_idx ON {s}.documents (sha256);
CREATE INDEX IF NOT EXISTS documents_owner_idx ON {s}.documents (owner_id, ingested_at DESC);

CREATE TABLE IF NOT EXISTS {s}.analyses (
    analysis_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES {s}.documents (document_id) ON DELETE CASCADE,
    status      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL,
    report_path TEXT,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS analyses_document_idx ON {s}.analyses (document_id, created_at DESC);

CREATE TABLE IF NOT EXISTS {s}.users (
    user_id       TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS {s}.sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES {s}.users (user_id) ON DELETE CASCADE,
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user_idx ON {s}.sessions (user_id);

ALTER TABLE {s}.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE {s}.analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE {s}.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE {s}.sessions ENABLE ROW LEVEL SECURITY;
"""


def is_postgres_url(url: str) -> bool:
    return url.startswith(("postgresql://", "postgres://", "postgresql+psycopg://"))


def _describe(url: str) -> str:
    """Host and database of ``url``, without the credentials, for messages."""
    parts = urlsplit(url)
    return f"{parts.hostname or 'local socket'}:{parts.port or 5432}{parts.path or ''}"


class PostgresConnection:
    """A small connection pool, opened once and shared by the repositories.

    Prepared statements are off so the pool also works through Supabase's
    transaction pooler (port 6543), which cannot keep them between calls.
    """

    def __init__(self, url: str, *, schema: str = "paperpilot", max_size: int = 8) -> None:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise ConfigurationError(
                "PAPERPILOT_DATABASE_URL points at PostgreSQL but the driver is not installed: "
                'pip install -e ".[postgres]"'
            ) from exc
        if not _SCHEMA_NAME.match(schema):
            raise ConfigurationError(f"invalid database schema name: {schema!r}")

        url = url.replace("postgresql+psycopg://", "postgresql://", 1)
        if "supabase." in url and "sslmode=" not in url:
            url += ("&" if "?" in url else "?") + "sslmode=require"
        self.schema = schema
        self.target = _describe(url)
        self.pool = ConnectionPool(
            url,
            min_size=1,
            max_size=max_size,
            kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None},
            open=False,
        )
        try:
            self.pool.open(wait=True, timeout=20)
        except Exception as exc:
            self.pool.close()
            raise ConfigurationError(
                f"could not connect to the database at {self.target}: {exc}. Check "
                "PAPERPILOT_DATABASE_URL (on Supabase, use the session pooler string "
                "if your network has no IPv6)."
            ) from exc
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(_DDL.format(s=schema))
        # Close the pool's worker threads at exit instead of leaving them to time out.
        atexit.register(self.close)
        info("storage.postgres_ready", target=self.target, schema=schema)

    def table(self, name: str) -> str:
        return f"{self.schema}.{name}"

    def execute(self, sql: str, params: tuple | dict = ()) -> int:
        with self.pool.connection() as conn:
            return conn.execute(sql, params).rowcount

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    def close(self) -> None:
        self.pool.close()


class PostgresDocumentRepository(DocumentRepository):
    def __init__(self, connection: PostgresConnection) -> None:
        self.db = connection
        self.t = connection.table("documents")

    def save(self, metadata: PaperMetadata, *, owner_id: str | None = None) -> None:
        self.db.execute(
            f"""
            INSERT INTO {self.t} AS d
                (document_id, title, filename, sha256, page_count, ingested_at, payload, owner_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE SET
                title = EXCLUDED.title, filename = EXCLUDED.filename, sha256 = EXCLUDED.sha256,
                page_count = EXCLUDED.page_count, payload = EXCLUDED.payload,
                owner_id = COALESCE(d.owner_id, EXCLUDED.owner_id)
            """,
            (
                metadata.document_id,
                metadata.title,
                metadata.filename,
                metadata.sha256,
                metadata.page_count,
                metadata.ingested_at,
                metadata.model_dump_json(),
                owner_id,
            ),
        )

    def get(self, document_id: str) -> PaperMetadata | None:
        rows = self.db.query(f"SELECT payload FROM {self.t} WHERE document_id = %s", (document_id,))
        return PaperMetadata.model_validate_json(rows[0]["payload"]) if rows else None

    def list(
        self, *, limit: int = 50, offset: int = 0, owner_id: str | None = None
    ) -> list[PaperMetadata]:
        if owner_id is None:
            rows = self.db.query(
                f"SELECT payload FROM {self.t} ORDER BY ingested_at DESC LIMIT %s OFFSET %s",
                (limit, offset),
            )
        else:
            rows = self.db.query(
                f"SELECT payload FROM {self.t} WHERE owner_id = %s "
                "ORDER BY ingested_at DESC LIMIT %s OFFSET %s",
                (owner_id, limit, offset),
            )
        return [PaperMetadata.model_validate_json(r["payload"]) for r in rows]

    def find_by_hash(self, sha256: str) -> PaperMetadata | None:
        rows = self.db.query(
            f"SELECT payload FROM {self.t} WHERE sha256 = %s ORDER BY ingested_at DESC LIMIT 1",
            (sha256,),
        )
        return PaperMetadata.model_validate_json(rows[0]["payload"]) if rows else None

    def owner_of(self, document_id: str) -> str | None:
        rows = self.db.query(f"SELECT owner_id FROM {self.t} WHERE document_id = %s", (document_id,))
        return rows[0]["owner_id"] if rows else None

    def claim_unowned(self, owner_id: str) -> int:
        return self.db.execute(f"UPDATE {self.t} SET owner_id = %s WHERE owner_id IS NULL", (owner_id,))


class PostgresAnalysisRepository(AnalysisRepository):
    def __init__(self, connection: PostgresConnection) -> None:
        self.db = connection
        self.t = connection.table("analyses")

    def save(self, record: AnalysisRecord) -> None:
        self.db.execute(
            f"""
            INSERT INTO {self.t}
                (analysis_id, document_id, status, created_at, updated_at, report_path, payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (analysis_id) DO UPDATE SET
                status = EXCLUDED.status, updated_at = EXCLUDED.updated_at,
                report_path = EXCLUDED.report_path, payload = EXCLUDED.payload
            """,
            (
                record.analysis_id,
                record.document_id,
                record.status.value,
                record.created_at,
                record.updated_at,
                record.report_path,
                record.model_dump_json(),
            ),
        )

    def get(self, analysis_id: str) -> AnalysisRecord | None:
        rows = self.db.query(f"SELECT payload FROM {self.t} WHERE analysis_id = %s", (analysis_id,))
        return AnalysisRecord.model_validate_json(rows[0]["payload"]) if rows else None

    def list_for_document(self, document_id: str, *, limit: int = 20) -> list[AnalysisRecord]:
        rows = self.db.query(
            f"SELECT payload FROM {self.t} WHERE document_id = %s ORDER BY created_at DESC LIMIT %s",
            (document_id, limit),
        )
        return [AnalysisRecord.model_validate_json(r["payload"]) for r in rows]

    def list_for_documents(self, document_ids: list[str], *, limit: int = 50) -> list[AnalysisRecord]:
        if not document_ids:
            return []
        rows = self.db.query(
            f"SELECT payload FROM {self.t} WHERE document_id = ANY(%s) ORDER BY created_at DESC LIMIT %s",
            (list(document_ids), limit),
        )
        return [AnalysisRecord.model_validate_json(r["payload"]) for r in rows]


def _user(row: dict) -> User:
    return User(
        user_id=row["user_id"],
        email=row["email"],
        name=row["name"],
        password_hash=row["password_hash"],
        created_at=row["created_at"],
    )


class PostgresUserRepository(UserRepository):
    def __init__(self, connection: PostgresConnection) -> None:
        self.db = connection
        self.users = connection.table("users")
        self.sessions = connection.table("sessions")

    def create(self, user: User) -> None:
        from psycopg.errors import UniqueViolation

        try:
            self.db.execute(
                f"INSERT INTO {self.users} (user_id, email, name, password_hash, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (user.user_id, user.email, user.name, user.password_hash, user.created_at),
            )
        except UniqueViolation as exc:
            raise ConflictError("an account with this email already exists") from exc

    def get(self, user_id: str) -> User | None:
        rows = self.db.query(f"SELECT * FROM {self.users} WHERE user_id = %s", (user_id,))
        return _user(rows[0]) if rows else None

    def get_by_email(self, email: str) -> User | None:
        rows = self.db.query(f"SELECT * FROM {self.users} WHERE email = %s", (email,))
        return _user(rows[0]) if rows else None

    def count(self) -> int:
        return self.db.query(f"SELECT COUNT(*) AS n FROM {self.users}")[0]["n"]

    def update_password(self, user_id: str, password_hash: str) -> None:
        self.db.execute(
            f"UPDATE {self.users} SET password_hash = %s WHERE user_id = %s", (password_hash, user_id)
        )

    def create_session(self, token_hash: str, user_id: str, expires_at: datetime) -> None:
        # Sweep expired sessions as new ones are made, so the table stays small.
        self.db.execute(f"DELETE FROM {self.sessions} WHERE expires_at < now()")
        self.db.execute(
            f"INSERT INTO {self.sessions} (token_hash, user_id, expires_at) VALUES (%s, %s, %s)",
            (token_hash, user_id, expires_at),
        )

    def session_user(self, token_hash: str, now: datetime) -> User | None:
        rows = self.db.query(
            f"SELECT u.* FROM {self.sessions} s JOIN {self.users} u ON u.user_id = s.user_id "
            "WHERE s.token_hash = %s AND s.expires_at > %s",
            (token_hash, now),
        )
        return _user(rows[0]) if rows else None

    def delete_session(self, token_hash: str) -> None:
        self.db.execute(f"DELETE FROM {self.sessions} WHERE token_hash = %s", (token_hash,))

    def delete_sessions(self, user_id: str, *, keep: str | None = None) -> None:
        self.db.execute(
            f"DELETE FROM {self.sessions} WHERE user_id = %s AND token_hash <> %s",
            (user_id, keep or ""),
        )
