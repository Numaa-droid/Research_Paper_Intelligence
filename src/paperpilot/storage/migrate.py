"""Copy a SQLite database into the configured backend (e.g. Supabase Postgres).

Accounts, papers (with their owners) and analyses are copied; login sessions
are not, so everyone signs in once on the new database. Safe to re-run: papers
and analyses are upserted, and an account whose email already exists is left
as it is.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from ..domain.models import AnalysisRecord, PaperMetadata
from ..errors import ValidationError
from .repository import AnalysisRepository, DocumentRepository, User, UserRepository


def copy_from_sqlite(
    source: Path,
    *,
    documents: DocumentRepository,
    analyses: AnalysisRepository,
    users: UserRepository,
) -> dict[str, int]:
    if not source.is_file():
        raise ValidationError(f"no SQLite database at {source}")
    db = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {r["name"] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    counts = {"users": 0, "users_skipped": 0, "documents": 0, "analyses": 0}

    if "users" in tables:
        for row in db.execute("SELECT * FROM users ORDER BY created_at"):
            if users.get_by_email(row["email"]) or users.get(row["user_id"]):
                counts["users_skipped"] += 1
                continue
            users.create(
                User(
                    user_id=row["user_id"],
                    email=row["email"],
                    name=row["name"],
                    password_hash=row["password_hash"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
            counts["users"] += 1

    owned = "owner_id" in {r["name"] for r in db.execute("PRAGMA table_info(documents)")}
    for row in db.execute(f"SELECT payload{', owner_id' if owned else ''} FROM documents"):
        documents.save(
            PaperMetadata.model_validate_json(row["payload"]),
            owner_id=row["owner_id"] if owned else None,
        )
        counts["documents"] += 1

    for row in db.execute("SELECT payload FROM analyses ORDER BY created_at"):
        analyses.save(AnalysisRecord.model_validate_json(row["payload"]))
        counts["analyses"] += 1

    db.close()
    return counts
