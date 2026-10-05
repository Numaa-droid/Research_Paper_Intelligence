"""Choose the storage backend from ``PAPERPILOT_DATABASE_URL``."""

from __future__ import annotations

from ..config import Settings
from .repository import AnalysisRepository, DocumentRepository, UserRepository


def build_repositories(
    settings: Settings,
) -> tuple[DocumentRepository, AnalysisRepository, UserRepository]:
    """SQLite for a ``sqlite:///`` URL, PostgreSQL (e.g. Supabase) for ``postgresql://``."""
    from .postgres_repo import is_postgres_url

    if is_postgres_url(settings.database_url):
        from .postgres_repo import (
            PostgresAnalysisRepository,
            PostgresConnection,
            PostgresDocumentRepository,
            PostgresUserRepository,
        )

        pg = PostgresConnection(settings.database_url, schema=settings.database_schema)
        return PostgresDocumentRepository(pg), PostgresAnalysisRepository(pg), PostgresUserRepository(pg)

    from .sqlite_repo import (
        SQLiteAnalysisRepository,
        SQLiteConnection,
        SQLiteDocumentRepository,
        SQLiteUserRepository,
    )

    lite = SQLiteConnection(settings.sqlite_path)
    return SQLiteDocumentRepository(lite), SQLiteAnalysisRepository(lite), SQLiteUserRepository(lite)
