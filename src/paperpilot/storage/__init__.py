"""Storage adapters. Everything above this layer depends only on the interfaces."""

from .files import LocalFileStorage, safe_filename
from .repository import AnalysisRepository, DocumentRepository, FileStorage
from .sqlite_repo import (
    InMemoryAnalysisRepository,
    InMemoryDocumentRepository,
    SQLiteAnalysisRepository,
    SQLiteConnection,
    SQLiteDocumentRepository,
)

__all__ = [
    "AnalysisRepository",
    "DocumentRepository",
    "FileStorage",
    "InMemoryAnalysisRepository",
    "InMemoryDocumentRepository",
    "LocalFileStorage",
    "SQLiteAnalysisRepository",
    "SQLiteConnection",
    "SQLiteDocumentRepository",
    "safe_filename",
]
