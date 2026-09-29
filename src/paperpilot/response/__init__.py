"""Model answers as Markdown, parsed into a deterministic JSON document."""

from .parser import extract_citations, normalize_key, parse_markdown, strip_citations
from .schema import SCHEMA_VERSION, ResponseDocument, response_document_json_schema

__all__ = [
    "SCHEMA_VERSION",
    "ResponseDocument",
    "extract_citations",
    "normalize_key",
    "parse_markdown",
    "response_document_json_schema",
    "strip_citations",
]
