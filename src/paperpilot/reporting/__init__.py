"""Reporting: deterministic Markdown synthesis from validated structures."""

from .citations import format_citation, format_citations, format_quote
from .synthesizer import ReportSynthesizer

__all__ = ["ReportSynthesizer", "format_citation", "format_citations", "format_quote"]
