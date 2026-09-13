"""Enumerations shared across the whole pipeline."""

from __future__ import annotations

from enum import StrEnum


class ElementType(StrEnum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    REFERENCE = "reference"
    UNKNOWN = "unknown"


class ExtractionMethod(StrEnum):
    NATIVE_TEXT = "native_text"
    OCR = "ocr"


class FindingStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class CitationQuality(StrEnum):
    """How faithfully an evidence quote maps back onto the indexed paper text."""

    EXACT = "exact"
    APPROXIMATE = "approximate"
    NORMALIZED = "normalized"
    UNVERIFIABLE = "unverifiable"


class AgentName(StrEnum):
    PLANNER = "planner"
    SUMMARY = "summary"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    MATH = "math"
    VERIFICATION = "verification"
    SYNTHESIZER = "synthesizer"


class AnalysisStatus(StrEnum):
    PENDING = "pending"
    INGESTING = "ingesting"
    INDEXING = "indexing"
    PLANNING = "planning"
    ANALYZING = "analyzing"
    VERIFYING = "verifying"
    RETRYING = "retrying"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATUSES = frozenset({AnalysisStatus.COMPLETED, AnalysisStatus.FAILED})

#: Statuses that mean a finding is *not* adequately grounded in the paper.
WEAK_STATUSES = frozenset(
    {
        FindingStatus.UNSUPPORTED,
        FindingStatus.CONTRADICTED,
        FindingStatus.UNKNOWN,
    }
)
