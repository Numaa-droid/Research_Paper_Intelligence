"""Deterministic (non-LLM) validation.

Page ranges, quote presence, element references, numeric consistency and schema
shape all have exact answers given the index, so none of them are delegated to a
model.
"""

from .evidence import (
    EvidenceValidation,
    FindingValidation,
    drop_unverifiable,
    resolve_evidence_chunk,
    validate_evidence,
    validate_finding,
    validate_findings,
)
from .numeric import (
    check_claim_numbers,
    check_numbers_against_text,
    extract_numbers,
    numeric_accuracy,
    parse_number,
)
from .quotes import QuoteMatch, match_quote, normalize_for_match
from .schema import (
    as_str_dict,
    as_str_list,
    clamp_confidence,
    coerce_element_type,
    coerce_status,
    parse_evidence,
    parse_finding,
    parse_findings,
)

__all__ = [
    "EvidenceValidation",
    "FindingValidation",
    "QuoteMatch",
    "as_str_dict",
    "as_str_list",
    "check_claim_numbers",
    "check_numbers_against_text",
    "clamp_confidence",
    "coerce_element_type",
    "coerce_status",
    "drop_unverifiable",
    "extract_numbers",
    "match_quote",
    "normalize_for_match",
    "numeric_accuracy",
    "parse_evidence",
    "parse_finding",
    "parse_findings",
    "parse_number",
    "resolve_evidence_chunk",
    "validate_evidence",
    "validate_finding",
    "validate_findings",
]
