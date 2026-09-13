"""Schema-level validation helpers.

Pydantic already enforces field types and ranges; these helpers cover the checks
that need the *document* as context, plus safe coercion of loosely-typed model
output into the domain models.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError as PydanticValidationError

from ..domain.enums import ElementType, FindingStatus
from ..domain.models import Evidence, Finding
from ..observability import warning

_STATUS_ALIASES = {
    "support": FindingStatus.SUPPORTED,
    "supported": FindingStatus.SUPPORTED,
    "partial": FindingStatus.PARTIALLY_SUPPORTED,
    "partially supported": FindingStatus.PARTIALLY_SUPPORTED,
    "partially_supported": FindingStatus.PARTIALLY_SUPPORTED,
    "unsupported": FindingStatus.UNSUPPORTED,
    "not supported": FindingStatus.UNSUPPORTED,
    "contradicted": FindingStatus.CONTRADICTED,
    "contradiction": FindingStatus.CONTRADICTED,
    "inferred": FindingStatus.INFERRED,
    "inference": FindingStatus.INFERRED,
    "unknown": FindingStatus.UNKNOWN,
}

_ELEMENT_ALIASES = {
    "text": ElementType.PARAGRAPH,
    "paragraph": ElementType.PARAGRAPH,
    "table": ElementType.TABLE,
    "figure": ElementType.FIGURE,
    "image": ElementType.FIGURE,
    "equation": ElementType.EQUATION,
    "formula": ElementType.EQUATION,
    "reference": ElementType.REFERENCE,
    "citation": ElementType.REFERENCE,
}


def coerce_status(value: Any, default: FindingStatus = FindingStatus.UNKNOWN) -> FindingStatus:
    if isinstance(value, FindingStatus):
        return value
    if isinstance(value, str):
        return _STATUS_ALIASES.get(value.strip().lower(), default)
    return default


def coerce_element_type(value: Any) -> ElementType:
    if isinstance(value, ElementType):
        return value
    if isinstance(value, str):
        return _ELEMENT_ALIASES.get(value.strip().lower(), ElementType.UNKNOWN)
    return ElementType.UNKNOWN


def clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return round(min(1.0, max(0.0, number)), 4)


def parse_evidence(payload: Any, *, agent: str = "unknown") -> Evidence | None:
    """Build an :class:`Evidence` from loose model output, or ``None``.

    Returning ``None`` for malformed evidence is deliberate: a citation that
    cannot even be parsed must not become a citation with invented defaults.
    """
    if not isinstance(payload, dict):
        return None
    quote = str(payload.get("quote") or "").strip()
    page_raw = payload.get("page")
    if not quote:
        return None
    try:
        page = int(page_raw)
    except (TypeError, ValueError):
        warning("validation.evidence_dropped", agent=agent, reason="page is not an integer")
        return None
    if page < 1:
        warning("validation.evidence_dropped", agent=agent, reason="page below 1")
        return None

    try:
        return Evidence(
            page=page,
            section=(str(payload["section"]).strip() or None) if payload.get("section") else None,
            source_type=coerce_element_type(payload.get("source_type")),
            element_id=(
                str(payload["element_id"]).strip() or None if payload.get("element_id") else None
            ),
            quote=quote,
            chunk_id=(str(payload["chunk_id"]).strip() or None) if payload.get("chunk_id") else None,
            confidence=clamp_confidence(payload.get("confidence"), 0.5),
        )
    except PydanticValidationError as exc:
        warning("validation.evidence_dropped", agent=agent, reason=str(exc)[:200])
        return None


def parse_finding(payload: Any, *, agent: str) -> Finding | None:
    """Build a :class:`Finding` from loose model output, or ``None``."""
    if not isinstance(payload, dict):
        return None
    claim = str(payload.get("claim") or "").strip()
    if not claim:
        return None

    evidence = [
        parsed
        for item in (payload.get("evidence") or [])
        if (parsed := parse_evidence(item, agent=agent)) is not None
    ]
    try:
        return Finding(
            claim=claim,
            explanation=str(payload.get("explanation") or "").strip(),
            evidence=evidence,
            confidence=clamp_confidence(payload.get("confidence"), 0.5),
            status=coerce_status(payload.get("status")),
            category=(str(payload["category"]).strip() or None) if payload.get("category") else None,
            agent=agent,
            impact=clamp_confidence(payload.get("impact"), 0.5),
        )
    except PydanticValidationError as exc:
        warning("validation.finding_dropped", agent=agent, reason=str(exc)[:200])
        return None


def parse_findings(payload: Any, *, agent: str) -> list[Finding]:
    if not isinstance(payload, list):
        return []
    return [f for item in payload if (f := parse_finding(item, agent=agent)) is not None]


def as_str_list(value: Any, *, limit: int = 40) -> list[str]:
    """Coerce model output into a clean list of non-empty strings."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:limit]:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def as_str_dict(value: Any, *, limit: int = 60) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in list(value.items())[:limit] if str(k).strip()}
