"""Citation rendering.

Citations are generated **only** from validated :class:`Evidence` objects. There
is deliberately no code path that turns a page number the model typed into a
citation: if the evidence did not survive validation, it is rendered as an
explicit unverifiable marker instead of a citation.

Format: ``[p. 7, §4.2 Main Results, Table 3]``
"""

from __future__ import annotations

import re

from ..domain.enums import CitationQuality, ElementType
from ..domain.models import Evidence

#: Rendered in place of a citation when evidence failed validation.
UNVERIFIED_MARKER = "[unverified]"

_ELEMENT_LABELS = {
    ElementType.TABLE: "Table",
    ElementType.FIGURE: "Figure",
    ElementType.EQUATION: "Equation",
}

_ELEMENT_ID_RE = re.compile(r"^(table|figure|equation|algorithm)[-\s]?(.+)$", re.IGNORECASE)


def element_label(evidence: Evidence) -> str | None:
    """Human-readable element label, e.g. ``Table 3``."""
    if evidence.element_id:
        match = _ELEMENT_ID_RE.match(evidence.element_id)
        if match:
            return f"{match.group(1).title()} {match.group(2)}"
        return evidence.element_id
    return _ELEMENT_LABELS.get(evidence.source_type)


def format_citation(evidence: Evidence, *, include_quality: bool = False) -> str:
    """Render one evidence object as a citation.

    Unverifiable evidence never produces a citation: doing so would launder a
    failed check into something that looks checked.
    """
    if not evidence.is_verified:
        return UNVERIFIED_MARKER

    parts = [f"p. {evidence.page}"]
    if evidence.section:
        parts.append(f"§{evidence.section}")
    label = element_label(evidence)
    if label:
        parts.append(label)

    citation = f"[{', '.join(parts)}]"
    if include_quality and evidence.citation_quality is not CitationQuality.EXACT:
        citation += f" _({evidence.citation_quality.value} match)_"
    return citation


def format_citations(evidence: list[Evidence], *, limit: int = 4) -> str:
    """Render a de-duplicated citation group for one statement."""
    seen: set[str] = set()
    citations: list[str] = []
    unverified = 0

    for item in evidence:
        rendered = format_citation(item)
        if rendered == UNVERIFIED_MARKER:
            unverified += 1
            continue
        if rendered in seen:
            continue
        seen.add(rendered)
        citations.append(rendered)
        if len(citations) >= limit:
            break

    if not citations:
        return UNVERIFIED_MARKER if unverified else "[no evidence]"
    out = " ".join(citations)
    if unverified:
        out += f" _({unverified} further citation(s) could not be verified)_"
    return out


def format_quote(evidence: Evidence, *, max_chars: int = 280) -> str:
    """Render an evidence quote for the appendix, flagging inexact matches."""
    quote = evidence.quote.replace("\n", " ").strip()
    if len(quote) > max_chars:
        quote = quote[: max_chars - 1].rstrip() + "…"
    suffix = ""
    if evidence.citation_quality is CitationQuality.APPROXIMATE:
        suffix = " _(approximate match to the source text)_"
    elif evidence.citation_quality is CitationQuality.NORMALIZED:
        suffix = " _(whitespace-normalised)_"
    elif evidence.citation_quality is CitationQuality.UNVERIFIABLE:
        suffix = " _(**not found in the paper**)_"
    return f"> {quote}{suffix}"


def citation_key(evidence: Evidence) -> tuple[int, str, str]:
    """Sort key that orders evidence by page, then section, then element."""
    return (evidence.page, evidence.section or "", evidence.element_id or "")
