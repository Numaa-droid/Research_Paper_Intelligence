"""Quote matching: does this quote actually appear in the paper?

An agent claiming ``[p. 4, §4.2, Table 1]`` is making a checkable assertion, and
this module checks it without a model. Matching runs in three tiers:

1. **exact** — the quote appears verbatim in the cited chunk.
2. **normalized** — it appears after whitespace/punctuation normalisation.
3. **approximate** — a sliding-window fuzzy match clears the similarity floor,
   which is what OCR output usually manages.

Anything below tier 3 is ``unverifiable``: the evidence is kept but flagged and
its confidence is cut, so an unsupported citation is visible rather than silent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..domain.enums import CitationQuality

#: Similarity at or above which a fuzzy match counts as approximate.
APPROXIMATE_THRESHOLD = 0.82
#: Confidence multipliers applied per tier.
QUALITY_PENALTY = {
    CitationQuality.EXACT: 1.0,
    CitationQuality.NORMALIZED: 0.95,
    CitationQuality.APPROXIMATE: 0.75,
    CitationQuality.UNVERIFIABLE: 0.25,
}

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s%.\-/+]")


def normalize_for_match(text: str) -> str:
    """Aggressive normalisation used only for comparison, never for storage."""
    if not text:
        return ""
    lowered = text.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    lowered = _PUNCT.sub(" ", lowered)
    return _WS.sub(" ", lowered).strip()


@dataclass(slots=True)
class QuoteMatch:
    """Outcome of checking one quote against one body of text."""

    quality: CitationQuality
    similarity: float
    matched_text: str | None = None
    note: str | None = None

    @property
    def found(self) -> bool:
        return self.quality is not CitationQuality.UNVERIFIABLE

    def adjusted_confidence(self, base: float) -> float:
        return round(min(1.0, max(0.0, base * QUALITY_PENALTY[self.quality])), 4)


def _best_window(needle: str, haystack: str) -> tuple[float, str]:
    """Best fuzzy similarity of ``needle`` against any same-length window."""
    if not needle or not haystack:
        return 0.0, ""
    if len(needle) >= len(haystack):
        return SequenceMatcher(None, needle, haystack).ratio(), haystack

    window = len(needle)
    step = max(1, window // 4)
    best_ratio = 0.0
    best_text = ""
    matcher = SequenceMatcher()
    matcher.set_seq2(needle)
    for start in range(0, len(haystack) - window + 1, step):
        candidate = haystack[start : start + window]
        matcher.set_seq1(candidate)
        # real_quick_ratio/quick_ratio are cheap upper bounds; skip hopeless windows.
        if matcher.real_quick_ratio() < best_ratio or matcher.quick_ratio() < best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_text = candidate
            if best_ratio > 0.99:
                break
    return best_ratio, best_text


def match_quote(
    quote: str, source_text: str, *, threshold: float = APPROXIMATE_THRESHOLD
) -> QuoteMatch:
    """Classify how well ``quote`` is supported by ``source_text``."""
    if not quote or not source_text:
        return QuoteMatch(CitationQuality.UNVERIFIABLE, 0.0, note="empty quote or source")

    if quote in source_text:
        return QuoteMatch(CitationQuality.EXACT, 1.0, matched_text=quote)

    needle = normalize_for_match(quote)
    haystack = normalize_for_match(source_text)
    if not needle:
        return QuoteMatch(CitationQuality.UNVERIFIABLE, 0.0, note="quote normalised to nothing")

    if needle in haystack:
        return QuoteMatch(
            CitationQuality.NORMALIZED,
            1.0,
            matched_text=quote,
            note="matched after whitespace and punctuation normalisation",
        )

    ratio, matched = _best_window(needle, haystack)
    if ratio >= threshold:
        return QuoteMatch(
            CitationQuality.APPROXIMATE,
            round(ratio, 4),
            matched_text=matched,
            note=(
                f"approximate match (similarity {ratio:.2f}); "
                "confidence reduced because the quote is not verbatim"
            ),
        )

    return QuoteMatch(
        CitationQuality.UNVERIFIABLE,
        round(ratio, 4),
        note=f"quote not found in the cited text (best similarity {ratio:.2f})",
    )
