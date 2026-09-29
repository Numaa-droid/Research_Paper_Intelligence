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
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…|\[\s*(?:\.\.\.|…)\s*\])\s*")
#: A part of an elided quote shorter than this proves nothing on its own.
_MIN_ELIDED_PART = 12
_PUNCT = re.compile(r"[^\w\s%.\-/+]")
#: A full stop that is not a decimal point: sentence punctuation, which a writer
#: may close an elided or shortened quote with where the paper has a comma.
_SENTENCE_STOP = re.compile(r"(?<!\d)\.|\.(?!\d)")
#: LaTeX a model writes for math the PDF shows as Unicode.
_LATEX_SYMBOLS = {
    r"\sqrt": "√", r"\cdot": "·", r"\times": "×", r"\in": "∈", r"\infty": "∞",
    r"\sum": "∑", r"\prod": "∏", r"\leq": "≤", r"\geq": "≥", r"\le": "≤", r"\ge": "≥",
    r"\neq": "≠", r"\approx": "≈", r"\epsilon": "ϵ", r"\varepsilon": "ϵ",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\lambda": "λ",
    r"\mu": "μ", r"\sigma": "σ", r"\theta": "θ", r"\pi": "π", r"\phi": "φ",
    r"\omega": "ω", r"\infin": "∞",
}
_LATEX_COMMAND = re.compile(r"\\([A-Za-z]+)")
#: Markup that carries layout, not content: sub/superscript markers and groups.
_LAYOUT_MARKS = re.compile(r"[_^{}\\$]")


def _latex_to_text(text: str) -> str:
    def symbol(match: re.Match) -> str:
        return _LATEX_SYMBOLS.get(match.group(0), match.group(1))

    return _LATEX_COMMAND.sub(symbol, text)


def normalize_for_match(text: str) -> str:
    """Aggressive normalisation used only for comparison, never for storage.

    Case, punctuation (decimal points excepted), dashes, LaTeX commands and
    sub/superscript markup are all levelled, so ``d_{model}``, ``d_model`` and
    the PDF's ``dmodel`` compare equal, as do ``transformation.`` and
    ``transformation,``.
    """
    if not text:
        return ""
    lowered = _latex_to_text(text).lower()
    for src, dst in (("’", "'"), ("–", "-"), ("—", "-"), ("−", "-"), ("ε", "ϵ")):
        lowered = lowered.replace(src, dst)
    lowered = _LAYOUT_MARKS.sub("", lowered)
    lowered = _PUNCT.sub(" ", lowered)
    lowered = _SENTENCE_STOP.sub(" ", lowered)
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


def _match_elided(quote: str, haystack: str) -> QuoteMatch | None:
    """A quote shortened with "..." matches when every part appears, in order.

    Writers elide the middle of a long sentence; that is still a faithful quote
    as long as nothing between the parts was changed. Parts too short to be
    meaningful are ignored, and at least one meaningful part is required.

    Parts found out of document order still match, but only as APPROXIMATE. A
    quote that joins two real sentences in the reverse order is every word the
    paper wrote, so discarding it outright pushes the finding to "unsupported"
    and reports the paper as not saying something it does say; but the ellipsis
    does claim the parts run in that order, and they do not. Grounding it at a
    lower quality keeps the evidence and prices in the discrepancy, and the note
    records the reordering for the evidence appendix.
    """
    if not _ELLIPSIS.search(quote):
        return None
    parts = [normalize_for_match(p) for p in _ELLIPSIS.split(quote)]
    parts = [p for p in parts if len(p) >= _MIN_ELIDED_PART]
    if not parts:
        return None

    position = 0
    ordered = True
    for part in parts:
        found = haystack.find(part, position)
        if found < 0:
            # Either absent, or present earlier in the page than the part
            # quoted before it.
            if haystack.find(part) < 0:
                return None
            ordered = False
            continue
        position = found + len(part)

    return QuoteMatch(
        CitationQuality.NORMALIZED if ordered else CitationQuality.APPROXIMATE,
        1.0,
        matched_text=quote,
        note=(
            f"elided quote: all {len(parts)} part(s) found in order"
            if ordered
            else f"elided quote: all {len(parts)} part(s) found, but not in the order quoted; "
            "graded approximate"
        ),
    )


def match_quote(
    quote: str,
    source_text: str,
    *,
    threshold: float = APPROXIMATE_THRESHOLD,
    fuzzy: bool = True,
) -> QuoteMatch:
    """Classify how well ``quote`` is supported by ``source_text``.

    ``fuzzy=False`` stops after the exact, normalised and elided tiers: cheap
    enough to run against every chunk of a paper.
    """
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

    elided = _match_elided(quote, haystack)
    if elided is not None:
        return elided

    if not fuzzy:
        return QuoteMatch(
            CitationQuality.UNVERIFIABLE, 0.0, note="quote not found verbatim in the text"
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
