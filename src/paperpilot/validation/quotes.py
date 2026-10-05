"""Quote matching: does this quote actually appear in the paper?

An agent claiming ``[p. 4, §4.2, Table 1]`` is making a checkable assertion, and
this module checks it without a model. Matching runs in three tiers:

1. **exact** — the quote appears verbatim in the cited chunk.
2. **normalized** — it appears after whitespace/punctuation normalisation.
3. **approximate** — a sliding-window fuzzy match clears the similarity floor,
   which is what OCR output usually manages, *and* the quote aligns with the
   paper word for word: every number and negation identical, every other word
   at most one character from the paper's, and no whole word added or dropped.

Character similarity alone cannot tell OCR noise from a changed claim:
"reaches 91.4 F1" is 97% similar to "reaches 81.4 F1", and "increasing" swapped
for "reducing" still clears the floor. The word alignment is what separates
"NarratlveQA" (a misread glyph) from "91.4" (a different result).

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
#: "doesn't" -> "does not", so a dropped or added negation is a whole word.
_CONTRACTED_NOT = re.compile(r"n['’]t\b")
#: Words that invert a statement. They must match exactly: one is never an OCR
#: misreading of another word, and dropping one flips the claim.
_NEGATIONS = frozenset(
    {"not", "no", "never", "neither", "nor", "without", "cannot", "none", "nothing"}
)
#: Paper text kept either side of the best fuzzy window when aligning words.
#: The window may sit well off the quote, by its step and by any words the
#: quote adds, so the margin grows with the quote.
_ALIGN_PAD = 24
_UNALIGNED = float("inf")
_DIGIT = re.compile(r"\d")
_NOT_DIGITS = re.compile(r"[^\d.]")
_NOT_LETTERS = re.compile(r"[^a-z]")


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
    lowered = _CONTRACTED_NOT.sub(" not", _latex_to_text(text).lower())
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


def _best_window(needle: str, haystack: str) -> tuple[float, int, str]:
    """Best fuzzy similarity of ``needle`` against any same-length window.

    Returns the similarity, the window's start offset and the window itself.
    """
    if not needle or not haystack:
        return 0.0, 0, ""
    if len(needle) >= len(haystack):
        return SequenceMatcher(None, needle, haystack).ratio(), 0, haystack

    window = len(needle)
    step = max(1, window // 4)
    best_ratio = 0.0
    best_start = 0
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
            best_start = start
            best_text = candidate
            if best_ratio > 0.99:
                break
    return best_ratio, best_start, best_text


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _word_cost(quoted: str, paper: str) -> float:
    """Cost of reading ``paper`` as ``quoted``: an OCR slip, or a different word.

    A word with digits must have the same digits, sign and letters as the
    paper's; only the punctuation between them may differ, because PDFs drop
    range dashes and slashes ("13.5–17.5" extracts as "13.517.5"). Negations
    must be identical. Any other word may differ from the paper's by one
    character, the size of a misread glyph; when that character is added or
    dropped rather than swapped, the words must start alike, so an affix that
    changes the meaning ("typical" / "atypical") is not a slip.
    """
    if quoted == paper:
        return 0.0
    if _DIGIT.search(quoted) or _DIGIT.search(paper):
        same = (
            _NOT_DIGITS.sub("", quoted) == _NOT_DIGITS.sub("", paper)
            and _NOT_LETTERS.sub("", quoted) == _NOT_LETTERS.sub("", paper)
            and quoted.startswith("-") == paper.startswith("-")
        )
        return 0.0 if same else _UNALIGNED
    if quoted in _NEGATIONS or paper in _NEGATIONS:
        return _UNALIGNED
    if _edit_distance(quoted, paper) > 1:
        return _UNALIGNED
    if len(quoted) != len(paper) and quoted[0] != paper[0]:
        return _UNALIGNED
    return 1.0


def _joined_cost(quoted: list[str], paper: list[str]) -> float:
    """Cost of a word split or run together by the PDF ("atten- tion", "1,024")."""
    a, b = "".join(quoted), "".join(paper)
    if any(w in _NEGATIONS for w in quoted + paper) and not (a in _NEGATIONS and a == b):
        return _UNALIGNED
    return _word_cost(a, b)


def _skip_quoted(word: str) -> float:
    """Cost of a quoted word the paper lacks: only a stray glyph, never a number."""
    if len(word) <= 2 and not _DIGIT.search(word) and word not in _NEGATIONS:
        return 1.0
    return _UNALIGNED


def _skip_paper(word: str) -> float:
    """Cost of a paper word the quote leaves out: a stray glyph or footnote mark.

    Footnote markers are digits ("gradients 4 to"), and leaving one out of a
    quote asserts nothing, so short numbers may be skipped on this side only.
    """
    if len(word) <= 2 and word not in _NEGATIONS:
        return 1.0
    return _UNALIGNED


def _aligns_word_for_word(needle: str, region: str) -> bool:
    """Whether the quote's words map onto a run of the paper's words.

    A semi-global alignment: the quote must be consumed whole, the paper text
    either side of it is free. Each step pairs two words, joins two words into
    one on either side, or skips a word on one side; the costs above make any
    changed number, negation or real word unalignable.
    """
    quoted, paper = needle.split(), region.split()
    n, m = len(quoted), len(paper)
    inf = _UNALIGNED
    rows = [[0.0] * (m + 1)]
    for i in range(1, n + 1):
        prev = rows[i - 1]
        row = [prev[0] + _skip_quoted(quoted[i - 1])] + [inf] * m
        for j in range(1, m + 1):
            best = min(
                prev[j - 1] + _word_cost(quoted[i - 1], paper[j - 1]),
                prev[j] + _skip_quoted(quoted[i - 1]),
                row[j - 1] + _skip_paper(paper[j - 1]),
            )
            if i >= 2:
                best = min(best, rows[i - 2][j - 1] + _joined_cost(quoted[i - 2 : i], [paper[j - 1]]))
            if j >= 2:
                best = min(best, prev[j - 2] + _joined_cost([quoted[i - 1]], paper[j - 2 : j]))
            row[j] = best
        rows.append(row)
    return min(rows[n]) < inf


def _numbers_missing(needle: str, region: str) -> list[str]:
    """Quoted words with digits that no word of the paper text matches."""
    available = {_NOT_DIGITS.sub("", w) for w in region.split() if _DIGIT.search(w)}
    return [w for w in needle.split() if _DIGIT.search(w) and _NOT_DIGITS.sub("", w) not in available]


def _align_region(haystack: str, start: int, length: int, needle: str) -> str:
    """The paper text around a fuzzy window, widened to whole words."""
    pad = len(needle) // 2 + _ALIGN_PAD
    lo, hi = max(0, start - pad), min(len(haystack), start + length + pad)
    while lo > 0 and haystack[lo - 1] != " ":
        lo -= 1
    while hi < len(haystack) and haystack[hi] != " ":
        hi += 1
    return haystack[lo:hi]


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
    all_parts = [p for p in (normalize_for_match(p) for p in _ELLIPSIS.split(quote)) if p]
    parts = [p for p in all_parts if len(p) >= _MIN_ELIDED_PART]
    if not parts:
        return None
    # A short part proves nothing on its own and may be a label the writer added
    # ("(Abstract) ..."), but one that carries a number or a negation is a claim:
    # "reaches ... 91.4 F1 ... on NarrativeQA" must not slip a number the paper
    # does not have between two parts that it does.
    padded = f" {haystack} "
    if any(
        f" {p} " not in padded
        for p in all_parts
        if len(p) < _MIN_ELIDED_PART and (_DIGIT.search(p) or _NEGATIONS & set(p.split()))
    ):
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

    ratio, start, matched = _best_window(needle, haystack)
    if ratio >= threshold:
        region = _align_region(haystack, start, len(matched), needle)
        if not _aligns_word_for_word(needle, region):
            missing = _numbers_missing(needle, region)
            reason = (
                f"the paper text it matches does not contain {', '.join(missing)}"
                if missing
                else "the quote changes, adds or drops a word of the paper text, beyond OCR noise"
            )
            return QuoteMatch(
                CitationQuality.UNVERIFIABLE,
                round(ratio, 4),
                note=f"near match (similarity {ratio:.2f}) rejected: {reason}",
            )
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
