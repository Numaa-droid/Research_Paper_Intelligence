"""Numbers stated together must be stated together in the paper.

A model summarising a results table can keep every number right and still pair
them wrong: "h=8 yields PPL 4.91 and BLEU 25.8" uses three numbers the paper
prints, but 4.91 and 25.8 belong to the h=16 row. Checking each number on its
own passes that sentence; checking that the numbers of one clause sit in one
*unit* of the paper (a table row, or a sentence and the one after it) does not.

This is applied to text a model wrote without quoting the paper (ablation
summaries, hyperparameter values, setup and methodology prose), which no quote
check ever sees. It needs no model and no knowledge of the paper's subject:
only where the paper puts its numbers.

Two kinds of number in a clause are context, not a pairing to check, and only
need to occur somewhere in the paper: the ends of a stated range ("h from 1 to
32") and years ("WMT 2014").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..domain.enums import ElementType
from ..retrieval.index import PaperIndex
from .numeric import ParsedNumber, extract_numbers, numbers_equal

#: Reference markers ("[18]", "[5, 2, 35]") name a cited work, not a value.
_CITATION = re.compile(r"\[\d{1,3}(?:\s*[,–-]\s*\d{1,3})*\]")
#: "Table 3", "Section 6.2", "Eq. (1)", "row (E)": names of things, not values.
_ELEMENT_REFERENCE = re.compile(
    r"\b(?:tables?|figures?|figs?\.|sections?|§|eqs?\.?|equations?|rows?|columns?|appendix)"
    r"\s*\(?[\dA-Z][\w.]*\)?",
    re.IGNORECASE,
)
_NUM = r"\d[\d,]*(?:\.\d+)?"
#: "from 1 to 32", "between 2 and 8", "2–8", "25.4-25.8": the ends of a span.
#: A sentence's closing full stop may follow it; a decimal point may not.
_RANGE = re.compile(
    rf"(?:from\s+|between\s+)?(?<![\w.])({_NUM})\s*(to|through|and|–|—|-)\s*({_NUM})(?!\w|\.\d)",
    re.IGNORECASE,
)
_RANGE_CUE = re.compile(r"\b(?:from|between|range|ranging|varying|varied|vary)\b", re.IGNORECASE)
#: A clause is one assertion: split on sentence ends, ";", ", " and line breaks
#: (never on the comma inside "100,000" or the point inside "0.4"), and on the
#: words that set two separately stated values side by side ("25.7 vs 25.8").
_CLAUSE = re.compile(
    r"(?<=[.!?])\s+|;|,\s+|\n|\s(?:vs\.?|versus|compared\s+(?:to|with)|than)\s",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_SEPARATOR_ROW = re.compile(r"^\|?\s*:?-{3,}")
_MAX_NEAREST = 2


def _clean(text: str) -> str:
    text = text.replace("−", "-").replace("^", "").replace("{", "").replace("}", "")
    text = _CITATION.sub(" ", text)
    return _ELEMENT_REFERENCE.sub(" ", text)


def _values(text: str) -> list[ParsedNumber]:
    return extract_numbers(_clean(text), skip_trivial=False)


def _is_year(number: ParsedNumber) -> bool:
    return (
        not number.suffix
        and "." not in number.raw
        and "," not in number.raw
        and 1900 <= number.value <= 2100
    )


def _range_ends(clause: str) -> list[ParsedNumber]:
    """Numbers that bound a stated range in ``clause``."""
    ends: list[ParsedNumber] = []
    for match in _RANGE.finditer(_clean(clause)):
        span, joiner = match.group(0).lower(), match.group(2).lower()
        # "5.29 and 24.9" is a pairing, not a range: "and" counts only after "between".
        if joiner == "and" and not span.startswith("between"):
            continue
        # "25.4-25.8" is a span; "5 - 3" written with spaces may be arithmetic.
        tight_hyphen = joiner == "-" and "-" in match.group(0) and " -" not in match.group(0)
        if joiner in ("to", "through", "–", "—", "and") or tight_hyphen or _RANGE_CUE.search(clause):
            ends.extend(extract_numbers(match.group(1), skip_trivial=False))
            ends.extend(extract_numbers(match.group(3), skip_trivial=False))
    return ends


def _contains(numbers: list[ParsedNumber], value: ParsedNumber) -> bool:
    return any(numbers_equal(value, n) for n in numbers)


def _dedupe(numbers: list[ParsedNumber]) -> list[ParsedNumber]:
    out: list[ParsedNumber] = []
    for n in numbers:
        if not _contains(out, n):
            out.append(n)
    return out


def _describe_row(header: str | None, row: str) -> str:
    """A table row as "column = value" pairs, so a reader can see what each number is."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    if not header:
        return " | ".join(c for c in cells if c)
    names = [c.strip() for c in header.strip().strip("|").split("|")]
    parts: list[str] = []
    for i, cell in enumerate(cells):
        if not cell:
            continue
        name = names[i] if i < len(names) else ""
        parts.append(f"{name} = {cell}" if name else cell)
    return ", ".join(parts)


@dataclass(slots=True)
class Unit:
    """A stretch of the paper whose numbers are stated together."""

    text: str
    page: int
    numbers: list[ParsedNumber]


@dataclass(slots=True)
class Mismatch:
    """A clause whose numbers the paper does not state, or not together."""

    clause: str
    absent: list[str] = field(default_factory=list)
    unpaired: list[str] = field(default_factory=list)
    nearest: list[tuple[str, int]] = field(default_factory=list)

    def describe(self) -> str:
        if self.absent:
            return f"“{self.clause}”: {', '.join(self.absent)} do not appear in the paper"
        text = f"“{self.clause}”: the paper never states {', '.join(self.unpaired)} together"
        if self.nearest:
            text += "; it has " + " and ".join(
                f"{desc} (p. {page})" for desc, page in self.nearest
            )
        return text


class PaperNumbers:
    """Every unit of the paper that states numbers, for checking model-written text."""

    def __init__(self, index: PaperIndex) -> None:
        self.units: list[Unit] = []
        for chunk in index.chunks:
            if chunk.element_type is ElementType.TABLE:
                lines = [ln for ln in chunk.text.split("\n") if ln.strip()]
                header = lines[0] if lines and lines[0].lstrip().startswith("|") else None
                for i, line in enumerate(lines):
                    if _SEPARATOR_ROW.match(line.strip()):
                        continue
                    numbers = _values(line)
                    if numbers:
                        described = _describe_row(header if i > 0 else None, line)
                        self.units.append(Unit(described, chunk.page, numbers))
                continue
            sentences = _SENTENCE.split(" ".join(chunk.text.split()))
            spans = sentences + [f"{a} {b}" for a, b in zip(sentences, sentences[1:], strict=False)]
            for span in spans:
                numbers = _values(span)
                if numbers:
                    self.units.append(Unit(span, chunk.page, numbers))
        self._all = [n for unit in self.units for n in unit.numbers]

    def audit(self, text: str | None) -> list[Mismatch]:
        """Clauses of ``text`` whose numbers the paper does not state, or not together."""
        mismatches: list[Mismatch] = []
        # Citation markers go first, so "[5, 2, 35]" is not split into clauses.
        for raw_clause in _CLAUSE.split(_CITATION.sub(" ", text or "")):
            clause = " ".join(raw_clause.split()).strip(" ()")
            numbers = _dedupe(_values(clause))
            if not numbers:
                continue
            absent = [n for n in numbers if not _contains(self._all, n)]
            if absent:
                mismatches.append(Mismatch(clause, absent=[n.raw + n.suffix for n in absent]))
                continue
            context = _range_ends(clause)
            tied = [n for n in numbers if not _is_year(n) and not _contains(context, n)]
            if len(tied) < 2:
                continue
            if any(all(_contains(unit.numbers, n) for n in tied) for unit in self.units):
                continue
            overlap = [(sum(_contains(u.numbers, n) for n in tied), u) for u in self.units]
            best = max(score for score, _ in overlap)
            nearest = [
                (u.text if len(u.text) <= 220 else u.text[:217] + "…", u.page)
                for score, u in overlap
                if score == best and best >= 2
            ][:_MAX_NEAREST]
            mismatches.append(
                Mismatch(clause, unpaired=[n.raw + n.suffix for n in tied], nearest=nearest)
            )
        return mismatches
