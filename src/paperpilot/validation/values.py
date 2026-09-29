"""Locating a reported value in the paper.

A model that reports "Transformer (base) BLEU 27.3" and cites a table caption, or
quotes the sentence around a table instead of the table, has still reported a
real number. Whether the number is real is decided here, without a model: the
value must appear on one line of the paper together with the words that name
what it measures — a table row that holds both the system's name and the value
is exactly the evidence a careful reader would point to.
"""

from __future__ import annotations

import re

from ..domain.enums import ElementType
from ..domain.models import Evidence, PaperChunk
from ..retrieval.index import PaperIndex
from .numeric import extract_numbers, numbers_equal

_TOKEN = re.compile(r"[a-z0-9]+")
#: Bracketed reference numbers ("ByteNet [18]") name a citation, not the system.
_REFERENCE_MARK = re.compile(r"\[\d{1,3}(?:\s*,\s*\d{1,3})*\]")
_STOPWORDS = frozenset(
    {"a", "an", "the", "of", "and", "or", "on", "in", "for", "with", "to", "by", "at", "our",
     "model", "models", "method", "ours", "set", "test", "score"}
)
EVIDENCE_CONFIDENCE = 0.9


def terms(text: str | None) -> set[str]:
    """Content words of a name, for matching it against a line of the paper."""
    cleaned = _REFERENCE_MARK.sub(" ", (text or "").lower())
    return {t for t in _TOKEN.findall(cleaned) if t not in _STOPWORDS}


def _share(wanted: set[str], line: str) -> float:
    if not wanted:
        return 1.0
    present = terms(line)
    return sum(1 for t in wanted if t in present) / len(wanted)


def value_in_text(value: str, text: str) -> bool:
    """True when the first number of ``value`` appears as a number in ``text``."""
    numbers = extract_numbers(value, skip_trivial=False)
    if not numbers:
        return False
    return any(numbers_equal(numbers[0], n) for n in extract_numbers(text, skip_trivial=False))


def _element_chunks(chunks: list[PaperChunk], index: PaperIndex) -> list[PaperChunk]:
    """The chunks plus every other chunk of the elements they belong to."""
    out: list[PaperChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        members = index.by_element_id(chunk.element_id) if chunk.element_id else [chunk]
        for member in members:
            if member.chunk_id not in seen:
                seen.add(member.chunk_id)
                out.append(member)
    return out


def find_value_line(
    value: str,
    candidates: list[PaperChunk],
    *,
    name_terms: set[str],
    min_share: float,
) -> tuple[PaperChunk, str] | None:
    """The line holding ``value`` that best matches ``name_terms``.

    Lines are ranked by the share of the name's words they contain, then by
    document order, so the result does not depend on anything but the paper.
    """
    best: tuple[float, int, PaperChunk, str] | None = None
    for position, chunk in enumerate(candidates):
        for line in chunk.text.split("\n"):
            if not line.strip() or not value_in_text(value, line):
                continue
            share = _share(name_terms, line)
            if share < min_share:
                continue
            if best is None or share > best[0]:
                best = (share, position, chunk, line.strip())
    return (best[2], best[3]) if best else None


def locate_value(
    value: str,
    index: PaperIndex,
    *,
    name: str | None,
    cited: list[PaperChunk],
) -> Evidence | None:
    """Evidence for a reported value, searched from the cited sources outwards.

    1. The cited chunks and the rest of their tables/figures: the value with at
       least half of the system's name on the same line (any line holding the
       value when no name was given).
    2. Every table, then every paragraph, of the paper: the value with the whole
       name on the same line.
    """
    name_terms = terms(name)
    searches: list[tuple[list[PaperChunk], float]] = [(_element_chunks(cited, index), 0.5)]
    if name_terms:
        content = [c for c in index.chunks if c.element_type is not ElementType.REFERENCE]
        searches.append(([c for c in content if c.element_type is ElementType.TABLE], 1.0))
        searches.append(([c for c in content if c.element_type is not ElementType.TABLE], 1.0))

    for candidates, min_share in searches:
        found = find_value_line(value, candidates, name_terms=name_terms, min_share=min_share)
        if found is not None:
            chunk, line = found
            return Evidence(
                page=chunk.page,
                section=chunk.citation_section,
                source_type=chunk.element_type,
                element_id=chunk.element_id,
                chunk_id=chunk.chunk_id,
                quote=line[:500],
                confidence=EVIDENCE_CONFIDENCE,
                validation_notes=[
                    f"value '{value}' located in the paper on the line that names it"
                ],
            )
    return None
