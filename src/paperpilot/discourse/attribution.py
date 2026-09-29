"""Which sentences of a paper report another work, and which work.

The paper is read once, in document order, as a sequence of sentences. A
sentence *opens* a report when its wording hands the floor to a cited work:
"In [20] the authors …", "According to [18] …", "Smith et al. [4] show …". The
report then *carries* over the following sentences ("They also found …",
"According to them …") until something ends it:

* a new report opens (the floor passes to another work);
* the paper speaks for itself again ("we", "our", "in this article");
* the section changes;
* several sentences pass with nothing tying them back to the cited work.

A sentence with an inline citation but no reporting wording ("attention [3] is
widely used") stays in the paper's own voice: the paper asserts it, with a
reference. Only sentences that *report* a work are attributed to it, so the
attribution errs towards the paper, never towards a cited work.

Everything here is a string question over the index. The same paper always gets
the same map.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ..domain.enums import ElementType
from ..domain.models import Attribution, Evidence, PaperChunk
from ..retrieval.index import PaperIndex
from ..validation.quotes import normalize_for_match

Voice = Literal["own", "reported", "neutral"]

#: Sentences a report may carry over without anything tying them back to it.
MAX_CARRY = 6
#: A sentence fragment shorter than this (normalised) is not matched against quotes.
MIN_SENTENCE_CHARS = 12
#: Word n-gram size for overlap between a quote and a sentence.
SHINGLE = 4
#: Share of a sentence's (or a quote's) shingles the other must contain.
OVERLAP = 0.6

_NON_CONTENT = ("reference", "bibliograph", "acknowledg")

_NUMERIC_CITE = r"\[\s*\d{1,4}(?:\s*[-–,;]\s*\d{1,4})*\s*\]"
_NAME = r"[A-Z][A-Za-z'’\-]+"
_AUTHOR_YEAR = rf"\(\s*{_NAME}(?:\s+et\s+al\.?|\s+(?:and|&)\s+{_NAME})?,?\s+\d{{4}}[a-z]?\s*\)"
_NAMED_CITE = rf"{_NAME}(?:\s+et\s+al\.?|\s+(?:and|&)\s+{_NAME})?\s*(?:{_NUMERIC_CITE}|\(\s*\d{{4}}[a-z]?\s*\))"
_REF = rf"(?:{_NUMERIC_CITE}|{_AUTHOR_YEAR}|{_NAMED_CITE})"

_VERB = (
    r"(?:show|shows|showed|shown|find|finds|found|report|reports|reported|propose|proposes|"
    r"proposed|present|presents|presented|argue|argues|argued|state|states|stated|describe|"
    r"describes|described|demonstrate|demonstrates|demonstrated|suggest|suggests|suggested|"
    r"investigate|investigates|investigated|use|uses|used|conclude|concludes|concluded|"
    r"introduce|introduces|introduced|analy[sz]e|analy[sz]es|analy[sz]ed|note|notes|noted|"
    r"highlight|highlights|highlighted|claim|claims|claimed|develop|develops|developed|"
    r"evaluate|evaluates|evaluated|examine|examines|examined|explore|explores|explored|"
    r"discuss|discusses|discussed|observe|observes|observed|compare|compares|compared|"
    r"test|tests|tested|train|trains|trained|achieve|achieves|achieved|implement|"
    r"implements|implemented|recommend|recommends|recommended|point|points|pointed)"
)

#: Sentence-initial wording that hands the floor to a cited work.
_OPENINGS = (
    re.compile(rf"^\W*(?:In|From|Within)\s+({_REF})", re.IGNORECASE),
    re.compile(rf"^\W*According\s+to\s+({_REF})", re.IGNORECASE),
    re.compile(
        rf"^\W*({_REF})\s*,?\s+(?:(?:the|these|their|its)\s+)?"
        r"(?:authors?|researchers?|stud(?:y|ies)|paper|work|they|he|she)\b",
        re.IGNORECASE,
    ),
    re.compile(rf"^\W*({_REF})\s*,?\s+(?:also\s+)?{_VERB}\b"),
    re.compile(rf"\bthe\s+authors?\s+(?:of|in)\s+({_REF})", re.IGNORECASE),
)
#: Mid-sentence wording that attributes only the sentence it is in.
_ONE_OFF = re.compile(
    rf"\bas\s+(?:shown|reported|demonstrated|described|noted|argued|found|proposed|stated)"
    rf"\s+(?:in|by)\s+({_REF})",
    re.IGNORECASE,
)
#: Wording that ties a sentence back to the work being reported.
_ANAPHORA = re.compile(
    rf"\baccording\s+to\s+(?:them|him|her|their|the\s+authors?)\b"
    rf"|\bthe\s+authors?\b|\bthey\b|\btheir\s+(?:study|analysis|results?|findings|work|"
    rf"approach|experiments?|model|method)\b|\b(?:he|she)\s+(?:also\s+)?{_VERB}\b",
    re.IGNORECASE,
)
#: The paper speaking for itself. "This analysis" or "this study" can as well
#: be the cited work's, so those end a report only when nothing ties the
#: sentence back to it ("In this analysis, he highlights …" is still [24]).
_OWN_VOICE = re.compile(
    r"\b(?:[Ww]e|[Oo]ur|[Oo]urs)\b"
    r"|\b[Tt]his\s+(?:paper|article|review|study|work|material|chapter|survey|manuscript|"
    r"research|section|analysis)\b"
    r"|\b[Tt]he\s+present\s+(?:paper|study|work|article)\b"
)

_ABBREVIATION = re.compile(
    r"(?:\b(?:al|e\.g|i\.e|cf|vs|etc|Fig|Figs|Eq|Eqs|No|Nos|Sec|Ref|Refs|Dr|Mr|Ms|Prof|approx|resp)"
    r"|\b[A-Z])\.$"
)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[]?[A-Z0-9])")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
_REFERENCE_ENTRY = re.compile(r"\[(\d{1,4})\]\s+[A-Z]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?%?")
_WORD = re.compile(r"[a-z][a-z\-]{3,}")
_STOPWORDS = frozenset(
    "with from that this these those their there which where when what were have been "
    "into onto over under about also than then them they such used using some more most "
    "only other each both between within without through against paper study".split()
)


@dataclass(frozen=True, slots=True)
class SentenceVoice:
    """One sentence of the paper and who is speaking in it."""

    chunk_id: str
    page: int
    section: str | None
    text: str
    norm: str
    voice: Voice
    #: The cited works a ``reported`` sentence reports, e.g. ``("[20]",)``.
    sources: tuple[str, ...] = ()


@dataclass(slots=True)
class AttributionMap:
    """Every content sentence of one paper, with its voice."""

    sentences: list[SentenceVoice] = field(default_factory=list)
    by_chunk: dict[str, list[SentenceVoice]] = field(default_factory=dict)
    by_page: dict[int, list[SentenceVoice]] = field(default_factory=dict)
    #: Sentences outside the front matter, the denominator of ``reported_share``.
    body_sentences: int = 0

    @property
    def reported(self) -> list[SentenceVoice]:
        return [s for s in self.sentences if s.voice == "reported"]

    @property
    def reported_share(self) -> float:
        body = [s for s in self.reported if (s.section or "").lower() != "front matter"]
        return round(len(body) / self.body_sentences, 4) if self.body_sentences else 0.0

    @property
    def cited_sources(self) -> list[str]:
        seen: list[str] = []
        for sentence in self.reported:
            for source in sentence.sources:
                if source not in seen:
                    seen.append(source)
        return seen

    def chunk_attribution(self, chunk_id: str) -> tuple[Attribution, tuple[str, ...]]:
        """How a whole chunk is attributed: what an agent's context header shows."""
        return _combine(self.by_chunk.get(chunk_id, []))


def _ref_labels(raw: str) -> tuple[str, ...]:
    """``[1, 2]`` -> ``[1]``, ``[2]``; ``[3-5]`` -> ``[3]``…``[5]``; names kept as written."""
    raw = " ".join(raw.split())
    if not raw.startswith("[") and not re.search(_NUMERIC_CITE, raw):
        return (raw.strip("() "),)
    labels: list[str] = []
    for match in re.finditer(r"(\d{1,4})\s*(?:[-–]\s*(\d{1,4}))?", raw):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end < start or end - start > 20:
            end = start
        labels.extend(f"[{n}]" for n in range(start, end + 1))
    if not raw.startswith("["):
        # "Smith et al. [4]": the number identifies the work.
        return tuple(labels) or (raw,)
    return tuple(labels)


def _valid(labels: tuple[str, ...], max_reference: int | None) -> bool:
    numbers = [int(label[1:-1]) for label in labels if label.startswith("[") and label[1:-1].isdigit()]
    if any(n < 1 for n in numbers):
        return False  # "[0, 1]" is an interval, not a citation
    return not (max_reference and any(n > max_reference for n in numbers))


def split_sentences(text: str) -> list[str]:
    """Sentences of a chunk, keeping "et al." and "Fig. 3" intact."""
    flat = " ".join((text or "").split())
    if not flat:
        return []
    pieces = _SENTENCE_BREAK.split(flat)
    sentences: list[str] = []
    for piece in pieces:
        if sentences and _ABBREVIATION.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return [s for s in sentences if s.strip()]


def _max_reference(index: PaperIndex) -> int | None:
    numbers = [
        int(n)
        for chunk in index.chunks
        if chunk.element_type is ElementType.REFERENCE
        or (chunk.section or "").lower().startswith(("reference", "bibliograph"))
        for n in _REFERENCE_ENTRY.findall(chunk.text)
    ]
    return max(numbers) if numbers else None


def _is_content(chunk: PaperChunk) -> bool:
    return chunk.element_type is not ElementType.REFERENCE and not (
        chunk.section or ""
    ).lower().startswith(_NON_CONTENT)


def _opening(sentence: str, max_reference: int | None) -> tuple[str, ...] | None:
    for pattern in _OPENINGS:
        match = pattern.search(sentence)
        if match:
            labels = _ref_labels(match.group(1))
            if labels and _valid(labels, max_reference):
                return labels
    return None


def build_attribution_map(index: PaperIndex) -> AttributionMap:
    """Read the paper once, in document order, and record each sentence's voice."""
    amap = AttributionMap()
    max_reference = _max_reference(index)
    current: tuple[str, ...] = ()
    carried = 0
    section: str | None = None

    for chunk in index.chunks:
        if not _is_content(chunk):
            continue
        if chunk.citation_section != section:
            section, current, carried = chunk.citation_section, (), 0
        scan = chunk.element_type not in (ElementType.TABLE, ElementType.EQUATION)
        for text in split_sentences(chunk.text):
            voice: Voice = "own" if not scan else "neutral"
            sources: tuple[str, ...] = ()
            if scan:
                opened = _opening(text, max_reference)
                if opened:
                    current, carried = opened, 0
                    voice, sources = "reported", opened
                elif current and _ANAPHORA.search(text):
                    carried = 0
                    voice, sources = "reported", current
                elif _OWN_VOICE.search(text):
                    current, carried = (), 0
                    voice = "own"
                elif current:
                    carried += 1
                    if carried > MAX_CARRY:
                        current, carried = (), 0
                    else:
                        voice, sources = "reported", current
                if voice != "reported" and (one_off := _ONE_OFF.search(text)):
                    labels = _ref_labels(one_off.group(1))
                    if _valid(labels, max_reference):
                        voice, sources = "reported", labels
            sentence = SentenceVoice(
                chunk_id=chunk.chunk_id,
                page=chunk.page,
                section=chunk.citation_section,
                text=text,
                norm=normalize_for_match(text),
                voice=voice,
                sources=sources,
            )
            amap.sentences.append(sentence)
            amap.by_chunk.setdefault(chunk.chunk_id, []).append(sentence)
            amap.by_page.setdefault(chunk.page, []).append(sentence)
            if (chunk.section or "").lower() != "front matter":
                amap.body_sentences += 1
    return amap


def attribution_map(index: PaperIndex) -> AttributionMap:
    """The paper's attribution map, built once per index and cached on it."""
    cached = index.__dict__.get("_attribution_map")
    if cached is not None and cached[0] == len(index.chunks):
        return cached[1]
    amap = build_attribution_map(index)
    index.__dict__["_attribution_map"] = (len(index.chunks), amap)
    return amap


# --- attributing quotes and phrases -----------------------------------------


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = text.split()
    if len(words) < SHINGLE:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def _overlaps(quote: str, sentence: SentenceVoice) -> bool:
    if len(sentence.norm) < MIN_SENTENCE_CHARS or not quote:
        return False
    if sentence.norm in quote or quote in sentence.norm:
        return True
    q, s = _shingles(quote), _shingles(sentence.norm)
    if not q or not s:
        return False
    common = len(q & s)
    return common / len(s) >= OVERLAP or common / len(q) >= OVERLAP


def _combine(sentences: list[SentenceVoice]) -> tuple[Attribution, tuple[str, ...]]:
    if not sentences:
        return "unknown", ()
    reported = [s for s in sentences if s.voice == "reported"]
    sources: list[str] = []
    for sentence in reported:
        sources.extend(x for x in sentence.sources if x not in sources)
    if not reported:
        return "own", ()
    if len(reported) == len(sentences):
        return "cited", tuple(sources)
    return "mixed", tuple(sources)


def quote_attribution(
    quote: str, amap: AttributionMap, *, chunk_id: str | None, page: int | None
) -> tuple[Attribution, tuple[str, ...]]:
    """Attribution of the sentences a quote was copied from."""
    parts = [normalize_for_match(p) for p in _ELLIPSIS.split(quote or "") if p.strip()]
    parts = [p for p in parts if p]
    pools = []
    if chunk_id and chunk_id in amap.by_chunk:
        pools.append(amap.by_chunk[chunk_id])
    if page is not None and page in amap.by_page:
        pools.append(amap.by_page[page])
    for pool in pools:
        matched = [s for s in pool if any(_overlaps(part, s) for part in parts)]
        if matched:
            return _combine(matched)
    return "unknown", ()


def evidence_attribution(
    evidence: Evidence, amap: AttributionMap
) -> tuple[Attribution, tuple[str, ...]]:
    return quote_attribution(evidence.quote, amap, chunk_id=evidence.chunk_id, page=evidence.page)


def text_attribution(text: str, amap: AttributionMap) -> tuple[Attribution, tuple[str, ...]]:
    """Attribution of a short phrase an agent wrote, e.g. a dataset name.

    The phrase is found by its numbers when it has any ("327 published
    articles"), otherwise by its content words. It is attributed to a cited work
    only if every sentence that states it reports that work: when the paper
    also says it in its own voice, it is the paper's.
    """
    numbers = _NUMBER.findall(text or "")
    words = [w for w in _WORD.findall((text or "").lower()) if w not in _STOPWORDS]
    if not numbers and len(words) < 2:
        return "unknown", ()
    matched: list[SentenceVoice] = []
    for sentence in amap.sentences:
        if numbers:
            tokens = set(_NUMBER.findall(sentence.text))
            if all(n in tokens for n in numbers):
                matched.append(sentence)
        else:
            present = sum(1 for w in words if w in sentence.norm)
            if present / len(words) >= OVERLAP:
                matched.append(sentence)
    return _combine(matched)


def describe_sources(sources: tuple[str, ...] | list[str], *, limit: int = 4) -> str:
    """``[20]``, ``[20], [21]`` or ``[20], [21], [22] and 3 more``."""
    items = list(sources)
    if not items:
        return "a cited work"
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")
