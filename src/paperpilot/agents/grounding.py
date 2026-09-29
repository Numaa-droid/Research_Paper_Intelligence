"""Turn cited quotes in a parsed response into typed, located evidence.

The model never writes a page number, section, element id or chunk id. It cites
a short source label (``[S3]``) from the context block, and everything about
where that source lives comes from the index. A model can still misquote, which
:func:`paperpilot.validation.evidence.validate_evidence` catches, but it can no
longer mis-locate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..domain.enums import CitationQuality, FindingStatus
from ..domain.models import Evidence, Finding, PaperChunk
from ..observability import warning
from ..response.parser import normalize_key
from ..response.reader import DocumentReader, QuoteRef, clean_text, parse_score, quotes_from_text
from ..response.schema import Section
from ..retrieval.index import PaperIndex
from ..retrieval.retriever import RetrievalResult
from ..validation.quotes import match_quote, normalize_for_match
from ..validation.schema import coerce_status

#: Starting confidence of a located quote; validation scales it by match quality.
EVIDENCE_CONFIDENCE = 0.9
#: An uncited quote shorter than this is not searched for in the paper.
MIN_UNLABELLED_QUOTE_CHARS = 20

_GENERIC_FINDING_HEADING = re.compile(r"^(?:finding|claim|result)s?(?:\s+\d+)?$")
#: "Finding 3: ", "Claim 2 - ", "1. ", "2) " — but not the "81." of "81.4 F1".
_FINDING_PREFIX = re.compile(r"^\s*(?:(?:finding|claim)\s*\d*|\d{1,3})\s*[:.)\-–—]\s+", re.IGNORECASE)

FINDING_COLUMNS: dict[str, tuple[str, ...]] = {
    "claim": ("statement", "finding"),
    "status": ("support", "verdict"),
    "confidence": ("conf",),
    "impact": ("importance",),
    "category": ("type", "topic"),
    "explanation": ("rationale", "reasoning", "notes"),
    "source": ("sources", "citation", "citations", "ref"),
    "quote": ("evidence", "verbatim quote", "quotes"),
}


def parse_status(text: str, default: FindingStatus = FindingStatus.UNKNOWN) -> FindingStatus:
    """``Partially supported.``, ``partially_supported``, ``**Supported**`` -> enum."""
    key = normalize_key(text)
    if not key:
        return default
    status = coerce_status(key, FindingStatus.UNKNOWN)
    if status is FindingStatus.UNKNOWN and key != "unknown":
        # "supported by table 2" -> the leading status word(s) decide.
        for words in (2, 1):
            status = coerce_status(" ".join(key.split()[:words]), FindingStatus.UNKNOWN)
            if status is not FindingStatus.UNKNOWN:
                return status
        return default
    return status


def source_label(position: int) -> str:
    """Label of the ``position``-th (0-based) retrieved chunk in a prompt."""
    return f"S{position + 1}"


#: "[S3]", "[S3, S4]", "S3" and "S3 and S4" as written in a model's prose.
_LABEL_MENTION = re.compile(
    r"\[\s*(?:sources?\s*:\s*)?S\d{1,4}(?:\s*(?:,|;|and)\s*S\d{1,4})*\s*\]"
    r"|\b(?:sources?\s+)?S\d{1,4}(?:\s*(?:,|and)\s*S\d{1,4})*\b",
    re.IGNORECASE,
)


def resolve_source_labels(text: str, sources: SourceMap) -> str:
    """Replace prompt source labels in prose with the locations they stand for.

    A reader of the report never saw "S1"; "[p. 4, §3.2.1 Scaled Dot-Product
    Attention]" is what the model meant. Labels that were not in the prompt are
    left as written.
    """

    def replace(match: re.Match) -> str:
        places: list[str] = []
        for label in re.findall(r"S\d{1,4}", match.group(0), re.IGNORECASE):
            chunk = sources.chunk(label)
            if chunk is None:
                return match.group(0)
            place = f"p. {chunk.page}" + (
                f", §{chunk.citation_section}" if chunk.citation_section else ""
            )
            if place not in places:
                places.append(place)
        return f"[{'; '.join(places)}]" if places else match.group(0)

    return _LABEL_MENTION.sub(replace, text or "")


class SourceMap:
    """Source labels shown in one prompt, mapped back to the chunks they name."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._chunks: dict[str, PaperChunk] = {
            source_label(i): result.chunk for i, result in enumerate(results)
        }

    def __len__(self) -> int:
        return len(self._chunks)

    def chunk(self, label: str) -> PaperChunk | None:
        return self._chunks.get(label.upper())

    def chunks(self) -> list[PaperChunk]:
        return list(self._chunks.values())


@dataclass(slots=True)
class Grounder:
    """Resolves quotes for one agent response."""

    sources: SourceMap
    index: PaperIndex
    agent: str

    def evidence(self, ref: QuoteRef) -> Evidence | None:
        quote = ref.text.strip()
        if not quote:
            return None
        notes: list[str] = []
        cited = [(label, self.sources.chunk(label)) for label in ref.labels]
        unknown = [label for label, chunk in cited if chunk is None]
        if unknown:
            notes.append(f"cited source {', '.join(unknown)} was not in the prompt")
        known = [chunk for _, chunk in cited if chunk is not None]

        chunk = next((c for c in known if match_quote(quote, c.text).found), None)
        if chunk is None and known:
            chunk = known[0]  # validation will report the quote as not found there
        if chunk is None:
            chunk = self._locate(quote)
            if chunk is None:
                warning(
                    "grounding.quote_dropped",
                    agent=self.agent,
                    reason="no valid source label and the quote was not found in the paper",
                    quote=quote[:120],
                )
                return None
            notes.append("no valid source label; attributed by locating the quote in the paper")

        return Evidence(
            page=chunk.page,
            section=chunk.citation_section,
            source_type=chunk.element_type,
            element_id=chunk.element_id,
            chunk_id=chunk.chunk_id,
            quote=quote,
            confidence=EVIDENCE_CONFIDENCE,
            citation_quality=CitationQuality.UNVERIFIABLE,
            validation_notes=notes,
        )

    def evidence_list(self, refs: list[QuoteRef]) -> list[Evidence]:
        out: list[Evidence] = []
        seen: set[tuple[str | None, str]] = set()
        for ref in refs:
            evidence = self.evidence(ref)
            if evidence is None:
                continue
            key = (evidence.chunk_id, evidence.quote)
            if key not in seen:
                seen.add(key)
                out.append(evidence)
        return out

    def evidence_for_value(self, value: str, labels: list[str]) -> list[Evidence]:
        """Evidence for a value cited without a quote: the cited line containing it.

        Used for table rows such as a reported metric, where copying the source
        line is exactly what a careful reader would do by hand.
        """
        needle = value.strip()
        if not needle:
            return []
        for label in labels:
            chunk = self.sources.chunk(label)
            if chunk is None:
                continue
            line = next((ln.strip() for ln in chunk.text.split("\n") if needle in ln), None)
            if line:
                return [
                    Evidence(
                        page=chunk.page,
                        section=chunk.citation_section,
                        source_type=chunk.element_type,
                        element_id=chunk.element_id,
                        chunk_id=chunk.chunk_id,
                        quote=line[:500],
                        confidence=EVIDENCE_CONFIDENCE,
                        validation_notes=["quote taken from the cited source line"],
                    )
                ]
        return []

    def _locate(self, quote: str) -> PaperChunk | None:
        """Find an unlabelled quote: fuzzy within the prompt's sources, exact elsewhere."""
        if len(quote) < MIN_UNLABELLED_QUOTE_CHARS:
            return None  # too short to attribute with any confidence
        for chunk in self.sources.chunks():
            if match_quote(quote, chunk.text).found:
                return chunk
        needle = normalize_for_match(quote)
        return next((c for c in self.index.chunks if needle in normalize_for_match(c.text)), None)


# --- findings ----------------------------------------------------------------


def parse_findings(
    reader: DocumentReader,
    grounder: Grounder,
    *,
    category: str,
    aliases: tuple[str, ...] = ("findings", "key findings", "claims"),
) -> list[Finding]:
    """Findings written as ``### Finding`` subsections, a table, or a list."""
    section = reader.section(*aliases)
    if section is None:
        return []

    findings: list[Finding] = []
    for sub in reader.subsections(section):
        finding = _finding_from_section(reader, grounder, sub, category)
        if finding is not None:
            findings.append(finding)
    if findings:
        return findings

    for table in reader.tables(section):
        for record in reader.records(table, FINDING_COLUMNS):
            finding = _finding_from_record(grounder, record, category)
            if finding is not None:
                findings.append(finding)
    if findings:
        return findings

    for item in reader.list_items(section):
        if item.key and normalize_key(item.key) not in ("claim", "finding"):
            continue
        claim = clean_text(item.value if item.key else item.text)
        if not claim:
            continue
        refs = [
            ref
            for child in item.children
            for ref in quotes_from_text(getattr(child, "text", "") or "")
            if getattr(child, "type", "") == "blockquote"
        ]
        findings.append(
            _build_finding(
                claim=claim,
                explanation="",
                status_text="",
                confidence=None,
                impact=None,
                category=category,
                evidence=grounder.evidence_list(refs),
                agent=grounder.agent,
            )
        )
    return findings


def _finding_from_section(
    reader: DocumentReader, grounder: Grounder, section: Section, category: str
) -> Finding | None:
    fields = reader.fields(section)
    claim = reader.field_text(fields, "claim", "statement", "finding")
    if not claim and not _GENERIC_FINDING_HEADING.match(section.key):
        claim = clean_text(_FINDING_PREFIX.sub("", section.heading))
    if not claim:
        return None
    confidence = reader.field(fields, "confidence")
    impact = reader.field(fields, "impact", "importance")
    return _build_finding(
        claim=claim,
        explanation=reader.field_text(fields, "explanation", "rationale", "reasoning", "why")
        or reader.paragraphs(section),
        status_text=reader.field_text(fields, "status", "support", "verdict"),
        confidence=parse_score(confidence.value) if confidence else None,
        impact=parse_score(impact.value) if impact else None,
        category=reader.field_text(fields, "category", "type") or category,
        evidence=grounder.evidence_list(reader.quotes(section)),
        agent=grounder.agent,
    )


def _finding_from_record(grounder: Grounder, record: dict, category: str) -> Finding | None:
    def value(name: str) -> str:
        cell = record.get(name)
        return clean_text(cell.value) if cell is not None else ""

    claim = value("claim")
    if not claim:
        return None
    quote_cell, source_cell = record.get("quote"), record.get("source")
    refs: list[QuoteRef] = []
    if quote_cell is not None and quote_cell.text.strip():
        refs = quotes_from_text(quote_cell.text)
        if source_cell is not None and source_cell.citations:
            for ref in refs:
                if not ref.citations:
                    ref.citations = list(source_cell.citations)
    return _build_finding(
        claim=claim,
        explanation=value("explanation"),
        status_text=value("status"),
        confidence=parse_score(value("confidence")),
        impact=parse_score(value("impact")),
        category=value("category") or category,
        evidence=grounder.evidence_list(refs),
        agent=grounder.agent,
    )


def _build_finding(
    *,
    claim: str,
    explanation: str,
    status_text: str,
    confidence: float | None,
    impact: float | None,
    category: str,
    evidence: list[Evidence],
    agent: str,
) -> Finding:
    return Finding(
        claim=claim[:2000],
        explanation=explanation,
        evidence=evidence,
        status=parse_status(status_text),
        confidence=0.5 if confidence is None else confidence,
        impact=0.5 if impact is None else impact,
        category=category or None,
        agent=agent,
    )
