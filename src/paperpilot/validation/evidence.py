"""Deterministic validation of evidence and findings.

This is the gate every agent output passes through before it can reach the
report. None of it uses a model: page ranges, chunk existence, section labels,
element ids and quote presence all have exact answers in the index.

The policy is to *annotate and downgrade* rather than to delete. A citation that
cannot be verified stays visible in the report's evidence appendix marked as
unverifiable, because silently dropping it would hide the failure from the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..domain.enums import CitationQuality, ElementType, FindingStatus
from ..domain.models import Evidence, Finding, PaperChunk
from ..retrieval.index import PaperIndex
from .numeric import check_claim_numbers, extract_numbers, numbers_equal, numeric_accuracy
from .quotes import QuoteMatch, match_quote, normalize_for_match
from .values import locate_value


@dataclass(slots=True)
class EvidenceValidation:
    """Report for one piece of evidence."""

    evidence: Evidence
    ok: bool
    notes: list[str] = field(default_factory=list)

    @property
    def quality(self) -> CitationQuality:
        return self.evidence.citation_quality


@dataclass(slots=True)
class FindingValidation:
    """Report for one finding and all of its evidence."""

    finding: Finding
    evidence_reports: list[EvidenceValidation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def verified_evidence_count(self) -> int:
        return sum(1 for r in self.evidence_reports if r.ok)

    @property
    def has_verified_evidence(self) -> bool:
        return self.verified_evidence_count > 0


def resolve_evidence_chunk(evidence: Evidence, index: PaperIndex) -> str | None:
    """Find the chunk a quote came from when the agent did not say.

    Improves citation precision: an unattributed quote that is present on exactly
    one chunk of the cited page can be attributed with certainty.
    """
    if evidence.chunk_id:
        return evidence.chunk_id
    candidates = [c for c in index.page(evidence.page) if match_quote(evidence.quote, c.text).found]
    if len(candidates) == 1:
        return candidates[0].chunk_id
    return None


def locate_quote(
    quote: str, index: PaperIndex, *, near_page: int | None = None
) -> tuple[PaperChunk, QuoteMatch] | None:
    """Find the chunk of the paper a quote comes from.

    Every chunk is searched for a verbatim or normalised match, in document
    order; fuzzy matching, which is slower, is limited to the pages next to
    ``near_page``, where a mis-cited quote most often really sits.
    """
    for chunk in index.chunks:
        match = match_quote(quote, chunk.text, fuzzy=False)
        if match.found:
            return chunk, match
    if near_page is None:
        return None
    for page in (near_page - 1, near_page + 1):
        for chunk in index.page(page):
            match = match_quote(quote, chunk.text)
            if match.found:
                return chunk, match
    return None


_SEPARATOR_ROW = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")
#: Rows substituted for a paraphrased table reading, at most.
_MAX_TABLE_ROWS = 8
#: "Table 3", "Figure 2", "Section 6.2", "Eq. (1)": names of things, not values.
_ELEMENT_REFERENCE = re.compile(
    r"\b(?:tables?|figures?|figs?\.|sections?|§|eqs?\.?|equations?|rows?|columns?)\s*\(?[\dA-Z][\w.]*\)?",
    re.IGNORECASE,
)


def _table_chunks(chunk: PaperChunk | None, page: int, index: PaperIndex) -> list[PaperChunk]:
    """The table a quote cites: the cited chunk's table, else the tables on its page."""
    if chunk is not None and chunk.element_type is ElementType.TABLE and chunk.element_id:
        return index.by_element_id(chunk.element_id)
    return [c for c in index.page(page) if c.element_type is ElementType.TABLE]


def match_table_quote(
    evidence: Evidence, chunk: PaperChunk | None, index: PaperIndex
) -> tuple[PaperChunk, QuoteMatch] | None:
    """Accept a quote that reads a table the way a table is read.

    Two deterministic cases, both limited to the cited table:

    * **Rows.** Every quoted line is a row of the table. Rows need not be
      adjacent: quoting the header and the two rows compared is faithful.
    * **Values.** The quote paraphrases the table ("Table 4 shows F1 91.3 …")
      and every number in it, at least two, is a cell of the table. The quote
      is replaced by the table rows that hold those numbers, verbatim, and the
      substitution is recorded.

    A quote with any number the table does not contain matches neither.
    """
    tables = _table_chunks(chunk, evidence.page, index)
    if not tables:
        return None
    rows = [
        (table, line.strip())
        for table in tables
        for line in table.text.split("\n")
        if line.strip() and not _SEPARATOR_ROW.match(line.strip())
    ]
    normalized_rows = [normalize_for_match(line) for _, line in rows]

    quoted = [
        normalize_for_match(line)
        for line in evidence.quote.split("\n")
        if line.strip() and not _SEPARATOR_ROW.match(line.strip())
    ]
    quoted = [line for line in quoted if line]
    if len(quoted) >= 2 and all(len(q) >= 8 for q in quoted) and all(any(q in row for row in normalized_rows) for q in quoted):
        table = next(t for (t, _), row in zip(rows, normalized_rows, strict=True) if quoted[0] in row)
        return table, QuoteMatch(
            CitationQuality.NORMALIZED, 1.0, note="quoted rows found in the cited table"
        )

    numbers = extract_numbers(_ELEMENT_REFERENCE.sub(" ", evidence.quote))
    table_numbers = [n for _, line in rows for n in extract_numbers(line, skip_trivial=False)]
    if len(numbers) < 2 or not all(
        any(numbers_equal(n, t) for t in table_numbers) for n in numbers
    ):
        return None
    holding = [
        (table, line)
        for table, line in rows
        if any(numbers_equal(n, c) for n in numbers for c in extract_numbers(line, skip_trivial=False))
    ][:_MAX_TABLE_ROWS]
    if not holding:
        return None
    evidence.quote = "\n".join(line for _, line in holding)
    return holding[0][0], QuoteMatch(
        CitationQuality.NORMALIZED,
        1.0,
        note=(
            "the quote paraphrased the table; replaced by the table rows holding its "
            f"{len(numbers)} number(s), all of which are in the table"
        ),
    )


def validate_evidence(evidence: Evidence, index: PaperIndex) -> EvidenceValidation:
    """Check one evidence object against the index, mutating it in place.

    Sets ``citation_quality``, appends human-readable ``validation_notes`` and
    scales ``confidence`` down to match the quality actually achieved.
    """
    notes: list[str] = []
    base_confidence = evidence.confidence

    # 1. The page must exist in this document.
    if not index.has_page(evidence.page):
        evidence.citation_quality = CitationQuality.UNVERIFIABLE
        evidence.confidence = 0.0
        note = (
            f"page {evidence.page} is outside the document "
            f"(1-{index.page_count}); citation rejected"
        )
        evidence.validation_notes.append(note)
        return EvidenceValidation(evidence=evidence, ok=False, notes=[note])

    # 2. Resolve the text this quote should be found in. An agent that quoted
    #    correctly but did not say which chunk it came from can still be
    #    attributed, as long as the attribution is unambiguous.
    if not evidence.chunk_id:
        resolved = resolve_evidence_chunk(evidence, index)
        if resolved:
            evidence.chunk_id = resolved
    chunk = index.get(evidence.chunk_id) if evidence.chunk_id else None
    if evidence.chunk_id and chunk is None:
        notes.append(f"chunk_id '{evidence.chunk_id}' does not exist; falling back to page text")
        evidence.chunk_id = None

    if chunk is not None and chunk.page != evidence.page:
        notes.append(
            f"cited page {evidence.page} does not match chunk page {chunk.page}; "
            f"corrected to {chunk.page}"
        )
        evidence.page = chunk.page

    source_text = chunk.text if chunk is not None else index.page_text(evidence.page)

    # 3. The quote must actually be there.
    match = match_quote(evidence.quote, source_text)
    if not match.found and chunk is not None:
        # The agent may have cited the right page but the wrong chunk.
        page_match = match_quote(evidence.quote, index.page_text(evidence.page))
        if page_match.found:
            notes.append(
                f"quote was not in chunk '{chunk.chunk_id}' but was found elsewhere on "
                f"page {evidence.page}; chunk reference dropped"
            )
            evidence.chunk_id = None
            chunk = None
            match = page_match

    if not match.found:
        table_match = match_table_quote(evidence, chunk, index)
        if table_match is not None:
            table_chunk, match = table_match
            if chunk is None or chunk.element_id != table_chunk.element_id:
                chunk = table_chunk
                evidence.chunk_id = table_chunk.chunk_id
            if match.note:
                notes.append(match.note)
                match.note = None

    if not match.found:
        # A real quote attributed to the wrong source is a mis-location, not an
        # invention: find where it actually is and correct the citation.
        located = locate_quote(evidence.quote, index, near_page=evidence.page)
        if located is not None:
            found_chunk, found_match = located
            notes.append(
                f"quote was not on the cited page {evidence.page}; found on page "
                f"{found_chunk.page} and the citation was corrected"
            )
            evidence.page = found_chunk.page
            evidence.chunk_id = found_chunk.chunk_id
            evidence.section = found_chunk.citation_section
            evidence.element_id = found_chunk.element_id
            evidence.source_type = found_chunk.element_type
            chunk = found_chunk
            match = found_match

    evidence.citation_quality = match.quality
    evidence.confidence = match.adjusted_confidence(base_confidence)
    if match.note:
        notes.append(match.note)

    # 4. Section and element references must be consistent with the index.
    if chunk is not None:
        actual_section = chunk.citation_section
        if evidence.section and actual_section and evidence.section.lower() not in (
            actual_section.lower(),
            (chunk.section or "").lower(),
        ):
            notes.append(
                f"section '{evidence.section}' corrected to '{actual_section}' from the index"
            )
            evidence.section = actual_section
        elif not evidence.section:
            evidence.section = actual_section

        if chunk.element_id:
            if evidence.element_id != chunk.element_id:
                if evidence.element_id:
                    notes.append(
                        f"element '{evidence.element_id}' corrected to '{chunk.element_id}'"
                    )
                evidence.element_id = chunk.element_id
        elif evidence.element_id:
            # The quote resolved to a chunk that is not part of any element, so
            # the claimed element reference cannot be right.
            notes.append(
                f"element '{evidence.element_id}' does not belong to the cited text; dropped"
            )
            evidence.element_id = None
        if evidence.source_type is ElementType.UNKNOWN:
            evidence.source_type = chunk.element_type
    elif evidence.element_id:
        known = {c.element_id for c in index.chunks if c.element_id}
        if evidence.element_id not in known:
            notes.append(f"element '{evidence.element_id}' does not exist in this paper; dropped")
            evidence.element_id = None

    evidence.validation_notes.extend(notes)
    return EvidenceValidation(evidence=evidence, ok=match.found, notes=notes)


def validate_finding(finding: Finding, index: PaperIndex) -> FindingValidation:
    """Validate a finding's evidence and reconcile its status with the result.

    Status policy, applied after evidence checking:

    * no verified evidence at all -> ``unsupported`` (unless the agent already
      said ``inferred``, which is an honest self-report and is preserved);
    * some evidence verified but a cited number is missing from it ->
      ``partially_supported``;
    * otherwise the agent's own status is kept.
    """
    reports: list[EvidenceValidation] = []
    notes: list[str] = []

    for evidence in finding.evidence:
        reports.append(validate_evidence(evidence, index))

    verified = [r for r in reports if r.ok]

    if not finding.evidence:
        if finding.status not in (FindingStatus.INFERRED, FindingStatus.UNKNOWN):
            notes.append("no evidence supplied; status downgraded to unsupported")
            finding.status = FindingStatus.UNSUPPORTED
        finding.confidence = min(finding.confidence, 0.3)
    elif not verified:
        if finding.status is not FindingStatus.INFERRED:
            notes.append(
                "none of the supplied quotes could be found in the paper; "
                "status downgraded to unsupported"
            )
            finding.status = FindingStatus.UNSUPPORTED
        finding.confidence = min(finding.confidence, 0.25)
    else:
        checks = check_claim_numbers(finding.claim, [r.evidence for r in verified])
        missing = [c.value for c in checks if not c.found_in_evidence]
        if missing:
            # A number read off a cited table is in that table, not in the
            # sentence quoted beside it: look in the cited elements' own lines.
            cited = [c for r in verified if r.evidence.chunk_id and (c := index.get(r.evidence.chunk_id))]
            for value in missing:
                located = locate_value(value, index, name=None, cited=cited)
                if located is not None and validate_evidence(located, index).ok:
                    finding.evidence.append(located)
                    reports.append(EvidenceValidation(evidence=located, ok=True))
            verified = [r for r in reports if r.ok]
            checks = check_claim_numbers(finding.claim, [r.evidence for r in verified])
        accuracy = numeric_accuracy(checks)
        if accuracy < 1.0:
            missing = [c.value for c in checks if not c.found_in_evidence]
            notes.append(
                "numbers not found in the cited evidence: " + ", ".join(missing)
            )
            if finding.status is FindingStatus.SUPPORTED:
                finding.status = FindingStatus.PARTIALLY_SUPPORTED
            finding.confidence = round(finding.confidence * accuracy, 4)

        # A finding can never be more confident than its best evidence.
        ceiling = max(r.evidence.confidence for r in verified)
        if finding.confidence > ceiling:
            finding.confidence = ceiling

    if notes:
        finding.verification_note = "; ".join(notes)
    return FindingValidation(finding=finding, evidence_reports=reports, notes=notes)


def validate_findings(findings: list[Finding], index: PaperIndex) -> list[FindingValidation]:
    return [validate_finding(f, index) for f in findings]


def drop_unverifiable(evidence: list[Evidence]) -> list[Evidence]:
    """Keep only evidence that survived validation. Used for citation rendering."""
    return [e for e in evidence if e.is_verified]
