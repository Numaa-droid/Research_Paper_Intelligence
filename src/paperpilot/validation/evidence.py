"""Deterministic validation of evidence and findings.

This is the gate every agent output passes through before it can reach the
report. None of it uses a model: page ranges, chunk existence, section labels,
element ids and quote presence all have exact answers in the index.

The policy is to *annotate and downgrade* rather than to delete. A citation that
cannot be verified stays visible in the report's evidence appendix marked as
unverifiable, because silently dropping it would hide the failure from the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.enums import CitationQuality, ElementType, FindingStatus
from ..domain.models import Evidence, Finding
from ..retrieval.index import PaperIndex
from .numeric import check_claim_numbers, numeric_accuracy
from .quotes import match_quote


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
