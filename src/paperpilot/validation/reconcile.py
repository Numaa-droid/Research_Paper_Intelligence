"""The last deterministic word on every finding, after the verifier.

The verifier's verdict replaces an agent's self-assessment, but a verifier is a
model, and two things it can miss have exact answers in the paper's text:

* **Whose finding it is.** A verbatim quote of "In [20] the authors found X"
  supports "X" only as work [20]'s finding. Each piece of evidence is attributed
  (see :mod:`paperpilot.discourse.attribution`), and so is the finding. A claim
  that presents as *this paper's* something its evidence only reports from a
  cited work is capped at ``partially_supported``.
* **Whose words it uses.** A claim that uses a term of art the paper never uses
  ("fast-changing fields") goes beyond its evidence however well the rest of it
  is quoted, and is capped at ``partially_supported`` too.

Both caps only ever lower a status and are safe to apply more than once: a
retry re-runs verification, and this pass runs again after it.
"""

from __future__ import annotations

import re

from ..discourse.attribution import (
    attribution_map,
    describe_sources,
    evidence_attribution,
)
from ..domain.enums import FindingStatus
from ..domain.models import Finding
from ..retrieval.index import PaperIndex
from .terms import PaperVocabulary

#: A claim that credits the paper itself with a finding.
_OWN_FRAMING = re.compile(
    r"\b(?:the|this)\s+(?:paper|study|review|article|work|survey|manuscript)(?:'s)?\s+"
    r"(?:\w+\s+){0,2}?(?:shows?|showed|finds?|found|demonstrat\w*|proves?|report\w*|"
    r"reveal\w*|establish\w*|measur\w*|test\w*|evaluat\w*|achiev\w*|conclud\w*|results?|"
    r"experiments?|findings?)\b"
    r"|\bthe\s+authors\s+(?:\w+\s+)?(?:show\w*|f(?:ind|ound)\w*|demonstrat\w*|measur\w*|"
    r"test\w*|evaluat\w*|conduct\w*|ran|run|achiev\w*)\b",
    re.IGNORECASE,
)


def _combine(attributions: list[str]) -> str:
    known = [a for a in attributions if a != "unknown"]
    if not known:
        return "unknown"
    if all(a == "cited" for a in known):
        return "cited"
    if any(a in ("cited", "mixed") for a in known):
        return "mixed"
    return "own"


def attribute_finding(finding: Finding, index: PaperIndex) -> None:
    """Attribute each piece of evidence, then the finding as a whole."""
    amap = attribution_map(index)
    for evidence in finding.evidence:
        kind, sources = evidence_attribution(evidence, amap)
        evidence.attribution = kind
        evidence.cited_sources = list(sources)
    verified = [e for e in finding.evidence if e.is_verified] or finding.evidence
    finding.attribution = _combine([e.attribution or "unknown" for e in verified])  # type: ignore[assignment]
    sources: list[str] = []
    for evidence in verified:
        sources.extend(s for s in evidence.cited_sources if s not in sources)
    finding.cited_sources = sources if finding.attribution in ("cited", "mixed") else []


def _note(finding: Finding, note: str) -> None:
    current = finding.verification_note or ""
    if note not in current:
        finding.verification_note = f"{current} [{note}]".strip() if current else note


def _cap(finding: Finding, note: str) -> None:
    if finding.status is FindingStatus.SUPPORTED:
        finding.status = FindingStatus.PARTIALLY_SUPPORTED
    _note(finding, note)


def reconcile_findings(findings: list[Finding], index: PaperIndex) -> list[Finding]:
    """Attribute every finding and apply the deterministic caps, in place."""
    vocabulary = PaperVocabulary(index)
    for finding in findings:
        attribute_finding(finding, index)
        grounded = finding.status in (FindingStatus.SUPPORTED, FindingStatus.PARTIALLY_SUPPORTED)
        if not grounded:
            continue
        if finding.attribution == "cited" and _OWN_FRAMING.search(finding.claim):
            _cap(
                finding,
                "the claim presents as this paper's own a finding that its evidence reports "
                f"from cited work {describe_sources(finding.cited_sources)}",
            )
        terms = vocabulary.unstated_terms(finding.claim)
        if terms:
            _cap(
                finding,
                "wording not found anywhere in the paper: "
                + ", ".join(f"“{t}”" for t in terms),
            )
    return findings
