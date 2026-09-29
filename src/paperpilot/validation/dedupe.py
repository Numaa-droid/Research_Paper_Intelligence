"""Merge findings that state the same thing.

Several agents read overlapping parts of a paper, and the coverage sweep reads
what they missed, so the same statement can arrive two or three times in
slightly different words. Repeated, it inflates the grounding counts and makes
the report read like padding. Two findings are the same when the content words
of their claims overlap almost entirely; the first one (registry order, so
specialists before the sweep) is kept and absorbs the others' evidence.
"""

from __future__ import annotations

import re

from ..domain.models import Finding
from .quotes import normalize_for_match

#: Jaccard overlap of content words at or above which two claims are one.
SAME_CLAIM = 0.8

_STOPWORDS = frozenset(
    "a an the and or of to in on for by with as at from is are was were be been it its "
    "this that these those can could may might should would will such not no also than "
    "their there they them which who what when where how into onto over under about".split()
)


def claim_key(claim: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9%.]+", normalize_for_match(claim))
    return frozenset(w.strip(".") for w in words if w not in _STOPWORDS and len(w) > 1)


def same_claim(a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= SAME_CLAIM


def merge_duplicate_findings(findings: list[Finding]) -> list[Finding]:
    """Findings with duplicates folded into their first occurrence."""
    kept: list[tuple[frozenset[str], Finding]] = []
    for finding in findings:
        key = claim_key(finding.claim)
        target = next((f for k, f in kept if same_claim(k, key)), None)
        if target is None:
            kept.append((key, finding))
            continue
        seen = {(e.chunk_id, e.quote[:80]) for e in target.evidence}
        for evidence in finding.evidence:
            if (evidence.chunk_id, evidence.quote[:80]) not in seen:
                seen.add((evidence.chunk_id, evidence.quote[:80]))
                target.evidence.append(evidence)
        target.confidence = max(target.confidence, finding.confidence)
        target.impact = max(target.impact, finding.impact)
    return [finding for _, finding in kept]
