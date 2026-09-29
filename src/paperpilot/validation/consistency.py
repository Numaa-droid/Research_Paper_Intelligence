"""Internal consistency of the numbers a paper reports.

Papers contradict themselves: a table says 41.8 and the prose says 41.0. A report
that copies either number alone is wrong by omission. This check needs no model
and no knowledge of the paper's field. For every metric that the results table
reports, it reads each sentence of the paper's prose that states a value for
that metric, and flags a stated value that

* appears in none of the paper's tables and matches no reported result, yet
* is within a few percent of a reported result for the same metric, stated in a
  sentence that names that result's dataset and system.

A value far from every reported result ("by more than 2.0 BLEU") is a
difference, not a restatement, and is left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..domain.enums import ElementType
from ..domain.models import Evidence, Finding, MetricResult
from ..retrieval.index import PaperIndex
from .evidence import validate_evidence
from .numeric import ParsedNumber, extract_numbers, numbers_equal
from .values import terms

#: A restatement differs from the value it restates by at most this much.
MAX_RELATIVE_GAP = 0.05
_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass(slots=True)
class ValueConflict:
    """One result the paper reports with two different values."""

    metric: str
    system: str | None
    dataset: str | None
    reported_value: str
    reported_evidence: list[Evidence]
    stated_value: str
    stated_evidence: Evidence

    def describe(self) -> str:
        subject = " ".join(p for p in (self.system, self.metric) if p)
        where = f" on {self.dataset}" if self.dataset else ""
        stated_at = f"p. {self.stated_evidence.page}"
        if self.stated_evidence.section:
            stated_at += f", §{self.stated_evidence.section}"
        return (
            f"The paper is internally inconsistent: {subject}{where} is reported as "
            f"{self.reported_value}, but the text on {stated_at} states "
            f"{self.stated_value} (\"{self.stated_evidence.quote[:160]}\")."
        )

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "system": self.system,
            "dataset": self.dataset,
            "reported_value": self.reported_value,
            "reported_evidence": [e.model_dump(mode="json") for e in self.reported_evidence],
            "stated_value": self.stated_value,
            "stated_evidence": self.stated_evidence.model_dump(mode="json"),
            "description": self.describe(),
        }


def _metric_patterns(metric: str) -> list[re.Pattern]:
    names = {metric.strip()}
    # "BLEU score" is also written "BLEU"; acronyms are the stable part of a name.
    names |= {w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]+", metric) if sum(c.isupper() for c in w) >= 2}
    patterns = []
    for name in sorted(n for n in names if n):
        escaped = re.escape(name)
        patterns.append(
            re.compile(rf"\b{escaped}\b(?:[^\d\n.;]{{0,24}}?)({_NUMBER})(?![\d.]*\s*[x×·])", re.I)
        )
        patterns.append(re.compile(rf"(?<![\d.])({_NUMBER})\s*%?\s*{escaped}\b", re.I))
    return patterns


def _first_number(value: str) -> ParsedNumber | None:
    numbers = extract_numbers(value, skip_trivial=False)
    return numbers[0] if numbers else None


def _decimals(raw: str) -> int:
    return len(raw.split(".", 1)[1]) if "." in raw else 0


def _restates(stated: ParsedNumber, reported: ParsedNumber) -> bool:
    """Close enough to be the same result, but not equal even after rounding."""
    if reported.value == 0 or numbers_equal(stated, reported):
        return False
    if round(reported.value, _decimals(stated.raw)) == stated.value:
        return False  # "41" for 41.2 is rounding, not a contradiction
    return abs(stated.value - reported.value) / abs(reported.value) <= MAX_RELATIVE_GAP


def find_value_conflicts(results: list[MetricResult], index: PaperIndex) -> list[ValueConflict]:
    """Every reported result the paper's own prose restates with a different value."""
    if not results:
        return []
    table_numbers = [
        n
        for chunk in index.by_element_type(ElementType.TABLE)
        for n in extract_numbers(chunk.text, skip_trivial=False)
    ]
    by_metric: dict[str, list[tuple[MetricResult, ParsedNumber]]] = {}
    for result in results:
        number = _first_number(result.value)
        if number is not None:
            by_metric.setdefault(result.metric.strip().lower(), []).append((result, number))

    prose = [
        c
        for c in index.chunks
        if c.element_type is ElementType.PARAGRAPH
        and not (c.section or "").lower().startswith(("reference", "bibliograph"))
    ]
    conflicts: list[ValueConflict] = []
    seen: set[tuple] = set()
    for metric_key, reported in by_metric.items():
        patterns = _metric_patterns(reported[0][0].metric)
        known = [number for _, number in reported] + table_numbers
        for chunk in prose:
            for sentence in _SENTENCE.split(" ".join(chunk.text.split())):
                sentence_terms = terms(sentence)
                for pattern in patterns:
                    for match in pattern.finditer(sentence):
                        stated = _first_number(match.group(1))
                        if stated is None or any(numbers_equal(stated, k) for k in known):
                            continue
                        conflict = _best_candidate(stated, reported, sentence_terms)
                        if conflict is None:
                            continue
                        result, _ = conflict
                        key = (metric_key, result.system, result.value, stated.raw)
                        if key in seen:
                            continue
                        seen.add(key)
                        stated_evidence = Evidence(
                            page=chunk.page,
                            section=chunk.citation_section,
                            source_type=chunk.element_type,
                            element_id=chunk.element_id,
                            chunk_id=chunk.chunk_id,
                            quote=sentence,
                            confidence=0.9,
                        )
                        validate_evidence(stated_evidence, index)
                        conflicts.append(
                            ValueConflict(
                                metric=result.metric,
                                system=result.system,
                                dataset=result.dataset,
                                reported_value=result.value,
                                reported_evidence=list(result.evidence),
                                stated_value=stated.raw,
                                stated_evidence=stated_evidence,
                            )
                        )
    return conflicts


def _best_candidate(
    stated: ParsedNumber,
    reported: list[tuple[MetricResult, ParsedNumber]],
    sentence_terms: set[str],
) -> tuple[MetricResult, ParsedNumber] | None:
    ranked: list[tuple[int, float, int]] = []
    for position, (result, number) in enumerate(reported):
        if not _restates(stated, number):
            continue
        dataset = terms(result.dataset)
        if dataset and sum(1 for t in dataset if t in sentence_terms) * 2 < len(dataset):
            continue
        system_overlap = sum(1 for t in terms(result.system) if t in sentence_terms)
        if result.system and system_overlap == 0:
            continue
        gap = abs(stated.value - number.value)
        ranked.append((-system_overlap, gap, position))
    if not ranked:
        return None
    return reported[min(ranked)[2]]


def annotate_conflicts(findings: list[Finding], conflicts: list[dict]) -> None:
    """Attach each conflict to the findings that state either of its values.

    Idempotent, so it can run again after a retry re-verifies the findings.
    """
    for conflict in conflicts:
        metric = (conflict.get("metric") or "").lower()
        values = [
            n
            for raw in (conflict.get("reported_value"), conflict.get("stated_value"))
            if (n := _first_number(raw or "")) is not None
        ]
        note = conflict.get("description") or ""
        for finding in findings:
            if metric and metric not in finding.claim.lower():
                continue
            claim_numbers = extract_numbers(finding.claim, skip_trivial=False)
            if not any(numbers_equal(c, v) for c in claim_numbers for v in values):
                continue
            existing = finding.verification_note or ""
            if note and note not in existing:
                finding.verification_note = f"{existing} {note}".strip()
