"""Evaluation metrics.

Every metric here is computed **without a model**, from the analysis record and
the paper index. That is what makes the harness provider-independent: the same
numbers are produced whether the run used a hosted model, a local model, or the
deterministic mock.

Metrics that need human-written ground truth degrade to ``None`` when no
expectations file is supplied, rather than inventing a score.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..domain.enums import FindingStatus
from ..domain.models import AnalysisRecord, Evidence
from ..retrieval.index import PaperIndex
from ..validation.numeric import extract_numbers, numbers_equal
from ..validation.quotes import match_quote, normalize_for_match

_WORD = re.compile(r"[a-z0-9]+")

_WEAK = (FindingStatus.UNSUPPORTED, FindingStatus.CONTRADICTED, FindingStatus.UNKNOWN)

_STOPWORDS = {
    "the", "a", "an", "of", "and", "is", "are", "in", "on", "to", "for", "that",
    "with", "by", "from", "it", "as", "at", "be", "this", "was", "were",
}


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _overlap(expected: str, actual: str) -> float:
    """Token-recall of an expected claim against a candidate string."""
    wanted = _tokens(expected) - _STOPWORDS
    if not wanted:
        return 0.0
    return len(wanted & _tokens(actual)) / len(wanted)


@dataclass
class EvaluationMetrics:
    """One paper's scores. ``None`` means 'not measurable for this paper'."""

    document_id: str
    paper: str

    claim_coverage: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    numeric_accuracy: float | None = None
    unsupported_claim_rate: float | None = None
    verification_accuracy: float | None = None

    latency_seconds: float | None = None
    estimated_cost_usd: float | None = None
    total_tokens: int = 0

    findings: int = 0
    evidence_items: int = 0
    verified_claims: int = 0
    retries: int = 0
    agents_run: list[str] = field(default_factory=list)
    agents_skipped: list[str] = field(default_factory=list)
    status: str = "unknown"
    errors: list[str] = field(default_factory=list)

    #: Per-metric detail, for inspecting *why* a score came out as it did.
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- individual metrics ----------------------------------------------------


def claim_coverage(
    record: AnalysisRecord, expected_claims: list[str], *, threshold: float = 0.5
) -> tuple[float | None, dict]:
    """Share of expected claims that appear somewhere in the report.

    Matching is lexical recall against the report text and the findings, which
    is deliberately generous: the metric asks whether the system *found* the
    fact, not whether it phrased it a particular way.
    """
    if not expected_claims:
        return None, {}

    haystacks = [record.report_markdown or ""]
    haystacks.extend(f"{f.claim} {f.explanation}" for f in record.findings)

    found: list[str] = []
    missing: list[str] = []
    for claim in expected_claims:
        best = max((_overlap(claim, h) for h in haystacks), default=0.0)
        (found if best >= threshold else missing).append(claim)

    return round(len(found) / len(expected_claims), 4), {
        "found": found,
        "missing": missing,
        "threshold": threshold,
    }


def citation_precision(record: AnalysisRecord, index: PaperIndex) -> tuple[float | None, dict]:
    """Share of citations whose quote is genuinely at the cited location.

    Every citation is re-checked against the index from scratch rather than
    trusting the validation that ran during the analysis.
    """
    evidence = _all_evidence(record)
    if not evidence:
        return None, {"checked": 0}

    good = 0
    failures: list[dict] = []
    for item in evidence:
        if not index.has_page(item.page):
            failures.append({"page": item.page, "reason": "page outside the document"})
            continue
        chunk = index.get(item.chunk_id) if item.chunk_id else None
        source = chunk.text if chunk else index.page_text(item.page)
        match = match_quote(item.quote, source)
        if match.found:
            good += 1
        else:
            failures.append(
                {
                    "page": item.page,
                    "similarity": match.similarity,
                    "quote": item.quote[:120],
                    "reason": "quote not found at the cited location",
                }
            )

    return round(good / len(evidence), 4), {
        "checked": len(evidence),
        "supported": good,
        "failures": failures[:20],
    }


def citation_recall(record: AnalysisRecord) -> tuple[float | None, dict]:
    """Share of *factual* claims carrying at least one verified citation.

    Claims the system honestly labelled ``inferred`` are excluded: an inference
    is not asserted as a fact about the paper, so demanding a citation for it
    would penalise the system for being candid.
    """
    factual = [f for f in record.findings if f.status is not FindingStatus.INFERRED]
    if not factual:
        return None, {"factual_claims": 0}

    cited = [f for f in factual if f.has_evidence]
    return round(len(cited) / len(factual), 4), {
        "factual_claims": len(factual),
        "cited": len(cited),
        "uncited": [f.claim[:140] for f in factual if not f.has_evidence][:20],
    }


def numeric_accuracy_score(record: AnalysisRecord, index: PaperIndex) -> tuple[float | None, dict]:
    """Share of numbers in claims that really occur somewhere in the paper.

    Checked against the whole document rather than only the cited chunk: the
    question here is "was this number invented?", which is strict about
    fabrication and forgiving about mis-citation. The pipeline's own per-claim
    check is the stricter one.
    """
    document_numbers = extract_numbers(index.full_text(), skip_trivial=False)
    total = 0
    correct = 0
    wrong: list[dict] = []

    for finding in record.findings:
        for number in extract_numbers(finding.claim):
            total += 1
            if any(numbers_equal(number, candidate) for candidate in document_numbers):
                correct += 1
            else:
                wrong.append(
                    {
                        "value": number.raw + number.suffix,
                        "claim": finding.claim[:140],
                        "agent": finding.agent,
                        "status": finding.status.value,
                    }
                )

    if total == 0:
        return None, {"numbers_checked": 0}
    return round(correct / total, 4), {
        "numbers_checked": total,
        "correct": correct,
        "not_in_paper": wrong[:20],
    }


def unsupported_claim_rate(record: AnalysisRecord) -> tuple[float | None, dict]:
    """Share of findings that failed to ground.

    A *low* rate is not automatically good: a system that never admits doubt
    scores zero here while being less trustworthy. Read it beside
    ``verification_accuracy``, which checks whether the admissions were right.
    """
    if not record.findings:
        return None, {"findings": 0}
    weak = [f for f in record.findings if f.status in _WEAK]
    return round(len(weak) / len(record.findings), 4), {
        "findings": len(record.findings),
        "unsupported": len(weak),
        "examples": [f.claim[:140] for f in weak][:10],
    }


_CLAIMS_SUPPORT = (FindingStatus.SUPPORTED, FindingStatus.PARTIALLY_SUPPORTED)


def _verification_oracle(
    claim: str, evidence: list[Evidence], index: PaperIndex, document_numbers: list
) -> tuple[bool, str]:
    """Whether the paper deterministically grounds ``claim``, and why.

    Grounded means some quote, from any source, is really in the paper and the
    quotes found there carry every number the claim states. A number the paper
    never prints anywhere makes the claim ungrounded outright, whatever is
    quoted beside it.
    """
    numbers = extract_numbers(claim)
    invented = [n for n in numbers if not any(numbers_equal(n, d) for d in document_numbers)]
    if invented:
        return False, "claim states a number the paper does not contain: " + ", ".join(
            n.raw + n.suffix for n in invented
        )
    located = [
        e for e in evidence
        if index.has_page(e.page) and match_quote(e.quote, index.page_text(e.page)).found
    ]
    if not located:
        return False, "no quote for the claim is in the paper"
    quoted = [q for e in located for q in extract_numbers(e.quote, skip_trivial=False)]
    unquoted = [n for n in numbers if not any(numbers_equal(n, q) for q in quoted)]
    if unquoted:
        return False, "quotes found, but none carries " + ", ".join(
            n.raw + n.suffix for n in unquoted
        )
    return True, "a quote is in the paper and carries the claim's numbers"


def verification_accuracy(record: AnalysisRecord, index: PaperIndex) -> tuple[float | None, dict]:
    """Balanced accuracy of the verifier's verdicts against a deterministic oracle.

    The oracle (``_verification_oracle``) labels each verified claim grounded or
    not from the paper itself, using every quote offered for it: the verifier's
    and, independently, the originating finding's. A verifier that cites nothing
    therefore cannot make a well-grounded claim look ungrounded.

    The score is the mean of the verifier's accuracy on grounded claims and on
    ungrounded ones, so a verifier that always answers the same way scores at
    most 0.5 when both kinds are present, rather than the share of the majority
    class. With only one kind present it is plain accuracy on that kind, and
    ``classes`` in the detail says so.

    Quote presence is evidence, not proof of meaning: a real quote cited for a
    claim it does not support still counts as grounding. Contradiction is not
    judged, so ``contradicted`` scores as a non-support verdict.
    """
    results = record.verification_results
    if not results:
        return None, {"claims": 0}

    findings = {f.finding_id: f for f in record.findings if f.finding_id}
    document_numbers = extract_numbers(index.full_text(), skip_trivial=False)
    tally = {True: [0, 0], False: [0, 0]}  # oracle label -> [agreements, total]
    disagreements: list[dict] = []
    for result in results:
        finding = findings.get(result.finding_id) if result.finding_id else None
        evidence = list(result.supporting_evidence) + (finding.evidence if finding else [])
        grounded, reason = _verification_oracle(result.claim, evidence, index, document_numbers)
        agrees = (result.status in _CLAIMS_SUPPORT) == grounded
        tally[grounded][1] += 1
        tally[grounded][0] += agrees
        if not agrees:
            disagreements.append(
                {
                    "claim": result.claim[:140],
                    "verdict": result.status.value,
                    "oracle": "grounded" if grounded else "ungrounded",
                    "reason": reason,
                }
            )

    recalls = [ok / total for ok, total in tally.values() if total]
    agreements = tally[True][0] + tally[False][0]
    return round(sum(recalls) / len(recalls), 4), {
        "claims": len(results),
        "agreements": agreements,
        "raw_accuracy": round(agreements / len(results), 4),
        "grounded": {"claims": tally[True][1], "agreed": tally[True][0]},
        "ungrounded": {"claims": tally[False][1], "agreed": tally[False][0]},
        "classes": sum(1 for _, total in tally.values() if total),
        "disagreements": disagreements[:20],
    }


# --- aggregation -----------------------------------------------------------


def _all_evidence(record: AnalysisRecord) -> list[Evidence]:
    """Every distinct piece of evidence the run produced."""
    evidence: list[Evidence] = []
    seen: set[tuple] = set()

    def add(item: Evidence) -> None:
        key = (item.page, item.chunk_id, normalize_for_match(item.quote)[:80])
        if key not in seen:
            seen.add(key)
            evidence.append(item)

    for finding in record.findings:
        for item in finding.evidence:
            add(item)
    for result in record.verification_results:
        for item in result.supporting_evidence:
            add(item)
    return evidence


def evaluate_record(
    record: AnalysisRecord,
    index: PaperIndex,
    *,
    expected_claims: list[str] | None = None,
    paper_name: str = "",
) -> EvaluationMetrics:
    """Score one completed analysis."""
    coverage, coverage_detail = claim_coverage(record, expected_claims or [])
    precision, precision_detail = citation_precision(record, index)
    recall, recall_detail = citation_recall(record)
    numeric, numeric_detail = numeric_accuracy_score(record, index)
    unsupported, unsupported_detail = unsupported_claim_rate(record)
    verification, verification_detail = verification_accuracy(record, index)

    costs = [u.estimated_cost_usd for u in record.usage if u.estimated_cost_usd is not None]
    stats = record.stats()

    return EvaluationMetrics(
        document_id=record.document_id,
        paper=paper_name or (index.metadata.filename or record.document_id),
        claim_coverage=coverage,
        citation_precision=precision,
        citation_recall=recall,
        numeric_accuracy=numeric,
        unsupported_claim_rate=unsupported,
        verification_accuracy=verification,
        latency_seconds=record.duration_seconds,
        estimated_cost_usd=round(sum(costs), 6) if costs else None,
        total_tokens=stats["total_tokens"],
        findings=len(record.findings),
        evidence_items=len(_all_evidence(record)),
        verified_claims=len(record.verification_results),
        retries=record.retry_count,
        agents_run=stats["agents_run"],
        agents_skipped=stats["agents_skipped"],
        status=record.status.value,
        errors=record.errors,
        detail={
            "claim_coverage": coverage_detail,
            "citation_precision": precision_detail,
            "citation_recall": recall_detail,
            "numeric_accuracy": numeric_detail,
            "unsupported_claim_rate": unsupported_detail,
            "verification_accuracy": verification_detail,
        },
    )


def aggregate(results: list[EvaluationMetrics]) -> dict[str, Any]:
    """Mean of each metric across papers, ignoring papers where it was ``None``."""
    numeric_fields = (
        "claim_coverage",
        "citation_precision",
        "citation_recall",
        "numeric_accuracy",
        "unsupported_claim_rate",
        "verification_accuracy",
        "latency_seconds",
    )
    summary: dict[str, Any] = {
        "papers": len(results),
        "completed": sum(1 for r in results if r.status == "completed"),
        "failed": sum(1 for r in results if r.status != "completed"),
        "total_tokens": sum(r.total_tokens for r in results),
    }

    for name in numeric_fields:
        values = [getattr(r, name) for r in results if getattr(r, name) is not None]
        summary[f"mean_{name}"] = round(sum(values) / len(values), 4) if values else None
        summary[f"measured_{name}"] = len(values)

    costs = [r.estimated_cost_usd for r in results if r.estimated_cost_usd is not None]
    summary["total_estimated_cost_usd"] = round(sum(costs), 6) if costs else None
    return summary
