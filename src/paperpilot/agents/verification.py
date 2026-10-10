"""Verification Agent.

The verifier re-retrieves evidence for a claim *independently* of whatever the
original agent cited. That is the point: if a claim only survives because the
agent that wrote it also chose its evidence, it has not been checked.

Two layers decide the verdict:

* a deterministic layer (page validity, quote presence, numeric equality) which
  can *cap* the verdict but never inflate it, and
* the model, which judges semantic support.

Where they disagree, the deterministic layer wins. A number that is not in the
paper is not in the paper, however confident the model sounds.
"""

from __future__ import annotations

import time

from ..domain.enums import CitationQuality, FindingStatus
from ..domain.models import (
    AgentUsage,
    ClaimCandidate,
    Evidence,
    Finding,
    NumericCheck,
    VerificationResult,
)
from ..errors import ProviderError
from ..observability import info, warning
from ..providers.llm import LLMProvider, LLMRequest
from ..response import parse_markdown
from ..response.reader import DocumentReader, clean_text, parse_score, section_blocks
from ..response.schema import ListBlock, MathBlock, ParagraphBlock
from ..retrieval.retriever import RetrievalQuery, RetrievalResult
from ..tracing import span
from ..validation.evidence import validate_evidence
from ..validation.numeric import check_claim_numbers, extract_numbers, numeric_accuracy
from .base import AgentContext, evidence_payload, format_context, format_metadata
from .grounding import Grounder, SourceMap, parse_status, resolve_source_labels
from .prompts.registry import VERIFICATION

#: Claims at or above this impact are always verified.
HIGH_IMPACT_THRESHOLD = 0.6
#: Never verify more than this many claims in one run (cost ceiling).
MAX_CLAIMS = 14


def _impact_score(finding: Finding) -> float:
    """How much damage this claim does if it is wrong.

    Numbers, superlatives and comparative results are what readers act on, so
    they are prioritised over background description.
    """
    score = finding.impact
    claim = finding.claim.lower()

    if extract_numbers(finding.claim):
        score += 0.25
    if any(word in claim for word in (
        "outperform", "improve", "better than", "state of the art", "state-of-the-art",
        "best", "highest", "lowest", "reduces", "increases", "achieves", "beats",
    )):
        score += 0.2
    if any(word in claim for word in ("all ", "every", "always", "never", "any ")):
        score += 0.1  # universal quantifiers are easy to overstate
    if finding.status in (FindingStatus.UNSUPPORTED, FindingStatus.UNKNOWN):
        score += 0.15
    if finding.category in ("experiments", "results"):
        score += 0.1
    if not finding.has_evidence:
        score += 0.1
    return round(min(score, 1.0), 4)


def extract_high_impact_claims(
    findings: list[Finding], *, threshold: float = HIGH_IMPACT_THRESHOLD, limit: int = MAX_CLAIMS
) -> list[ClaimCandidate]:
    """Select the claims worth spending verification calls on."""
    scored: list[tuple[float, Finding]] = []
    seen: set[str] = set()

    for finding in findings:
        key = finding.claim.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        scored.append((_impact_score(finding), finding))

    scored.sort(key=lambda item: -item[0])
    candidates: list[ClaimCandidate] = []

    for score, finding in scored:
        if score < threshold and len(candidates) >= 3:
            break  # always verify at least three claims, however dull
        reasons = []
        if extract_numbers(finding.claim):
            reasons.append("contains numeric values")
        if not finding.has_evidence:
            reasons.append("no verified evidence attached")
        if finding.status in (FindingStatus.UNSUPPORTED, FindingStatus.UNKNOWN):
            reasons.append(f"status is {finding.status.value}")
        if not reasons:
            reasons.append("high-impact claim about the paper's contribution")

        candidates.append(
            ClaimCandidate(
                claim=finding.claim,
                finding_id=finding.finding_id,
                source_agent=finding.agent or "unknown",
                impact=score,
                reason="; ".join(reasons),
                existing_evidence=list(finding.evidence),
            )
        )
        if len(candidates) >= limit:
            break
    return candidates


def _format_numeric_report(checks: list[NumericCheck]) -> str:
    if not checks:
        return "The claim contains no numeric values."
    lines = []
    for check in checks:
        if check.found_in_evidence:
            lines.append(
                f"- {check.value}: FOUND in the retrieved evidence"
                + (f" (page {check.page})" if check.page else "")
            )
        else:
            lines.append(f"- {check.value}: NOT FOUND in the retrieved evidence")
    return "\n".join(lines)


class VerificationAgent:
    """Checks one claim at a time against independently retrieved evidence."""

    name = "verification"
    prompt = VERIFICATION
    context_limit = 8

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.usage: list[AgentUsage] = []

    def verify_all(
        self, candidates: list[ClaimCandidate], context: AgentContext
    ) -> list[VerificationResult]:
        results = [self.verify(candidate, context) for candidate in candidates]
        supported = sum(1 for r in results if r.status is FindingStatus.SUPPORTED)
        info(
            "verification.completed",
            claims=len(results),
            supported=supported,
            weak=sum(1 for r in results if r.is_weak),
        )
        return results

    def verify(self, candidate: ClaimCandidate, context: AgentContext) -> VerificationResult:
        with span(
            "verify_claim",
            kind="agent",
            inputs={
                "claim": candidate.claim,
                "source_agent": candidate.source_agent,
                "finding_id": candidate.finding_id,
                "attempt": context.attempt,
            },
            metadata={"prompt_version": self.prompt.version},
        ) as s:
            result = self._verify(candidate, context)
            s.set_outputs(
                status=result.status.value,
                confidence=result.confidence,
                explanation=result.explanation,
                pages=result.pages,
                citation_quality=getattr(result.citation_quality, "value", result.citation_quality),
                deterministic_notes=result.deterministic_notes,
            )
            return result

    def _verify(self, candidate: ClaimCandidate, context: AgentContext) -> VerificationResult:
        started = time.perf_counter()
        retrieved = self._retrieve(candidate, context)

        # --- deterministic layer, computed before the model is consulted ---
        evidence_pool = [r.chunk for r in retrieved]
        numeric_checks = check_claim_numbers(
            candidate.claim,
            [
                Evidence(
                    page=c.page,
                    quote=c.text,
                    source_type=c.element_type,
                    chunk_id=c.chunk_id,
                    section=c.citation_section,
                    confidence=1.0,
                    citation_quality=CitationQuality.EXACT,
                )
                for c in evidence_pool
            ],
        ) if evidence_pool else check_claim_numbers(candidate.claim, [])
        accuracy = numeric_accuracy(numeric_checks)
        deterministic_notes: list[str] = []

        if not retrieved:
            deterministic_notes.append("no evidence could be retrieved for this claim")
            return VerificationResult(
                claim=candidate.claim,
                finding_id=candidate.finding_id,
                source_agent=candidate.source_agent,
                status=FindingStatus.UNSUPPORTED,
                confidence=0.0,
                explanation=(
                    "No passage in the paper was retrieved for this claim, so it could "
                    "not be grounded."
                ),
                numeric_checks=numeric_checks,
                deterministic_notes=deterministic_notes,
                attempt=context.attempt,
            )

        # --- model layer ---------------------------------------------------
        try:
            response = self._call_model(candidate, context, retrieved, numeric_checks)
            self.usage.append(
                AgentUsage(
                    agent=self.name,
                    prompt_version=self.prompt.version,
                    model=response.model,
                    provider=response.provider,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    attempt=context.attempt,
                    estimated_cost_usd=response.estimated_cost_usd(),
                )
            )
        except ProviderError as exc:
            warning("verification.provider_failed", error=str(exc)[:200])
            return VerificationResult(
                claim=candidate.claim,
                finding_id=candidate.finding_id,
                source_agent=candidate.source_agent,
                status=FindingStatus.UNKNOWN,
                confidence=0.0,
                explanation=f"Verification could not be completed: {exc}",
                numeric_checks=numeric_checks,
                deterministic_notes=deterministic_notes,
                attempt=context.attempt,
            )

        document = parse_markdown(response.text)
        reader = DocumentReader(document)
        grounder = Grounder(sources=SourceMap(retrieved), index=context.index, agent=self.name)

        verdict = reader.fields(reader.section("verdict")) or {
            key: item for section in document.sections for key, item in reader.fields(section).items()
        }
        status = parse_status(reader.field_text(verdict, "status", "verdict"))
        score = parse_score(reader.field_text(verdict, "confidence"))
        confidence = 0.5 if score is None else score
        explanation_section = reader.section("explanation", "reasoning", "rationale")
        # Paragraphs *and* list items: a verifier that lists its evidence as bullets
        # otherwise leaves "supported by the following evidence:" pointing at nothing.
        parts: list[str] = []
        for block in section_blocks(explanation_section) if explanation_section else []:
            if isinstance(block, ParagraphBlock | MathBlock):
                parts.append(block.text)
            elif isinstance(block, ListBlock):
                parts.extend(f"- {item.text}" for item in block.items)
        explanation = clean_text(
            resolve_source_labels("\n".join(p for p in parts if p.strip()), grounder.sources)
        ) or reader.text(explanation_section)

        supporting = self._validated(
            grounder.evidence_list(reader.quotes(reader.section("supporting evidence", "support"))),
            context,
        )
        contradicting = self._validated(
            grounder.evidence_list(
                reader.quotes(reader.section("contradicting evidence", "contradiction"))
            ),
            context,
        )

        # --- reconciliation: deterministic checks cap the model's verdict ---
        if status in (FindingStatus.SUPPORTED, FindingStatus.PARTIALLY_SUPPORTED):
            if not supporting:
                deterministic_notes.append(
                    "the verifier reported support but supplied no verifiable quote"
                )
                status = FindingStatus.UNSUPPORTED
                confidence = min(confidence, 0.3)
            elif accuracy < 1.0:
                missing = [c.value for c in numeric_checks if not c.found_in_evidence]
                deterministic_notes.append(
                    "numbers absent from the retrieved evidence: " + ", ".join(missing)
                )
                if status is FindingStatus.SUPPORTED:
                    status = FindingStatus.PARTIALLY_SUPPORTED
                confidence = round(confidence * accuracy, 4)

        if contradicting and status is FindingStatus.SUPPORTED:
            deterministic_notes.append(
                "contradicting evidence was supplied alongside a 'supported' verdict; "
                "downgraded to partially_supported"
            )
            status = FindingStatus.PARTIALLY_SUPPORTED

        if not explanation:
            explanation = "The verifier returned no explanation for this verdict."

        pages = sorted({e.page for e in supporting + contradicting})
        sections = sorted({e.section for e in supporting + contradicting if e.section})
        quality = (
            max(
                (e.citation_quality for e in supporting),
                key=lambda q: [
                    CitationQuality.UNVERIFIABLE,
                    CitationQuality.APPROXIMATE,
                    CitationQuality.NORMALIZED,
                    CitationQuality.EXACT,
                ].index(q),
            )
            if supporting
            else CitationQuality.UNVERIFIABLE
        )

        result = VerificationResult(
            claim=candidate.claim,
            finding_id=candidate.finding_id,
            source_agent=candidate.source_agent,
            status=status,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            pages=pages,
            sections=sections,
            citation_quality=quality,
            confidence=confidence,
            explanation=explanation,
            numeric_checks=numeric_checks,
            deterministic_notes=deterministic_notes,
            attempt=context.attempt,
            response_markdown=response.text,
            response_document=document.model_dump(mode="json", exclude_none=True),
        )
        info(
            "verification.claim",
            status=status.value,
            confidence=confidence,
            numeric_accuracy=accuracy,
            source_agent=candidate.source_agent,
            claim=candidate.claim[:120],
        )
        return result

    # --- internals --------------------------------------------------------

    def _retrieve(self, candidate: ClaimCandidate, context: AgentContext) -> list[RetrievalResult]:
        """Retrieve evidence for the claim independently of what was cited."""
        queries = [RetrievalQuery(text=candidate.claim, top_k=self.context_limit)]

        # Numbers are strong lexical anchors; look them up exactly as well.
        for number in extract_numbers(candidate.claim)[:3]:
            queries.append(
                RetrievalQuery(
                    text=number.raw + number.suffix, top_k=3, mode="lexical"
                )
            )

        seen: set[str] = set()
        merged: list[RetrievalResult] = []
        for query in queries:
            for result in context.retriever.search(query):
                if result.chunk_id not in seen:
                    seen.add(result.chunk_id)
                    merged.append(result)
        return merged[: self.context_limit]

    def _call_model(
        self,
        candidate: ClaimCandidate,
        context: AgentContext,
        retrieved: list[RetrievalResult],
        numeric_checks: list[NumericCheck],
    ):
        user = self.prompt.render(
            metadata=format_metadata(context.metadata),
            claim=candidate.claim,
            source_agent=candidate.source_agent,
            context=format_context(retrieved, index=context.index),
            numeric_report=_format_numeric_report(numeric_checks),
        )
        request = LLMRequest(
            agent=self.name,
            task="verification",
            system=self.prompt.system,
            user=user,
            prompt_version=self.prompt.version,
            evidence=evidence_payload(retrieved),
            context={
                "claim": candidate.claim,
                "source_agent": candidate.source_agent,
                "numeric_checks": [c.model_dump(mode="json") for c in numeric_checks],
            },
        )
        return self.provider.generate(request)

    @staticmethod
    def _validated(evidence: list[Evidence], context: AgentContext) -> list[Evidence]:
        return [e for e in evidence if validate_evidence(e, context.index).ok]


def apply_verification(
    findings: list[Finding], results: list[VerificationResult]
) -> list[Finding]:
    """Fold verification verdicts back into the findings.

    The verifier's verdict replaces the originating agent's self-assessment: the
    whole point of a separate verifier is that it, not the author, has the last
    word on whether a claim is grounded.
    """
    by_claim = {r.claim.strip().lower(): r for r in results}
    for finding in findings:
        result = by_claim.get(finding.claim.strip().lower())
        if result is None:
            continue
        finding.status = result.status
        finding.confidence = result.confidence
        if result.supporting_evidence:
            finding.evidence = result.supporting_evidence
        note = result.explanation
        if result.deterministic_notes:
            note = f"{note} [checks: {'; '.join(result.deterministic_notes)}]"
        finding.verification_note = note
    return findings
