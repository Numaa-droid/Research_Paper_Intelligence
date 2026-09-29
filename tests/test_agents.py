"""Agent contracts, conditional activation and claim extraction."""

from __future__ import annotations

import pytest

from paperpilot.agents.base import dedupe_results, format_context, format_metadata
from paperpilot.agents.experiments import ExperimentAgent
from paperpilot.agents.math_agent import MathAgent, detect_math_content
from paperpilot.agents.methodology import MethodologyAgent
from paperpilot.agents.summary import SummaryAgent
from paperpilot.agents.verification import (
    VerificationAgent,
    apply_verification,
    extract_high_impact_claims,
)
from paperpilot.domain.enums import FindingStatus
from paperpilot.domain.models import ClaimCandidate, Evidence, Finding
from paperpilot.errors import ProviderError
from paperpilot.providers.llm import LLMProvider, LLMResponse
from paperpilot.providers.mock_llm import UNGROUNDED_CLAIM


class MarkdownProvider(LLMProvider):
    """Answers every request with fixed Markdown."""

    name = "markdown"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, request):
        return LLMResponse(text=self.text, model="x", provider=self.name)


class FailingProvider(LLMProvider):
    name = "failing"

    def generate(self, request):
        raise ProviderError("provider is down")


class MalformedProvider(MarkdownProvider):
    """Ignores the requested layout entirely."""

    def __init__(self) -> None:
        super().__init__('{"unexpected": "shape"}')


class HallucinatingProvider(MarkdownProvider):
    """Returns a confident finding whose quote is not in the paper."""

    def __init__(self) -> None:
        super().__init__(
            "## Findings\n\n"
            "### Finding 1\n"
            "- Claim: The model was trained on 400 billion tokens of web text.\n"
            "- Status: supported\n"
            "- Confidence: 0.99\n"
            "- Impact: 0.9\n"
            "- Explanation: fabricated\n\n"
            "> [S1] We trained on 400 billion tokens of web text.\n"
        )


# --- shared agent behaviour ------------------------------------------------


@pytest.mark.parametrize("agent_cls", [SummaryAgent, MethodologyAgent, ExperimentAgent])
def test_agents_return_typed_results_with_evidence(agent_cls, llm, agent_context):
    result = agent_cls(llm).run(agent_context)
    assert result.status == "ok"
    assert result.agent == agent_cls.name
    assert result.findings, "an agent must produce findings on a normal paper"
    assert any(f.has_evidence for f in result.findings)
    assert result.usage is not None
    assert result.usage.prompt_version
    assert result.usage.model


@pytest.mark.parametrize("agent_cls", [SummaryAgent, MethodologyAgent, ExperimentAgent])
def test_agent_evidence_pages_are_inside_the_document(agent_cls, llm, agent_context):
    result = agent_cls(llm).run(agent_context)
    for finding in result.findings:
        for evidence in finding.evidence:
            assert agent_context.index.has_page(evidence.page)


def test_agents_do_not_read_the_whole_paper(llm, agent_context):
    """Targeted retrieval is the design: context must stay bounded."""
    agent = SummaryAgent(llm)
    results = agent.retrieve(agent_context)
    assert len(results) <= agent.context_limit
    assert len(results) < len(agent_context.index.chunks)


def test_provider_failure_is_reported_not_raised(agent_context):
    result = SummaryAgent(FailingProvider()).run(agent_context)
    assert result.status == "failed"
    assert "provider is down" in (result.error or "")
    assert result.findings == []


def test_malformed_provider_output_yields_no_findings(agent_context):
    result = SummaryAgent(MalformedProvider()).run(agent_context)
    assert result.status == "ok"
    assert result.findings == []


def test_hallucinated_evidence_is_caught_by_validation(agent_context):
    """An agent's own confidence must not survive a failed quote check."""
    result = SummaryAgent(HallucinatingProvider()).run(agent_context)
    finding = result.findings[0]
    assert finding.status is FindingStatus.UNSUPPORTED
    assert finding.confidence < 0.3


def test_retry_feedback_reaches_the_prompt(llm, agent_context):
    agent_context.retry_feedback = "Your quote about routing could not be found."
    SummaryAgent(llm).run(agent_context)
    assert any("could not be found" in call.user for call in llm.calls)


def test_context_block_exposes_citable_metadata(retriever, index):
    from paperpilot.retrieval.retriever import RetrievalQuery

    results = retriever.search(RetrievalQuery(text="main results", top_k=3))
    block = format_context(results)
    assert "source: S1" in block and "page:" in block and "section:" in block


def test_context_block_is_truncated_to_the_budget(retriever):
    from paperpilot.retrieval.retriever import RetrievalQuery

    results = retriever.search(RetrievalQuery(text="routing", top_k=8))
    assert len(format_context(results, max_chars=200)) <= 400


def test_metadata_block_states_the_page_numbering_convention(ingested):
    assert "1-based PDF indices" in format_metadata(ingested.metadata)


def test_metadata_block_warns_about_ocr(ingested):
    ingested.metadata.ocr_page_count = 3
    assert "OCR" in format_metadata(ingested.metadata)


def test_dedupe_interleaves_groups_without_duplicates(retriever):
    from paperpilot.retrieval.retriever import RetrievalQuery

    groups = [
        retriever.search(RetrievalQuery(text="routing", top_k=4)),
        retriever.search(RetrievalQuery(text="results", top_k=4)),
    ]
    merged = dedupe_results(groups, limit=6)
    assert len(merged) == len({r.chunk_id for r in merged})
    assert len(merged) <= 6


# --- conditional math activation -------------------------------------------


def test_math_agent_activates_on_a_mathematical_paper(agent_context):
    activate, reason = detect_math_content(agent_context, min_equations=3)
    assert activate is True
    assert "3 equations" in reason


def test_math_agent_is_skipped_on_a_paper_without_mathematics(non_math_context):
    activate, reason = detect_math_content(non_math_context, min_equations=3)
    assert activate is False
    assert "below the threshold" in reason


def test_math_agent_skip_is_recorded_with_a_reason(llm, non_math_context):
    result = MathAgent(llm, min_equations=3).run(non_math_context)
    assert result.status == "skipped"
    assert result.skip_reason
    assert result.findings == []


def test_math_agent_activates_on_discussion_when_equation_extraction_failed(
    math_prose_context,
):
    """A scanned paper can lose its equations; the prose still says it has maths."""
    activate, reason = detect_math_content(math_prose_context, min_equations=3)
    assert activate is True
    assert "mathematical discussion" in reason


def test_math_agent_runs_when_activated(llm, agent_context):
    result = MathAgent(llm, min_equations=3).run(agent_context)
    assert result.status == "ok"
    assert len(result.payload["equations"]) == 3


def test_math_agent_preserves_equations_verbatim(llm, agent_context):
    result = MathAgent(llm, min_equations=3).run(agent_context)
    sources = [e["source_text"] for e in result.payload["equations"]]
    assert any("L_read" in s for s in sources)
    assert all(e["representation"] in ("latex", "extracted_text", "unavailable") for e in result.payload["equations"])


def test_math_agent_marks_unreconstructable_equations_as_unavailable(agent_context):
    provider = MarkdownProvider(
        "## Equations\n\n"
        "### Equation 9\n"
        "- Page: 3\n"
        "- Representation: latex\n"
        "- Purpose: fabricated\n\n"
        "```latex\nE = mc^2 + invented_term_not_in_paper\n```\n\n"
        "## Findings\nNot stated.\n"
    )
    result = MathAgent(provider, min_equations=1).run(agent_context)
    invented = next(e for e in result.payload["equations"] if e["purpose"] == "fabricated")
    assert invented["representation"] == "unavailable"
    assert result.open_questions


def test_math_agent_discards_equations_on_impossible_pages(agent_context):
    provider = MarkdownProvider(
        "## Equations\n\n### Equation eq\n- Page: 999\n\n```text\nx = 1\n```\n"
    )
    result = MathAgent(provider, min_equations=1).run(agent_context)
    assert all(e["page"] != 999 for e in result.payload["equations"])
    assert any("outside this document" in q for q in result.open_questions)


# --- experiment numeric discipline -----------------------------------------


def test_experiment_agent_reports_numbers_with_evidence(llm, agent_context):
    result = ExperimentAgent(llm).run(agent_context)
    reported = result.payload["reported_results"]
    assert reported, "the results table should yield reported values"
    assert all(item["evidence"] for item in reported)


def test_experiment_agent_flags_values_absent_from_their_evidence(agent_context):
    class BadNumberProvider(LLMProvider):
        name = "bad_numbers"

        def generate(self, request):
            quote = " ".join(request.evidence[0]["text"][:100].split()).replace("|", "\\|")
            return LLMResponse(
                text=(
                    "## Reported Results\n\n"
                    "| System | Metric | Value | Source | Quote |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    f"| SRT-Large | F1 | 99.9 | [S1] | {quote} |\n"
                ),
                model="x",
                provider=self.name,
            )

    result = ExperimentAgent(BadNumberProvider()).run(agent_context)
    assert result.payload["unverified_numbers"]
    assert any("99.9" in u for u in result.payload["unverified_numbers"])


# --- claim extraction ------------------------------------------------------


def test_numeric_claims_are_prioritised_for_verification():
    findings = [
        Finding(claim="The paper is about transformers.", agent="summary", impact=0.3),
        Finding(claim="SRT-Large reaches 81.4 F1.", agent="experiments", impact=0.3),
    ]
    candidates = extract_high_impact_claims(findings, threshold=0.5, limit=1)
    assert candidates[0].claim.startswith("SRT-Large")
    assert "numeric" in candidates[0].reason


def test_claims_without_evidence_are_prioritised():
    findings = [
        Finding(
            claim="A well grounded statement about method design.",
            agent="methodology",
            impact=0.5,
            status=FindingStatus.SUPPORTED,
            evidence=[
                Evidence(page=1, quote="x", confidence=0.9, citation_quality="exact")
            ],
        ),
        Finding(claim="An unsupported statement about method design.", agent="methodology", impact=0.5),
    ]
    candidates = extract_high_impact_claims(findings, threshold=0.9, limit=5)
    top = candidates[0]
    assert "no verified evidence" in top.reason or "unknown" in top.reason


def test_duplicate_claims_are_verified_once():
    findings = [
        Finding(claim="Same claim", agent="summary"),
        Finding(claim="same claim", agent="experiments"),
    ]
    assert len(extract_high_impact_claims(findings)) == 1


def test_at_least_three_claims_are_always_verified():
    findings = [Finding(claim=f"Dull statement {i}", agent="summary", impact=0.0) for i in range(6)]
    assert len(extract_high_impact_claims(findings, threshold=0.99)) >= 3


def test_claim_count_is_capped():
    findings = [
        Finding(claim=f"Claim {i} reaching 8{i}.4 F1", agent="experiments", impact=0.9)
        for i in range(40)
    ]
    assert len(extract_high_impact_claims(findings, limit=14)) <= 14


# --- verification ----------------------------------------------------------


def test_verifier_rejects_a_claim_the_paper_does_not_support(llm, agent_context):
    candidate = ClaimCandidate(
        claim=UNGROUNDED_CLAIM, source_agent="summary", impact=0.95
    )
    result = VerificationAgent(llm).verify(candidate, agent_context)
    assert result.status is FindingStatus.UNSUPPORTED
    assert result.is_weak
    assert result.confidence < 0.5


def test_verifier_supports_a_claim_that_is_in_the_paper(llm, agent_context):
    chunk = next(c for c in agent_context.index.chunks if "load-balancing" in c.text)
    candidate = ClaimCandidate(
        claim=chunk.text[:180], source_agent="methodology", impact=0.7
    )
    result = VerificationAgent(llm).verify(candidate, agent_context)
    assert result.status in (FindingStatus.SUPPORTED, FindingStatus.PARTIALLY_SUPPORTED)
    assert result.supporting_evidence


def test_verifier_records_numeric_checks(llm, agent_context):
    candidate = ClaimCandidate(
        claim="SRT-Large reaches 99.9 F1 on NarrativeQA", source_agent="experiments", impact=0.9
    )
    result = VerificationAgent(llm).verify(candidate, agent_context)
    assert result.numeric_checks
    assert any(not c.found_in_evidence for c in result.numeric_checks)


def test_deterministic_checks_override_an_over_confident_verifier(agent_context):
    """A 'supported' verdict with no verifiable quote must be downgraded."""

    provider = MarkdownProvider(
        "## Verdict\n- Status: supported\n- Confidence: 0.99\n\n"
        "## Explanation\nlooks right to me\n\n"
        "## Supporting Evidence\nNone.\n"
    )
    candidate = ClaimCandidate(claim="Some claim about routing", source_agent="summary", impact=0.8)
    result = VerificationAgent(provider).verify(candidate, agent_context)
    assert result.status is FindingStatus.UNSUPPORTED
    assert any("no verifiable quote" in n for n in result.deterministic_notes)


def test_verifier_failure_yields_unknown_not_a_guess(agent_context):
    candidate = ClaimCandidate(claim="Some claim", source_agent="summary", impact=0.8)
    result = VerificationAgent(FailingProvider()).verify(candidate, agent_context)
    assert result.status is FindingStatus.UNKNOWN


def test_verification_replaces_the_authoring_agents_self_assessment():
    findings = [
        Finding(claim="Bold claim", agent="summary", status=FindingStatus.SUPPORTED, confidence=0.9)
    ]
    from paperpilot.domain.models import VerificationResult

    results = [
        VerificationResult(
            claim="Bold claim",
            source_agent="summary",
            status=FindingStatus.UNSUPPORTED,
            confidence=0.1,
            explanation="no evidence found",
        )
    ]
    apply_verification(findings, results)
    assert findings[0].status is FindingStatus.UNSUPPORTED
    assert findings[0].confidence == 0.1
    assert "no evidence found" in findings[0].verification_note
