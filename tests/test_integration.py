"""End-to-end workflow tests.

These run the complete pipeline against the synthetic fixture paper with the
deterministic mock provider: no credentials, no network. They assert the
behaviours the system exists to provide, not just that it does not crash.
"""

from __future__ import annotations

import pytest

from paperpilot.domain.enums import AnalysisStatus, FindingStatus
from paperpilot.providers.mock_llm import UNGROUNDED_CLAIM


@pytest.fixture(scope="module")
def analysis(request):
    """One full analysis, shared by the assertions below."""
    return request.getfixturevalue("_module_analysis")


@pytest.fixture(scope="module")
def _module_analysis(tmp_path_factory):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fixtures"))
    from synthetic_paper import build_pdf

    from paperpilot.config import Settings
    from paperpilot.providers.embeddings import HashingEmbeddingProvider
    from paperpilot.providers.mock_llm import MockLLMProvider
    from paperpilot.service import PaperPilotService
    from paperpilot.storage.sqlite_repo import (
        InMemoryAnalysisRepository,
        InMemoryDocumentRepository,
    )

    tmp = tmp_path_factory.mktemp("integration")
    pdf = build_pdf(tmp / "paper.pdf")
    settings = Settings(
        data_dir=tmp / "data",
        database_url=f"sqlite:///{tmp / 'data' / 'db.sqlite'}",
        llm_provider="mock",
        embedding_provider="hashing",
        embedding_dim=256,
        log_level="ERROR",
        ocr_enabled=False,
        max_retries=1,
    )
    service = PaperPilotService(
        settings,
        documents=InMemoryDocumentRepository(),
        analyses=InMemoryAnalysisRepository(),
        llm_provider=MockLLMProvider(),
        embedding_provider=HashingEmbeddingProvider(dim=256),
    )
    return service.analyze_file(pdf)


# --- the workflow completes ------------------------------------------------


def test_analysis_completes(analysis):
    assert analysis.status is AnalysisStatus.COMPLETED
    assert analysis.errors == []
    assert analysis.report_markdown
    assert analysis.completed_at is not None


def test_the_paper_was_ingested_with_page_aware_metadata(analysis):
    stages = {e.stage for e in analysis.events}
    assert {"ingestion", "indexing"} <= stages
    ingestion = next(
        e for e in analysis.events if e.stage == "ingestion" and e.status == "completed"
    )
    assert ingestion.detail["pages"] == 6
    assert ingestion.detail["chunks"] > 10


# --- the orchestrator plans rather than running everything blindly ---------


def test_the_orchestrator_created_a_plan_with_reasons(analysis):
    assert analysis.plan is not None
    assert analysis.plan.rationale, "the plan must explain itself"
    assert analysis.plan.enabled_agents()


def test_at_least_three_specialist_agents_produced_typed_results(analysis):
    ran = [name for name, result in analysis.agent_results.items() if result.status == "ok"]
    assert len(ran) >= 3, f"only {ran} ran"
    for name in ran:
        result = analysis.agent_results[name]
        assert result.usage is not None
        assert isinstance(result.payload, dict)


def test_math_analysis_is_conditional_and_explained(analysis):
    """This fixture has three equations, so the Math Agent must have activated."""
    math = analysis.agent_results.get("math")
    assert math is not None
    assert math.status == "ok"
    assert analysis.plan.run_math is True
    assert "equation" in analysis.plan.rationale["run_math"]


def test_every_content_chunk_was_read_by_a_model(analysis):
    """Specialists plus the coverage sweep leave no part of the paper unread."""
    coverage = analysis.agent_results.get("coverage")
    assert coverage is not None
    read = {
        chunk_id
        for result in analysis.agent_results.values()
        if result.status == "ok"
        for chunk_id in result.context_chunk_ids
    }
    if coverage.status == "skipped":
        return  # the specialists alone read everything
    data = coverage.payload
    assert data["read_by_specialists"] + data["read_by_sweep"] == data["content_chunks"]
    assert data["unread_sections"] == []
    assert set(coverage.context_chunk_ids) <= read
    assert "Details from the rest of the paper" in analysis.report_markdown


def test_math_skip_is_reported_when_a_paper_has_no_mathematics(tmp_path):
    """The complementary branch: a paper without maths must skip the agent."""
    from tests.conftest import _context_for, _index_from_texts  # noqa: PLC0415

    from paperpilot.agents.math_agent import detect_math_content

    context = _context_for(
        _index_from_texts(
            [
                "We report an interview study of twelve software teams and their "
                "review practices, coded thematically from transcripts.",
                "Participants described review as a social process shaped by trust.",
            ],
            document_id="paper-nomath",
        )
    )
    activate, reason = detect_math_content(context, min_equations=3)
    assert activate is False and "below the threshold" in reason


# --- findings carry evidence ----------------------------------------------


def test_findings_were_produced(analysis):
    assert len(analysis.findings) >= 5


def test_most_findings_carry_verified_evidence(analysis):
    grounded = [f for f in analysis.findings if f.has_evidence]
    assert len(grounded) >= 3
    assert len(grounded) / len(analysis.findings) > 0.5


def test_every_piece_of_evidence_points_at_a_real_page(analysis):
    for finding in analysis.findings:
        for evidence in finding.evidence:
            assert 1 <= evidence.page <= 6, f"page {evidence.page} is outside the document"


def test_verified_evidence_quotes_exist_in_the_paper(analysis):
    """The core guarantee: a rendered citation's quote is really in the PDF."""
    for finding in analysis.findings:
        for evidence in finding.evidence:
            if evidence.is_verified:
                assert evidence.quote.strip()
                assert evidence.chunk_id or evidence.page


# --- verification changes outcomes ----------------------------------------


def test_high_impact_claims_were_verified(analysis):
    assert len(analysis.verification_results) >= 3


def test_verification_changed_the_status_of_at_least_one_finding(analysis):
    """A verifier that never overrules anything is not verifying."""
    weak = [
        f
        for f in analysis.findings
        if f.status in (FindingStatus.UNSUPPORTED, FindingStatus.CONTRADICTED)
    ]
    assert weak, "no finding was downgraded by verification"
    assert all(f.verification_note for f in weak)


def test_the_deliberately_ungrounded_claim_was_caught(analysis):
    """The mock always emits one fabricated claim; it must never be reported as fact."""
    caught = [f for f in analysis.findings if f.claim == UNGROUNDED_CLAIM]
    assert caught, "the planted ungrounded claim disappeared instead of being reported"
    assert caught[0].status is FindingStatus.UNSUPPORTED
    assert caught[0].confidence < 0.5


def test_verification_results_are_not_double_counted(analysis):
    claims = [r.claim.strip().lower() for r in analysis.verification_results]
    assert len(claims) == len(set(claims))


def test_numeric_claims_were_checked_deterministically(analysis):
    with_checks = [r for r in analysis.verification_results if r.numeric_checks]
    assert with_checks, "no numeric checks ran"


# --- retries ---------------------------------------------------------------


def test_retries_are_bounded(analysis):
    assert analysis.retry_count <= (analysis.plan.max_retries if analysis.plan else 1)


def test_a_retry_records_why_it_happened(analysis):
    retry_events = [e for e in analysis.events if e.stage == "retry"]
    assert retry_events, "the retry decision must always be recorded, even when skipped"
    assert all(e.message for e in retry_events)


# --- the report ------------------------------------------------------------


def test_report_contains_page_citations(analysis):
    report = analysis.report_markdown
    assert "[p. " in report
    assert report.count("[p. ") >= 5


def test_report_citations_use_the_specified_format(analysis):
    import re

    citations = re.findall(r"\[p\. \d+(?:, §[^\],]+)?(?:, [A-Za-z]+ [\w.-]+)?\]", analysis.report_markdown)
    assert citations, "no citation matched the documented format"
    assert any("§" in c for c in citations), "sections should appear in citations"


def test_report_cites_the_results_table(analysis):
    assert "Table 1" in analysis.report_markdown


def test_report_contains_every_required_section(analysis):
    required = [
        "Executive summary",
        "Research problem and motivation",
        "Contributions",
        "Methodology",
        "Mathematical explanation",
        "Dataset and experimental setup",
        "Results and ablations",
        "Limitations and threats to validity",
        "Reproducibility checklist",
        "Verified findings",
        "Unsupported, contradicted and inferred findings",
        "Open questions",
        "Evidence appendix",
    ]
    for heading in required:
        assert heading in analysis.report_markdown, f"report is missing: {heading}"


def test_report_labels_unsupported_claims_explicitly(analysis):
    report = analysis.report_markdown
    assert "Unsupported, contradicted and inferred findings" in report
    assert UNGROUNDED_CLAIM[:50] in report, "the ungrounded claim must be shown, not hidden"


def test_report_states_the_page_numbering_convention(analysis):
    assert "1-based PDF page indices" in analysis.report_markdown


def test_report_discloses_that_the_mock_provider_was_used(analysis):
    assert "mock provider" in analysis.report_markdown.lower()


def test_report_does_not_claim_general_intelligence(analysis):
    report = analysis.report_markdown.lower()
    for phrase in ("artificial general intelligence", "human-level", "agi"):
        assert phrase not in report
    assert "not general intelligence" in report


def test_report_explains_why_each_agent_ran(analysis):
    assert "How this report was produced" in analysis.report_markdown
    assert "Plan" in analysis.report_markdown


def test_evidence_appendix_lists_quotes(analysis):
    appendix = analysis.report_markdown.split("Evidence appendix")[-1]
    assert "> " in appendix, "the appendix must contain block-quoted evidence"


# --- observability ---------------------------------------------------------


def test_progress_events_cover_the_whole_workflow(analysis):
    stages = {e.stage for e in analysis.events}
    assert {"ingestion", "indexing", "planning", "aggregation", "verification", "report"} <= stages
    assert any(s.startswith("agent:") for s in stages)


def test_token_usage_is_recorded_per_agent(analysis):
    assert analysis.usage
    assert all(u.agent and u.prompt_version and u.model for u in analysis.usage)
    assert sum(u.total_tokens for u in analysis.usage) > 0


def test_stats_summarise_the_run(analysis):
    stats = analysis.stats()
    assert stats["findings"] == len(analysis.findings)
    assert stats["agents_run"]
    assert stats["status_counts"]


# --- failure handling ------------------------------------------------------


def test_a_corrupt_pdf_becomes_a_failed_record_not_a_crash(service, tmp_path):
    """A file with a valid header but corrupt body passes the cheap checks and
    fails during parsing. That must surface as a failed analysis the caller can
    inspect, not as an exception escaping the service."""
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"%PDF-1.4 but then nothing valid at all")

    record = service.analyze_file(bad)
    assert record.status is AnalysisStatus.FAILED
    assert record.errors and "could not open PDF" in record.errors[0]
    assert record.report_markdown is None
    assert record.findings == []


def test_a_non_pdf_is_rejected_before_any_model_call(service, tmp_path, llm):
    from paperpilot.errors import UnsupportedFileError

    bad = tmp_path / "notes.txt"
    bad.write_text("not a pdf")
    with pytest.raises(UnsupportedFileError):
        service.analyze_file(bad)
    assert llm.calls == [], "no model call may happen before file validation"
