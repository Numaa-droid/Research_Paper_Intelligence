"""The evaluation harness and its metrics."""

from __future__ import annotations

import json

import pytest

from paperpilot.domain.enums import AnalysisStatus, CitationQuality, FindingStatus
from paperpilot.domain.models import AnalysisRecord, Evidence, Finding, VerificationResult
from paperpilot.evaluation import (
    EvaluationHarness,
    citation_precision,
    citation_recall,
    claim_coverage,
    evaluate_record,
    load_expectations,
    numeric_accuracy_score,
    unsupported_claim_rate,
    verification_accuracy,
)
from paperpilot.evaluation.metrics import aggregate


def _evidence(index, needle: str, *, quality=CitationQuality.EXACT) -> Evidence:
    chunk = next(c for c in index.chunks if needle in c.text)
    return Evidence(
        page=chunk.page,
        quote=chunk.text[:100],
        chunk_id=chunk.chunk_id,
        confidence=0.9,
        citation_quality=quality,
    )


def _record(findings=None, verifications=None) -> AnalysisRecord:
    return AnalysisRecord(
        analysis_id="analysis-1",
        document_id="paper-test",
        status=AnalysisStatus.COMPLETED,
        findings=findings or [],
        verification_results=verifications or [],
        report_markdown="# report",
    )


# --- claim coverage --------------------------------------------------------


def test_claim_coverage_is_none_without_ground_truth():
    score, _ = claim_coverage(_record(), [])
    assert score is None


def test_claim_coverage_finds_a_claim_present_in_the_report():
    record = _record()
    record.report_markdown = "SRT-Large reaches 81.4 F1 on the NarrativeQA benchmark."
    score, detail = claim_coverage(record, ["SRT-Large reaches 81.4 F1 on NarrativeQA"])
    assert score == 1.0
    assert not detail["missing"]


def test_claim_coverage_reports_a_missing_claim():
    record = _record()
    record.report_markdown = "This paper is about sparse routing."
    score, detail = claim_coverage(
        record, ["the router is trained with a load-balancing loss function"]
    )
    assert score == 0.0
    assert detail["missing"]


def test_claim_coverage_matches_findings_not_only_the_report():
    record = _record(
        findings=[Finding(claim="Evaluation uses NarrativeQA and QuALITY datasets")]
    )
    record.report_markdown = ""
    score, _ = claim_coverage(record, ["evaluation uses NarrativeQA and QuALITY"])
    assert score == 1.0


# --- citation precision ----------------------------------------------------


def test_citation_precision_is_one_when_every_quote_is_real(index):
    record = _record(
        findings=[Finding(claim="A claim", evidence=[_evidence(index, "81.4")])]
    )
    score, detail = citation_precision(record, index)
    assert score == 1.0
    assert detail["failures"] == []


def test_citation_precision_catches_an_invented_quote(index):
    record = _record(
        findings=[
            Finding(
                claim="A claim",
                evidence=[
                    Evidence(
                        page=1,
                        quote="This sentence is nowhere in the paper.",
                        confidence=0.9,
                        citation_quality=CitationQuality.EXACT,
                    )
                ],
            )
        ]
    )
    score, detail = citation_precision(record, index)
    assert score == 0.0
    assert detail["failures"][0]["reason"] == "quote not found at the cited location"


def test_citation_precision_catches_an_out_of_range_page(index):
    record = _record(
        findings=[
            Finding(
                claim="A claim",
                evidence=[Evidence(page=999, quote="anything", confidence=0.9)],
            )
        ]
    )
    score, detail = citation_precision(record, index)
    assert score == 0.0
    assert "outside the document" in detail["failures"][0]["reason"]


def test_citation_precision_is_none_without_any_citations(index):
    score, _ = citation_precision(_record(), index)
    assert score is None


def test_citation_precision_ignores_duplicate_citations(index):
    evidence = _evidence(index, "81.4")
    record = _record(
        findings=[
            Finding(claim="A", evidence=[evidence]),
            Finding(claim="B", evidence=[evidence.model_copy(deep=True)]),
        ]
    )
    _, detail = citation_precision(record, index)
    assert detail["checked"] == 1


# --- citation recall -------------------------------------------------------


def test_citation_recall_counts_cited_factual_claims(index):
    record = _record(
        findings=[
            Finding(claim="Cited", evidence=[_evidence(index, "81.4")]),
            Finding(claim="Uncited"),
        ]
    )
    score, detail = citation_recall(record)
    assert score == 0.5
    assert detail["uncited"] == ["Uncited"]


def test_citation_recall_excludes_honest_inferences(index):
    record = _record(
        findings=[
            Finding(claim="Cited", evidence=[_evidence(index, "81.4")]),
            Finding(claim="An inference", status=FindingStatus.INFERRED),
        ]
    )
    score, detail = citation_recall(record)
    assert score == 1.0, "a claim labelled 'inferred' must not be penalised"
    assert detail["factual_claims"] == 1


# --- numeric accuracy ------------------------------------------------------


def test_numeric_accuracy_accepts_numbers_that_are_in_the_paper(index):
    record = _record(findings=[Finding(claim="SRT-Large reaches 81.4 F1")])
    score, _ = numeric_accuracy_score(record, index)
    assert score == 1.0


def test_numeric_accuracy_catches_a_fabricated_number(index):
    record = _record(findings=[Finding(claim="It achieves 99.9% accuracy")])
    score, detail = numeric_accuracy_score(record, index)
    assert score == 0.0
    assert detail["not_in_paper"][0]["value"].startswith("99.9")


def test_numeric_accuracy_is_none_when_no_claim_has_numbers(index):
    score, _ = numeric_accuracy_score(_record(findings=[Finding(claim="No numbers")]), index)
    assert score is None


# --- unsupported rate ------------------------------------------------------


def test_unsupported_rate():
    record = _record(
        findings=[
            Finding(claim="a", status=FindingStatus.SUPPORTED),
            Finding(claim="b", status=FindingStatus.SUPPORTED),
            Finding(claim="c", status=FindingStatus.UNSUPPORTED),
            Finding(claim="d", status=FindingStatus.CONTRADICTED),
        ]
    )
    score, detail = unsupported_claim_rate(record)
    assert score == 0.5
    assert detail["unsupported"] == 2


def test_unsupported_rate_is_none_without_findings():
    assert unsupported_claim_rate(_record())[0] is None


# --- verification accuracy -------------------------------------------------


def test_verifier_agreeing_with_the_oracle_scores_one(index):
    record = _record(
        verifications=[
            VerificationResult(
                claim="A real claim",
                status=FindingStatus.SUPPORTED,
                supporting_evidence=[_evidence(index, "81.4")],
            ),
            VerificationResult(claim="A fake claim", status=FindingStatus.UNSUPPORTED),
        ]
    )
    score, _ = verification_accuracy(record, index)
    assert score == 1.0


def test_a_verifier_supporting_a_claim_with_no_real_quote_is_marked_wrong(index):
    record = _record(
        verifications=[
            VerificationResult(
                claim="Bogus",
                status=FindingStatus.SUPPORTED,
                supporting_evidence=[
                    Evidence(page=1, quote="nowhere in the paper at all", confidence=0.9)
                ],
            )
        ]
    )
    score, detail = verification_accuracy(record, index)
    assert score == 0.0
    assert detail["disagreements"][0]["oracle"] == "ungrounded"


def _verdicts(index, status):
    """One well-grounded claim (its finding quotes the paper) and one fabricated claim."""
    chunk = next(c for c in index.chunks if "81.4" in c.text)
    at = chunk.text.index("81.4")
    quote = chunk.text[max(0, at - 40) : at + 40]
    findings = [
        Finding(
            claim="SRT-Large reaches 81.4 F1",
            finding_id="f-real",
            evidence=[Evidence(page=chunk.page, quote=quote, confidence=0.9)],
        ),
        Finding(claim="SRT-Large reaches 97.2 F1", finding_id="f-fake"),
    ]
    verifications = [
        VerificationResult(claim=f.claim, finding_id=f.finding_id, status=status) for f in findings
    ]
    return _record(findings=findings, verifications=verifications)


@pytest.mark.parametrize("status", [FindingStatus.UNSUPPORTED, FindingStatus.SUPPORTED])
def test_a_verifier_that_always_gives_the_same_verdict_cannot_score_well(index, status):
    """Abstaining on everything, or approving everything, is half right at best."""
    score, detail = verification_accuracy(_verdicts(index, status), index)
    assert score == 0.5
    assert detail["classes"] == 2


def test_the_oracle_grounds_a_claim_from_its_finding_when_the_verifier_cites_nothing(index):
    record = _verdicts(index, FindingStatus.UNSUPPORTED)
    record.verification_results[0].status = FindingStatus.SUPPORTED
    score, _ = verification_accuracy(record, index)
    assert score == 1.0


def test_a_real_quote_does_not_ground_a_claim_whose_number_is_invented(index):
    record = _verdicts(index, FindingStatus.SUPPORTED)
    record.verification_results[1].supporting_evidence = list(record.findings[0].evidence)
    _, detail = verification_accuracy(record, index)
    wrong = detail["disagreements"][0]
    assert wrong["oracle"] == "ungrounded" and "97.2" in wrong["reason"]


def test_verification_accuracy_is_none_without_verifications(index):
    assert verification_accuracy(_record(), index)[0] is None


# --- record-level scoring --------------------------------------------------


def test_evaluate_record_produces_a_complete_scorecard(index):
    record = _record(
        findings=[Finding(claim="SRT-Large reaches 81.4 F1", evidence=[_evidence(index, "81.4")])]
    )
    metrics = evaluate_record(record, index, expected_claims=["SRT-Large reaches 81.4 F1"])
    assert metrics.claim_coverage == 1.0
    assert metrics.citation_precision == 1.0
    assert metrics.numeric_accuracy == 1.0
    assert metrics.findings == 1
    assert "citation_precision" in metrics.detail
    assert isinstance(metrics.to_dict(), dict)


def test_aggregate_ignores_unmeasurable_metrics():
    from paperpilot.evaluation import EvaluationMetrics

    results = [
        EvaluationMetrics(document_id="a", paper="a.pdf", citation_precision=1.0, status="completed"),
        EvaluationMetrics(document_id="b", paper="b.pdf", citation_precision=None, status="completed"),
    ]
    summary = aggregate(results)
    assert summary["mean_citation_precision"] == 1.0
    assert summary["measured_citation_precision"] == 1
    assert summary["mean_claim_coverage"] is None


# --- expectations loading --------------------------------------------------


def test_load_expectations(tmp_path):
    path = tmp_path / "expectations.json"
    path.write_text(
        json.dumps(
            {"papers": [{"filename": "a.pdf", "expected_claims": ["one", "two"]}]}
        )
    )
    expectations = load_expectations(path)
    assert expectations["a.pdf"].expected_claims == ["one", "two"]


def test_missing_expectations_file_is_not_fatal(tmp_path):
    assert load_expectations(tmp_path / "nope.json") == {}
    assert load_expectations(None) == {}


def test_malformed_expectations_file_is_not_fatal(tmp_path):
    path = tmp_path / "expectations.json"
    path.write_text("{ not valid json")
    assert load_expectations(path) == {}


def test_expectation_entries_without_a_filename_are_skipped(tmp_path):
    path = tmp_path / "expectations.json"
    path.write_text(json.dumps({"papers": [{"expected_claims": ["x"]}, "junk"]}))
    assert load_expectations(path) == {}


# --- the harness end to end ------------------------------------------------


def test_harness_runs_a_benchmark_directory(tmp_path, settings, service, paper_pdf, expected):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "paper.pdf").write_bytes(paper_pdf.read_bytes())

    from paperpilot.evaluation import PaperExpectation

    harness = EvaluationHarness(
        settings,
        service=service,
        expectations={
            "paper.pdf": PaperExpectation(
                filename="paper.pdf", expected_claims=expected["expected_claims"]
            )
        },
    )
    results = harness.evaluate(papers)

    assert results["aggregate"]["papers"] == 1
    assert results["aggregate"]["completed"] == 1
    assert results["aggregate"]["mean_citation_precision"] == 1.0
    assert results["per_paper"][0]["status"] == "completed"
    assert results["raw"][0]["report_markdown"]


def test_harness_writes_metrics_raw_and_per_paper_files(tmp_path, settings, service, paper_pdf):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "paper.pdf").write_bytes(paper_pdf.read_bytes())

    harness = EvaluationHarness(settings, service=service)
    results = harness.evaluate(papers)
    paths = harness.save(results, tmp_path / "out")

    assert set(paths) == {"metrics", "per_paper", "raw"}
    for path in paths.values():
        assert path.exists()
        json.loads(path.read_text())


def test_harness_summary_is_readable(tmp_path, settings, service, paper_pdf):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "paper.pdf").write_bytes(paper_pdf.read_bytes())

    harness = EvaluationHarness(settings, service=service)
    summary = harness.format_summary(harness.evaluate(papers))
    assert "citation precision" in summary
    assert "mock provider" in summary


def test_harness_rejects_an_empty_directory(tmp_path, settings, service):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        EvaluationHarness(settings, service=service).evaluate(empty)


def test_harness_records_a_failed_paper_without_aborting(tmp_path, settings, service, paper_pdf):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "good.pdf").write_bytes(paper_pdf.read_bytes())
    (papers / "bad.pdf").write_bytes(b"%PDF-1.4 corrupt beyond repair")

    results = EvaluationHarness(settings, service=service).evaluate(papers)
    assert results["aggregate"]["papers"] == 2
    assert results["aggregate"]["completed"] == 1
    assert results["aggregate"]["failed"] == 1
