"""Deterministic validation: quotes, numbers, schemas, evidence and citations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from paperpilot.domain.enums import CitationQuality, ElementType, FindingStatus
from paperpilot.domain.models import Evidence, Finding
from paperpilot.reporting.citations import (
    UNVERIFIED_MARKER,
    format_citation,
    format_citations,
    format_quote,
)
from paperpilot.validation.evidence import validate_evidence, validate_finding
from paperpilot.validation.numeric import (
    check_claim_numbers,
    check_numbers_against_text,
    extract_numbers,
    numeric_accuracy,
    parse_number,
)
from paperpilot.validation.quotes import match_quote, normalize_for_match
from paperpilot.validation.schema import (
    clamp_confidence,
    coerce_element_type,
    coerce_status,
    parse_evidence,
    parse_finding,
    parse_findings,
)

SOURCE = (
    "SRT-Large reaches 81.4 F1 on NarrativeQA while reducing attention FLOPs by 3.1x "
    "relative to a dense baseline of the same parameter count."
)


# --- quote matching --------------------------------------------------------


def test_exact_quote_is_recognised():
    match = match_quote("SRT-Large reaches 81.4 F1", SOURCE)
    assert match.quality is CitationQuality.EXACT
    assert match.similarity == 1.0


def test_whitespace_differences_are_normalised_not_rejected():
    match = match_quote("SRT-Large   reaches\n81.4  F1", SOURCE)
    assert match.quality is CitationQuality.NORMALIZED
    assert match.found


def test_ocr_style_noise_matches_approximately_with_lower_confidence():
    noisy = "SRT-Larqe reaches 81.4 F1 on NarratlveQA"
    match = match_quote(noisy, SOURCE)
    assert match.quality is CitationQuality.APPROXIMATE
    assert match.adjusted_confidence(0.9) < 0.9


def test_invented_quote_is_rejected():
    match = match_quote(
        "The method achieves 99.9% accuracy on ImageNet and was deployed worldwide.", SOURCE
    )
    assert match.quality is CitationQuality.UNVERIFIABLE
    assert not match.found


def test_empty_inputs_are_unverifiable():
    assert not match_quote("", SOURCE).found
    assert not match_quote("something", "").found


def test_normalize_for_match_is_aggressive_but_keeps_numbers():
    normalized = normalize_for_match("SRT-Large  reaches— 81.4% F1!")
    assert "81.4%" in normalized
    assert "  " not in normalized


def test_confidence_penalty_ordering():
    """Each quality tier must cost strictly more confidence than the one above."""
    from paperpilot.validation.quotes import QuoteMatch

    scores = [
        QuoteMatch(quality, 1.0).adjusted_confidence(1.0)
        for quality in (
            CitationQuality.EXACT,
            CitationQuality.NORMALIZED,
            CitationQuality.APPROXIMATE,
            CitationQuality.UNVERIFIABLE,
        )
    ]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == 4


def test_a_quote_with_ocr_noise_still_matches_approximately():
    noisy = (
        "SRT-Large reaches 81.4 F1 on NarratlveQA whlle reducing attentlon FLOPs by 3.1x "
        "relatlve to a dense basellne of the same parameter count."
    )
    match = match_quote(noisy, SOURCE)
    assert match.quality is CitationQuality.APPROXIMATE
    assert match.adjusted_confidence(0.9) < 0.9


# --- numeric validation ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("81.4", ["81.4"]),
        ("3.1x faster", ["3.1"]),
        ("92%", ["92"]),
        ("1,024 tokens", ["1,024"]),
        ("3e-5 learning rate", ["3"]),
        ("no numbers here", []),
    ],
)
def test_extract_numbers(text, expected):
    assert [n.raw for n in extract_numbers(text)] == expected


def test_trailing_zeros_compare_equal():
    checks = check_numbers_against_text("the score was 92.40", "we report 92.4 F1")
    assert all(c.found_in_evidence for c in checks)


def test_a_number_absent_from_the_source_is_flagged():
    checks = check_numbers_against_text("the score was 95.0", SOURCE)
    assert checks and not checks[0].found_in_evidence
    assert checks[0].note


def test_numbers_are_checked_against_the_union_of_evidence():
    evidence = [
        Evidence(page=4, quote="SRT-Large reaches 81.4 F1", confidence=0.9),
        Evidence(page=4, quote="reducing attention FLOPs by 3.1x", confidence=0.9),
    ]
    checks = check_claim_numbers("SRT-Large reaches 81.4 F1 at 3.1x lower cost", evidence)
    assert numeric_accuracy(checks) == 1.0


def test_claim_without_evidence_fails_every_number():
    checks = check_claim_numbers("achieves 99.9% accuracy", [])
    assert checks and not checks[0].found_in_evidence


def test_numeric_accuracy_is_vacuously_true_without_numbers():
    assert numeric_accuracy([]) == 1.0


def test_parse_number_handles_separators_and_exponents():
    assert parse_number("1,024").value == 1024.0
    assert parse_number("3", "e-5").value == pytest.approx(3e-5)
    assert parse_number("not a number") is None


# --- schema coercion -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("supported", FindingStatus.SUPPORTED),
        ("Partially Supported", FindingStatus.PARTIALLY_SUPPORTED),
        ("contradiction", FindingStatus.CONTRADICTED),
        ("nonsense", FindingStatus.UNKNOWN),
        (None, FindingStatus.UNKNOWN),
    ],
)
def test_status_coercion(raw, expected):
    assert coerce_status(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("table", ElementType.TABLE), ("formula", ElementType.EQUATION), ("", ElementType.UNKNOWN)],
)
def test_element_type_coercion(raw, expected):
    assert coerce_element_type(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"), [(1.5, 1.0), (-2, 0.0), ("0.75", 0.75), ("abc", 0.5), (None, 0.5)]
)
def test_confidence_is_clamped(raw, expected):
    assert clamp_confidence(raw) == expected


def test_evidence_with_a_non_integer_page_is_dropped():
    assert parse_evidence({"page": "page seven", "quote": "text"}) is None


def test_evidence_without_a_quote_is_dropped():
    assert parse_evidence({"page": 3, "quote": "   "}) is None


def test_evidence_with_page_zero_is_dropped():
    assert parse_evidence({"page": 0, "quote": "text"}) is None


def test_finding_without_a_claim_is_dropped():
    assert parse_finding({"explanation": "no claim here"}, agent="test") is None


def test_parse_findings_skips_malformed_entries():
    findings = parse_findings(
        [{"claim": "good"}, {"no_claim": True}, "a string", None], agent="test"
    )
    assert len(findings) == 1 and findings[0].claim == "good"


def test_models_reject_out_of_range_confidence():
    with pytest.raises(PydanticValidationError):
        Evidence(page=1, quote="x", confidence=1.4)


def test_models_reject_an_empty_quote():
    with pytest.raises(PydanticValidationError):
        Evidence(page=1, quote="  ")


# --- evidence validation against the index ---------------------------------


def test_a_page_outside_the_document_is_rejected(index):
    evidence = Evidence(page=999, quote="anything", confidence=0.9)
    report = validate_evidence(evidence, index)
    assert not report.ok
    assert evidence.citation_quality is CitationQuality.UNVERIFIABLE
    assert evidence.confidence == 0.0
    assert "outside the document" in evidence.validation_notes[0]


def test_a_real_quote_validates_and_keeps_confidence(index):
    chunk = next(c for c in index.chunks if "81.4" in c.text)
    evidence = Evidence(
        page=chunk.page, quote=chunk.text[:80], chunk_id=chunk.chunk_id, confidence=0.9
    )
    report = validate_evidence(evidence, index)
    assert report.ok
    assert evidence.citation_quality is CitationQuality.EXACT
    assert evidence.confidence == pytest.approx(0.9)


def test_an_invented_quote_is_marked_unverifiable(index):
    evidence = Evidence(
        page=1, quote="This sentence appears nowhere in the paper at all.", confidence=0.95
    )
    report = validate_evidence(evidence, index)
    assert not report.ok
    assert evidence.confidence < 0.3


def test_a_nonexistent_chunk_id_is_dropped_but_the_page_is_kept(index):
    chunk = next(c for c in index.chunks if len(c.text) > 100)
    evidence = Evidence(
        page=chunk.page, quote=chunk.text[:60], chunk_id="bogus-chunk-id", confidence=0.8
    )
    report = validate_evidence(evidence, index)
    assert report.ok
    assert evidence.chunk_id != "bogus-chunk-id"
    assert any("does not exist" in n for n in evidence.validation_notes)


def test_a_wrong_section_label_is_corrected_from_the_index(index):
    chunk = next(c for c in index.chunks if c.citation_section == "4.2 Main Results")
    evidence = Evidence(
        page=chunk.page,
        section="9 Imaginary Section",
        quote=chunk.text[:60],
        chunk_id=chunk.chunk_id,
        confidence=0.8,
    )
    validate_evidence(evidence, index)
    assert evidence.section == "4.2 Main Results"
    assert any("corrected" in n for n in evidence.validation_notes)


def test_a_nonexistent_element_id_is_dropped(index):
    chunk = next(c for c in index.chunks if c.element_type is ElementType.PARAGRAPH)
    evidence = Evidence(
        page=chunk.page, quote=chunk.text[:60], element_id="table-99", confidence=0.8
    )
    validate_evidence(evidence, index)
    assert evidence.element_id != "table-99"


def test_unattributed_quotes_are_resolved_to_their_chunk(index):
    chunk = next(c for c in index.chunks if "load-balancing" in c.text)
    evidence = Evidence(page=chunk.page, quote=chunk.text[:70], confidence=0.8)
    validate_evidence(evidence, index)
    assert evidence.chunk_id == chunk.chunk_id


# --- finding validation ----------------------------------------------------


def test_a_finding_without_evidence_is_downgraded(index):
    finding = Finding(claim="Bold claim", status=FindingStatus.SUPPORTED, confidence=0.95)
    validate_finding(finding, index)
    assert finding.status is FindingStatus.UNSUPPORTED
    assert finding.confidence <= 0.3


def test_an_inferred_finding_without_evidence_keeps_its_honest_status(index):
    finding = Finding(claim="Probably uses 8 GPUs", status=FindingStatus.INFERRED, confidence=0.4)
    validate_finding(finding, index)
    assert finding.status is FindingStatus.INFERRED


def test_a_finding_whose_quotes_are_all_fake_is_downgraded(index):
    finding = Finding(
        claim="Fabricated",
        status=FindingStatus.SUPPORTED,
        confidence=0.9,
        evidence=[Evidence(page=1, quote="Nothing like this is in the paper.", confidence=0.9)],
    )
    validate_finding(finding, index)
    assert finding.status is FindingStatus.UNSUPPORTED


def test_a_supported_finding_with_a_bad_number_becomes_partially_supported(index):
    chunk = next(c for c in index.chunks if "81.4" in c.text)
    finding = Finding(
        claim="SRT-Large reaches 95.7 F1 on NarrativeQA",
        status=FindingStatus.SUPPORTED,
        confidence=0.9,
        evidence=[
            Evidence(
                page=chunk.page, quote=chunk.text[:120], chunk_id=chunk.chunk_id, confidence=0.9
            )
        ],
    )
    validate_finding(finding, index)
    assert finding.status is FindingStatus.PARTIALLY_SUPPORTED
    assert "95.7" in (finding.verification_note or "")


def test_a_finding_cannot_be_more_confident_than_its_evidence(index):
    chunk = next(c for c in index.chunks if len(c.text) > 120)
    finding = Finding(
        claim="A grounded claim",
        status=FindingStatus.SUPPORTED,
        confidence=1.0,
        evidence=[
            Evidence(
                page=chunk.page, quote=chunk.text[:80], chunk_id=chunk.chunk_id, confidence=0.6
            )
        ],
    )
    validate_finding(finding, index)
    assert finding.confidence <= 0.6


# --- citation formatting ---------------------------------------------------


def test_citation_format_matches_the_specification():
    evidence = Evidence(
        page=7,
        section="4.2 Main Results",
        source_type=ElementType.TABLE,
        element_id="table-3",
        quote="x",
        confidence=0.9,
        citation_quality=CitationQuality.EXACT,
    )
    assert format_citation(evidence) == "[p. 7, §4.2 Main Results, Table 3]"


def test_citation_without_a_section_or_element():
    evidence = Evidence(
        page=2, quote="x", confidence=0.8, citation_quality=CitationQuality.EXACT
    )
    assert format_citation(evidence) == "[p. 2]"


def test_unverifiable_evidence_never_renders_as_a_citation():
    evidence = Evidence(
        page=3, quote="x", confidence=0.0, citation_quality=CitationQuality.UNVERIFIABLE
    )
    assert format_citation(evidence) == UNVERIFIED_MARKER


def test_citation_groups_are_deduplicated():
    evidence = Evidence(
        page=4, quote="x", confidence=0.9, citation_quality=CitationQuality.EXACT
    )
    assert format_citations([evidence, evidence.model_copy()]) == "[p. 4]"


def test_no_evidence_renders_explicitly():
    assert format_citations([]) == "[no evidence]"


def test_quality_annotation_is_shown_for_approximate_matches():
    evidence = Evidence(
        page=4, quote="x", confidence=0.5, citation_quality=CitationQuality.APPROXIMATE
    )
    assert "approximate match" in format_citation(evidence, include_quality=True)


def test_quote_rendering_flags_unverifiable_evidence():
    evidence = Evidence(
        page=4, quote="invented", confidence=0.0, citation_quality=CitationQuality.UNVERIFIABLE
    )
    assert "not found in the paper" in format_quote(evidence)


def test_long_quotes_are_truncated():
    evidence = Evidence(
        page=1, quote="word " * 200, confidence=0.9, citation_quality=CitationQuality.EXACT
    )
    assert len(format_quote(evidence, max_chars=100)) < 130
