"""Claims that the paper is silent on something are checked against the paper itself."""

from __future__ import annotations

import pytest

from paperpilot.domain.models import PaperChunk, PaperMetadata
from paperpilot.retrieval.index import PaperIndex
from paperpilot.validation.absence import PaperCoverage

SECTIONS = [
    ("5.1 Training Data and Batching", "We trained on the standard WMT 2014 dataset of 4.5 million sentence pairs."),
    ("5.2 Hardware and Schedule", "We trained our models on one machine with 8 GPUs."),
    ("5.3 Optimizer", "We used Adam. We varied the learning rate over the course of training, according to the formula."),
    ("5.4 Regularization", "We employ three types of regularization during training."),
    ("4 Proposed Experimental Evaluation", "Experimental results will be reported in future work."),
    ("References", "[1] A. Author. Statistical significance testing for error analysis. 2016."),
]


@pytest.fixture(scope="module")
def coverage():
    chunks = [
        PaperChunk(chunk_id=f"c{i}", document_id="d", page=7, section=heading, text=text)
        for i, (heading, text) in enumerate(SECTIONS)
    ]
    index = PaperIndex(document_id="d", metadata=PaperMetadata(document_id="d"), chunks=chunks)
    index._reindex()
    return PaperCoverage(index)


@pytest.mark.parametrize(
    ("text", "where"),
    [
        ("Training data and batching details are not stated in the provided context.", "§5.1"),
        ("Hardware details are not specified.", "§5.2"),
        ("The paper does not describe the optimizer.", "§5.3"),
        ("Regularization techniques are not described in the excerpt.", "§5.4"),
    ],
)
def test_a_refuted_absence_claim_is_removed_and_located(coverage, text, where):
    out, [found] = coverage.check(text)
    assert out == ""
    assert where in found.where and "the paper covers" in found.describe()


def test_a_parenthetical_absence_is_cut_and_the_sentence_kept(coverage):
    out, [found] = coverage.check(
        "The learning rate varies over training according to a formula (not detailed here)."
    )
    assert out == "The learning rate varies over training according to a formula."
    assert found.subject == "learning rate" and "§5.3" in found.where


def test_a_bare_field_placeholder_uses_the_field_subject(coverage):
    _, [found] = coverage.check("Not explicitly stated", subject="hardware")
    assert "§5.2" in found.where


@pytest.mark.parametrize(
    "text",
    [
        # True absences: nothing in the paper covers them.
        "The paper does not report statistical significance.",
        "Not explicitly discussed in the provided context",
        "Not explicitly stated train/validation/test splits for WMT datasets",
        # A paper that only promises results does not cover them.
        "The authors have not yet reported experimental results.",
        # Content, not a claim about the text.
        "ChatGPT could help patients when no other way of help is available.",
    ],
)
def test_true_absences_and_content_are_left_alone(coverage, text):
    out, found = coverage.check(text, subject="error analysis")
    assert found == [] and out == text


def test_a_claim_inside_a_longer_sentence_is_flagged_but_the_sentence_kept(coverage):
    text = "The authors describe a training setup but do not describe the optimizer."
    out, [found] = coverage.check(text)
    assert out == text and not found.remove and "§5.3" in found.where
