"""Numbers stated together in model-written text must be stated together in the paper."""

from __future__ import annotations

import pytest

from paperpilot.domain.enums import ElementType
from paperpilot.domain.models import PaperChunk, PaperMetadata
from paperpilot.retrieval.index import PaperIndex
from paperpilot.validation.cooccurrence import PaperNumbers

TABLE = """| | N | h | d_k | PPL (dev) | BLEU (dev) |
| --- | --- | --- | --- | --- | --- |
| base | 6 | 8 | 64 | 4.92 | 25.8 |
| (A) | | 1 | 512 | 5.29 | 24.9 |
| (A) | | 16 | 32 | 4.91 | 25.8 |
| (C) | 2 | | | 6.11 | 23.7 |
| (D) dropout 0.0 | | | | 5.77 | 24.6 |"""

PROSE = (
    "We trained the base models for a total of 100,000 steps or 12 hours. The big models "
    "were trained for 300,000 steps (3.5 days). On the WMT 2014 English-to-German task the "
    "big model reaches 28.4 BLEU [38, 2, 9]. We used beam search with a beam size of 4."
)


@pytest.fixture(scope="module")
def numbers():
    chunks = [
        PaperChunk(chunk_id="c1", document_id="d", page=7, text=PROSE),
        PaperChunk(chunk_id="c2", document_id="d", page=9, text=TABLE, element_type=ElementType.TABLE),
    ]
    index = PaperIndex(document_id="d", metadata=PaperMetadata(document_id="d"), chunks=chunks)
    index._reindex()
    return PaperNumbers(index)


@pytest.mark.parametrize(
    "text",
    [
        "h=1 yields PPL 5.29 and BLEU 24.9",
        "N=2 PPL 6.11 BLEU 23.7, dropout 0.0 PPL 5.77 BLEU 24.6",
        "Varying the number of heads h from 1 to 16 changes BLEU between 24.9 and 25.8.",
        "BLEU stays around 24.9-25.8 as heads vary",
        "h=16 reaches PPL 4.91 vs PPL 4.92 for the base model",
        "Training steps (base model): 100,000 steps (12 hours)",
        "28.4 BLEU on WMT 2014 English-to-German [38, 2, 9]",
        "The base model trained 100,000 steps. The big model trained 300,000 steps (3.5 days).",
        "Nothing numeric here.",
    ],
)
def test_faithful_statements_pass(numbers, text):
    assert numbers.audit(text) == []


def test_a_setting_paired_with_another_rows_results_is_caught(numbers):
    """The h=8 base row has 4.92; 4.91 belongs to h=16. Each number exists; the pairing does not."""
    [mismatch] = numbers.audit("e.g., h=1 yields PPL 5.29 and BLEU 24.9, h=8 yields PPL 4.91 and BLEU 25.8")
    assert mismatch.unpaired == ["8", "4.91", "25.8"]
    rows = [text for text, _ in mismatch.nearest]
    assert any("h = 16" in r and "PPL (dev) = 4.91" in r for r in rows)
    assert any("h = 8" in r and "PPL (dev) = 4.92" in r for r in rows)
    assert "never states 8, 4.91, 25.8 together" in mismatch.describe()


def test_values_from_unrelated_sentences_are_not_paired(numbers):
    """A sentence and the next one count as stated together; sentences further apart do not."""
    [mismatch] = numbers.audit("The base model trained for 100,000 steps with a beam size of 4")
    assert mismatch.unpaired == ["100,000", "4"]


def test_a_number_the_paper_never_prints_is_reported_as_absent(numbers):
    [mismatch] = numbers.audit("dropout 0.3 gives BLEU 26.9")
    assert mismatch.absent == ["0.3", "26.9"]


def test_the_report_flags_the_pairing_next_to_the_ablation(numbers, monkeypatch):
    from paperpilot.reporting.synthesizer import ReportSynthesizer

    synth = ReportSynthesizer()
    synth._numbers = numbers
    synth._number_flags = []
    line = synth._prose("h=8 yields PPL 4.91 and BLEU 25.8")
    assert "⚠️ _Numbers not as in the paper" in line and "h = 16" in line
    assert synth._prose("h=16 yields PPL 4.91 and BLEU 25.8") == "h=16 yields PPL 4.91 and BLEU 25.8"
    assert len(synth._number_flags) == 1
