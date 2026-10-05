"""What the paper states but the report leaves out is shown, found without a model."""

from __future__ import annotations

import pytest

from paperpilot.domain.enums import ElementType
from paperpilot.domain.models import PaperChunk, PaperMetadata
from paperpilot.reporting.completeness import StatedSettings, partly_read_tables
from paperpilot.retrieval.index import PaperIndex

RESULTS = """| Model | BLEU EN-DE | BLEU EN-FR | Training Cost (FLOPs) EN-DE |
| --- | --- | --- | --- |
| ConvS2S | 25.16 | 40.46 | 9.6 · 10^18 |
| Transformer (big) | 28.4 | 41.8 | 2.3 · 10^19 |"""

VARIATIONS = """| | N | d_model | h | PPL (dev) | BLEU (dev) | params |
| --- | --- | --- | --- | --- | --- | --- |
| base | 6 | 512 | 8 | 4.92 | 25.8 | 65 |
| big | 6 | 1024 | 16 | 4.33 | 26.4 | 213 |"""

PROSE = (
    "We used warmup_steps = 4000. During training, we employed label smoothing of value "
    "ϵls = 0.1 [36]. For the base model, we use a rate of Pdrop = 0.1. Their dot product "
    "q · k = P^dk i=1 q^ik^i has mean 0."
)


def _table(eid: str, text: str, page: int) -> list[PaperChunk]:
    return [
        PaperChunk(chunk_id=f"{eid}-cap", document_id="d", page=page, element_type=ElementType.TABLE,
                   element_id=eid, text=f"{eid}: a caption."),
        PaperChunk(chunk_id=f"{eid}-grid", document_id="d", page=page, element_type=ElementType.TABLE,
                   element_id=eid, text=text),
    ]


@pytest.fixture(scope="module")
def index():
    chunks = [
        PaperChunk(chunk_id="p", document_id="d", page=7, section="5 Training", text=PROSE),
        *_table("Table 2", RESULTS, 8),
        *_table("Table 3", VARIATIONS, 9),
    ]
    idx = PaperIndex(document_id="d", metadata=PaperMetadata(document_id="d"), chunks=chunks)
    idx._reindex()
    return idx


def _reported(value: str, table: str) -> dict:
    return {"value": value, "evidence": [{"element_id": table}]}


def test_unread_columns_of_a_cited_table_are_shown(index):
    [table] = [t for t in partly_read_tables(index, [_reported("28.4", "Table 2"), _reported("41.8", "Table 2")])]
    assert not table.whole and table.read_columns == ["BLEU EN-DE", "BLEU EN-FR"]
    assert table.header == ["Model", "Training Cost (FLOPs) EN-DE"]
    assert ["Transformer (big)", "2.3 · 10^19"] in table.rows


def test_a_mostly_unread_table_is_shown_whole(index):
    [table] = partly_read_tables(index, [_reported("4.33", "Table 3")])
    assert table.whole and ["big", "6", "1024", "16", "4.33", "26.4", "213"] in table.rows


def test_a_fully_read_or_uncited_table_adds_nothing(index):
    every = [_reported(v, "Table 2") for v in ("25.16", "40.46", "9.6 · 10^18")]
    assert partly_read_tables(index, every) == []


def test_settings_the_report_never_states_are_listed(index):
    settings = StatedSettings(index)
    assert [s.chain for s in settings.settings] == ["warmup_steps = 4000", "ϵls = 0.1", "Pdrop = 0.1"]
    report = (
        "| Warmup steps | 4000 |\n"
        "| Dropout (base model) | 0.1 |\n"            # Pdrop = 0.1, said in other words
        "- dropout 0.1 ⚠️ _Numbers not as in the paper: ϵ_ls = 0.1 quoted in a warning._\n"
    )
    assert [s.chain for s in settings.unmentioned(report)] == ["ϵls = 0.1"]
    assert settings.unmentioned(report + "| Label smoothing ϵ_ls | 0.1 |\n") == []
