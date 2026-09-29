"""Deterministic completeness: nothing a paper states is lost between PDF and report.

Every behaviour here is layout- or text-driven and paper-independent: table grids
from cell geometry, display equations from indentation, floats attributed to the
section that cites them, quotes and values located wherever they really are,
self-contradicting numbers surfaced, and every content chunk read by a model.
"""

from __future__ import annotations

from conftest import _index_from_texts
from paperpilot.agents.coverage import BATCH_CHUNKS, plan_coverage
from paperpilot.agents.grounding import SourceMap, resolve_source_labels
from paperpilot.domain.enums import CitationQuality, ElementType
from paperpilot.domain.models import AgentResult, Evidence, Finding, MetricResult, PaperChunk
from paperpilot.ingestion.chunker import _split_long_text, attribute_floats, element_counts
from paperpilot.ingestion.elements import assemble_display_equations
from paperpilot.ingestion.pdf_parser import TextBlock, _Cell, _is_folio, _span_text, table_grid_text
from paperpilot.response.parser import strip_citations
from paperpilot.retrieval.retriever import RetrievalResult
from paperpilot.validation.consistency import annotate_conflicts, find_value_conflicts
from paperpilot.validation.evidence import validate_evidence
from paperpilot.validation.quotes import match_quote, normalize_for_match
from paperpilot.validation.values import locate_value

# --- ingestion ---------------------------------------------------------------


def _cell(text: str, x0: float, x1: float, y: float) -> _Cell:
    return _Cell(text=text, x0=x0, x1=x1, y=y, height=10.0)


def test_table_grid_keeps_values_under_their_headers():
    cells = [
        _cell("BLEU", 300, 340, 100),  # spans the two score columns
        _cell("Model", 130, 160, 110),
        _cell("EN-DE", 290, 320, 118),
        _cell("EN-FR", 330, 360, 118),
        _cell("ByteNet", 130, 170, 130),
        _cell("23.75", 292, 315, 130),  # EN-FR left blank
        _cell("Big", 130, 150, 142),
        _cell("28.4", 295, 312, 142),
        _cell("41.8", 336, 354, 142),
    ]
    grid = table_grid_text(cells)
    assert grid is not None
    lines = grid.split("\n")
    assert lines[0] == "| Model | BLEU EN-DE | BLEU EN-FR |"
    assert "| ByteNet | 23.75 |  |" in lines
    assert "| Big | 28.4 | 41.8 |" in lines


def test_prose_under_a_caption_is_not_a_grid():
    cells = [_cell("One long line of prose.", 100, 500, 100), _cell("Another.", 100, 480, 112)]
    assert table_grid_text(cells) is None


def test_superscripts_are_marked_only_when_attached():
    raised = {"flags": 1}
    assert _span_text([{"text": "1.0 · 10", "flags": 0}, {"text": "20", **raised}]) == "1.0 · 10^20"
    assert _span_text([{"text": "gradients ", "flags": 0}, {"text": "4", **raised}]) == "gradients 4"


def test_a_raised_numerator_is_not_a_superscript():
    spans = [
        {"text": "softmax(", "flags": 0, "size": 10.0},
        {"text": "QK", "flags": 1, "size": 10.0},
        {"text": "T", "flags": 1, "size": 7.0},
    ]
    assert _span_text(spans) == "softmax(QK^T"


def test_running_page_numbers_are_dropped_but_body_numbers_are_not():
    assert _is_folio({"bbox": (300, 760, 310, 770)}, "7", 792.0)
    assert not _is_folio({"bbox": (300, 400, 310, 410)}, "7", 792.0)
    assert not _is_folio({"bbox": (300, 760, 310, 770)}, "Results", 792.0)


def _block(text: str, x0: float, x1: float, y0: float, y1: float) -> TextBlock:
    return TextBlock(page=1, text=text, bbox=(x0, y0, x1, y1), font_size=10.0)


def test_a_centred_named_function_is_a_display_equation():
    body = "Running text of the paper that fills the column from margin to margin. " * 3
    blocks = [
        _block(body, 108, 504, 100, 130),
        _block("MultiHead(Q, K, V) = Concat(head1, ..., headh)W", 187, 411, 146, 158),
        _block(body, 108, 504, 170, 200),
    ]
    merged = assemble_display_equations(blocks)
    assert [b.element_hint for b in merged] == [None, ElementType.EQUATION, None]


def test_an_inline_formula_in_running_text_stays_prose():
    body = "We set dk = dv = dmodel/h = 64 for every head in the running text of the paper."
    blocks = [_block(body * 2, 108, 504, 100, 130)]
    assert [b.element_hint for b in assemble_display_equations(blocks)] == [None]


def _chunk(position: int, page: int, section: str, text: str, **kw) -> PaperChunk:
    return PaperChunk(
        chunk_id=f"doc-page-{page}-chunk-{position:02d}",
        document_id="doc",
        page=page,
        section=section,
        text=text,
        **kw,
    )


def test_a_float_belongs_to_the_section_that_cites_it():
    chunks = [
        _chunk(1, 7, "5 Training", "We employ three types of regularization:"),
        _chunk(2, 8, "5 Training", "Table 2: Results.", element_type=ElementType.TABLE, element_id="table-2"),
        _chunk(3, 8, "5 Training", "| a | 1 |", element_type=ElementType.TABLE, element_id="table-2"),
        _chunk(4, 8, "6 Results", "Our big model (Table 2) reaches 28.4."),
    ]
    attribute_floats(chunks)
    assert {c.section for c in chunks if c.element_id == "table-2"} == {"6 Results"}


def test_elements_are_counted_once_however_many_chunks_they_span():
    chunks = [
        _chunk(1, 1, "S", "caption", element_type=ElementType.TABLE, element_id="table-1"),
        _chunk(2, 1, "S", "body", element_type=ElementType.TABLE, element_id="table-1"),
        _chunk(3, 1, "S", "text"),
    ]
    assert element_counts(chunks) == {"table": 1, "paragraph": 1}


def test_overlapping_pieces_never_start_mid_word():
    text = "Self-attention, sometimes called intra-attention, relates positions. " * 40
    for piece in _split_long_text(text, max_chars=300, overlap=60)[1:]:
        assert piece[0].isupper() or piece.startswith(("intra", "relates", "sometimes", "called"))


# --- validation ----------------------------------------------------------------

PAGE_ONE = "Some unrelated introduction text about sequence models and their training."
PAGE_TWO = (
    "Similarly to other models, we use learned embeddings of dimension dmodel. In our "
    "model, we share the same weight matrix between the two embedding layers and the "
    "pre-softmax linear transformation, similar to [30]."
)


def test_sentence_punctuation_and_latex_do_not_break_a_quote():
    quote = (
        "we use learned embeddings of dimension d_{model}. ... we share the same weight "
        "matrix between the two embedding layers and the pre-softmax linear transformation."
    )
    assert match_quote(quote, PAGE_TWO).found
    assert "81.4%" in normalize_for_match("reaches 81.4% F1.")


def test_a_real_quote_cited_to_the_wrong_page_is_relocated():
    index = _index_from_texts([PAGE_ONE, PAGE_TWO])
    evidence = Evidence(page=1, quote="we share the same weight matrix between the two embedding layers")
    report = validate_evidence(evidence, index)
    assert report.ok
    assert evidence.page == 2
    assert evidence.citation_quality is not CitationQuality.UNVERIFIABLE
    assert any("corrected" in note for note in evidence.validation_notes)


TABLE_3 = (
    "| | N | dmodel | PPL (dev) | BLEU (dev) |\n| --- | --- | --- | --- | --- |\n"
    "| base | 6 | 512 | 4.92 | 25.8 |\n| (A) | | | 5.29 | 24.9 |\n| big | 6 | 1024 | 4.33 | 26.4 |"
)


def _table_index():
    index = _index_from_texts([PAGE_ONE, TABLE_3])
    index.chunks[1].element_type = ElementType.TABLE
    index.chunks[1].element_id = "table-3"
    return index


def test_non_adjacent_rows_of_the_cited_table_are_a_faithful_quote():
    index = _table_index()
    quote = "| base | 6 | 512 | 4.92 | 25.8 |\n| big | 6 | 1024 | 4.33 | 26.4 |"
    evidence = Evidence(page=2, chunk_id=index.chunks[1].chunk_id, quote=quote)
    assert validate_evidence(evidence, index).ok
    assert evidence.quote == quote
    assert evidence.element_id == "table-3"


def test_a_paraphrased_table_reading_is_replaced_by_its_rows():
    index = _table_index()
    evidence = Evidence(
        page=2,
        chunk_id=index.chunks[1].chunk_id,
        quote="Table 3 shows the big model reaching PPL 4.33 and BLEU 26.4 against 25.8 for base.",
    )
    assert validate_evidence(evidence, index).ok
    assert evidence.quote.splitlines() == ["| base | 6 | 512 | 4.92 | 25.8 |", "| big | 6 | 1024 | 4.33 | 26.4 |"]
    assert any("paraphrased" in note for note in evidence.validation_notes)


def test_a_table_reading_with_an_invented_number_is_not_accepted():
    index = _table_index()
    evidence = Evidence(
        page=2, chunk_id=index.chunks[1].chunk_id, quote="Table 3 shows BLEU 26.4 for big and 27.9 for huge."
    )
    assert not validate_evidence(evidence, index).ok


RESULTS_TABLE = (
    "| Model | BLEU EN-DE | BLEU EN-FR |\n| --- | --- | --- |\n"
    "| Transformer (base model) | 27.3 | 38.1 |\n| Transformer (big) | 28.4 | 41.8 |"
)
RESULTS_PROSE = (
    "On the WMT 2014 English-to-French translation task, our big model achieves a BLEU "
    "score of 41.0, outperforming all previous single models. The big model outperforms "
    "the best previously reported models by more than 2.0 BLEU."
)


def _results_index():
    index = _index_from_texts([RESULTS_TABLE, RESULTS_PROSE])
    table = index.chunks[0]
    table.element_type = ElementType.TABLE
    table.element_id = "table-2"
    return index


def test_a_value_is_located_on_the_row_that_names_its_system():
    index = _results_index()
    evidence = locate_value("27.3", index, name="Transformer (base)", cited=[])
    assert evidence is not None
    assert evidence.element_id == "table-2"
    assert "Transformer (base model)" in evidence.quote
    assert locate_value("27.3", index, name="ConvS2S", cited=[]) is None


def test_a_value_the_paper_states_two_ways_is_flagged():
    index = _results_index()
    results = [
        MetricResult(metric="BLEU", value="41.8", system="Transformer (big)", dataset="WMT 2014 English-French"),
        MetricResult(metric="BLEU", value="28.4", system="Transformer (big)", dataset="WMT 2014 English-German"),
        MetricResult(metric="BLEU", value="27.3", system="Transformer (base)", dataset="WMT 2014 English-German"),
    ]
    conflicts = find_value_conflicts(results, index)
    assert [(c.reported_value, c.stated_value) for c in conflicts] == [("41.8", "41.0")]

    finding = Finding(claim="The big model reaches 41.8 BLEU on English-French.")
    annotate_conflicts([finding], [c.to_dict() for c in conflicts])
    annotate_conflicts([finding], [c.to_dict() for c in conflicts])  # idempotent
    assert finding.verification_note.count("internally inconsistent") == 1


# --- response text -------------------------------------------------------------


def test_adjacent_citation_markers_leave_no_orphan_punctuation():
    assert strip_citations("Bigger models are better [S6], [S20].") == "Bigger models are better."
    assert strip_citations("a [S1] and [S2] b") == "a b"


def test_source_labels_in_prose_become_locations():
    index = _index_from_texts([PAGE_ONE, PAGE_TWO])
    sources = SourceMap([RetrievalResult(chunk=index.chunks[1], score=1.0)])
    text = resolve_source_labels("The evidence in S1 states it; see [S1]. S9 is unknown.", sources)
    assert "S1" not in text
    assert "[p. 2, §2 Section]" in text
    assert "S9" in text


# --- coverage ------------------------------------------------------------------


def test_the_sweep_reads_exactly_what_no_specialist_was_shown():
    texts = [f"Content paragraph number {i} with enough words to count as content." for i in range(1, 41)]
    index = _index_from_texts(texts)
    table_body = index.chunks[5]
    table_caption = index.chunks[4]
    for chunk in (table_caption, table_body):
        chunk.element_type = ElementType.TABLE
        chunk.element_id = "table-1"

    shown = [c.chunk_id for c in index.chunks[:4]]
    results = {"summary": AgentResult(agent="summary", context_chunk_ids=shown)}
    plan = plan_coverage(index, results, max_batches=10)

    swept = [c.chunk_id for batch in plan.batches for c in batch]
    assert set(swept) == {c.chunk_id for c in index.chunks} - set(shown)
    assert swept == [c.chunk_id for c in index.chunks if c.chunk_id not in shown]
    assert all(len(batch) <= BATCH_CHUNKS for batch in plan.batches)
    assert any({table_caption.chunk_id, table_body.chunk_id} <= {c.chunk_id for c in b} for b in plan.batches)
    assert plan_coverage(index, results, max_batches=10).batches == plan.batches  # deterministic

    limited = plan_coverage(index, results, max_batches=1)
    assert len(limited.batches) == 1 and limited.unread


def test_terms_of_art_the_paper_never_uses_are_flagged():
    from paperpilot.validation.terms import PaperVocabulary

    index = _index_from_texts(["We employed label smoothing and a shared source-\ntarget vocabulary; BLEU improved."])
    vocabulary = PaperVocabulary(index)
    assert vocabulary.unstated_terms("Decoding is source-targeted.") == []
    assert PaperVocabulary(_index_from_texts(["F1 on WSJ."])).unstated_terms("F1 improves.") == []
    text = "Trained with cross-entropy loss and label smoothing over a source-target vocabulary; BLEU and ROUGE."
    assert vocabulary.unstated_terms(text) == ["cross-entropy", "ROUGE"]


def test_a_not_stated_placeholder_does_not_tick_the_reproducibility_checklist():
    """An agent reporting an absence must not read as having found the item."""
    from paperpilot.reporting.synthesizer import _stated

    for absent in (
        "Not explicitly stated.",
        "not specified",
        "None reported",
        "N/A",
        "Not applicable",
        "unclear",
    ):
        assert not _stated(absent), absent

    assert _stated("80/10/10 train/dev/test") == "80/10/10 train/dev/test"
    assert _stated(["WMT 2014", "Not stated"]) == ["WMT 2014"]


def test_a_chunk_read_twice_is_counted_once_in_the_coverage_total():
    """A sweep that re-reads a specialist's chunk must not inflate the total."""
    from paperpilot.agents.coverage import merge_coverage_results

    texts = [f"Content paragraph number {i} with enough words to count as content." for i in range(1, 13)]
    index = _index_from_texts(texts)
    shown = [c.chunk_id for c in index.chunks[:4]]
    plan = plan_coverage(index, {"summary": AgentResult(agent="summary", context_chunk_ids=shown)}, max_batches=10)

    # The sweep re-reads two chunks the specialist already saw.
    swept_ids = [c.chunk_id for batch in plan.batches for c in batch] + shown[:2]
    sweep = AgentResult(agent="coverage", status="ok", context_chunk_ids=swept_ids)

    merged = merge_coverage_results([sweep], plan=plan, index=index)
    payload = merged.payload
    total_read = payload["read_by_specialists"] + payload["read_by_sweep"]
    assert total_read <= payload["content_chunks"]
    assert total_read == len(set(shown) | set(swept_ids))
