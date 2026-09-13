"""PDF extraction, OCR decisions, structure detection and chunk metadata."""

from __future__ import annotations

import pytest

from paperpilot.domain.enums import ElementType, ExtractionMethod
from paperpilot.errors import FileTooLargeError, UnsupportedFileError, ValidationError
from paperpilot.ingestion.chunker import chunk_document
from paperpilot.ingestion.elements import (
    classify_block,
    extract_element_id,
    looks_like_equation,
    looks_like_heading,
    looks_like_table,
)
from paperpilot.ingestion.ocr import NullOCREngine, OCRResult, build_ocr_engine
from paperpilot.ingestion.pdf_parser import (
    TextBlock,
    needs_ocr,
    normalize_text,
    parse_pdf,
    validate_pdf_file,
)

# --- page extraction -------------------------------------------------------


def test_every_page_is_extracted_with_its_number(paper_pdf, expected):
    doc = parse_pdf(paper_pdf)
    assert doc.page_count == expected["page_count"]
    assert [p.page for p in doc.pages] == list(range(1, expected["page_count"] + 1))
    assert all(p.text.strip() for p in doc.pages)


def test_pages_are_one_based_pdf_indices(paper_pdf):
    doc = parse_pdf(paper_pdf)
    assert doc.pages[0].page == 1, "page numbering must be 1-based, not 0-based"


def test_page_text_stays_on_its_own_page(paper_pdf):
    """Content unique to the references page must not leak onto page 1."""
    doc = parse_pdf(paper_pdf)
    assert "Beltagy" in doc.pages[5].text
    assert "Beltagy" not in doc.pages[0].text


def test_title_survives_a_line_break(paper_pdf, expected):
    doc = parse_pdf(paper_pdf)
    assert doc.title == expected["title"]


def test_parse_records_size_and_hash(paper_pdf):
    doc = parse_pdf(paper_pdf)
    assert doc.file_size_bytes > 0
    assert doc.sha256 and len(doc.sha256) == 64


# --- file validation -------------------------------------------------------


def test_rejects_non_pdf_extension(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(UnsupportedFileError):
        validate_pdf_file(bad, max_bytes=1024)


def test_rejects_pdf_extension_without_pdf_content(tmp_path):
    """A hostile upload can rename anything to .pdf; the magic header is checked."""
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"GIF89a this is not a pdf")
    with pytest.raises(UnsupportedFileError):
        validate_pdf_file(fake, max_bytes=1024)


def test_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(ValidationError):
        validate_pdf_file(empty, max_bytes=1024)


def test_rejects_oversized_file(paper_pdf):
    with pytest.raises(FileTooLargeError):
        validate_pdf_file(paper_pdf, max_bytes=10)


def test_rejects_missing_file(tmp_path):
    with pytest.raises(ValidationError):
        validate_pdf_file(tmp_path / "nope.pdf", max_bytes=1024)


# --- OCR decision ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_decision"),
    [
        ("", True),
        ("   ", True),
        ("short", True),
        ("a" * 179, True),
        ("word " * 100, False),
        ("1234567890!@#$%^&*()" * 20, True),  # symbol soup: a scanned-page symptom
    ],
)
def test_needs_ocr(text, expected_decision):
    assert needs_ocr(text, min_chars=180) is expected_decision


def test_needs_ocr_respects_threshold():
    text = "a" * 100
    assert needs_ocr(text, min_chars=180) is True
    assert needs_ocr(text, min_chars=50) is False


def test_ocr_disabled_yields_null_engine():
    engine = build_ocr_engine(enabled=False)
    assert isinstance(engine, NullOCREngine)
    assert engine.available is False
    assert engine.extract_page(None, page_number=1) is None


def test_ocr_engine_is_used_when_native_text_is_thin(paper_pdf):
    """A page with enough native text must not be handed to OCR."""

    class SpyEngine(NullOCREngine):
        name = "spy"
        calls: list[int] = []

        @property
        def available(self) -> bool:
            return True

        def extract_page(self, page, *, page_number: int):
            self.calls.append(page_number)
            return OCRResult(text="OCR TEXT " * 40, confidence=0.6, engine="spy")

    spy = SpyEngine()
    doc = parse_pdf(paper_pdf, ocr_engine=spy, ocr_min_chars=180)
    assert spy.calls == [], "text-rich pages should never trigger OCR"
    assert all(p.source is ExtractionMethod.NATIVE_TEXT for p in doc.pages)


def test_ocr_result_replaces_thin_native_text(paper_pdf):
    class AlwaysEngine(NullOCREngine):
        @property
        def available(self) -> bool:
            return True

        def extract_page(self, page, *, page_number: int):
            return OCRResult(text="RECOVERED " * 800, confidence=0.55, engine="fake")

    # A very high threshold forces every page down the OCR path.
    doc = parse_pdf(paper_pdf, ocr_engine=AlwaysEngine(), ocr_min_chars=100_000)
    assert all(p.source is ExtractionMethod.OCR for p in doc.pages)
    assert all(p.confidence == 0.55 for p in doc.pages)


def test_ocr_is_not_used_when_it_recovers_less_than_the_native_layer(paper_pdf):
    """OCR is a fallback, not an override: worse output must be discarded."""

    class PoorEngine(NullOCREngine):
        @property
        def available(self) -> bool:
            return True

        def extract_page(self, page, *, page_number: int):
            return OCRResult(text="blurry", confidence=0.3, engine="poor")

    doc = parse_pdf(paper_pdf, ocr_engine=PoorEngine(), ocr_min_chars=100_000)
    assert all(p.source is ExtractionMethod.NATIVE_TEXT for p in doc.pages)
    assert all("blurry" not in p.text for p in doc.pages)


def test_pages_needing_ocr_without_an_engine_lose_confidence(paper_pdf):
    doc = parse_pdf(paper_pdf, ocr_engine=None, ocr_min_chars=100_000)
    assert all(p.confidence < 1.0 for p in doc.pages)
    assert all(p.source is ExtractionMethod.NATIVE_TEXT for p in doc.pages)


# --- normalisation ---------------------------------------------------------


def test_normalize_text_repairs_ligatures_and_hyphenation():
    assert normalize_text("efﬁcient") == "efficient"
    assert normalize_text("atten-\ntion") == "attention"
    assert normalize_text("“quoted”") == '"quoted"'


def test_normalize_text_preserves_column_spacing():
    """Multiple spaces carry table structure and must survive extraction."""
    assert "   " in normalize_text("Model   76.2   61.4")


# --- structure detection ---------------------------------------------------


def _block(text: str, *, size: float = 13.0, bold: bool = True, page: int = 1) -> TextBlock:
    return TextBlock(page=page, text=text, bbox=(0, 0, 100, 20), font_size=size, is_bold=bold)


@pytest.mark.parametrize(
    "text",
    ["1 Introduction", "4.2 Main Results", "3 Method", "Abstract", "References", "Conclusion"],
)
def test_detects_headings(text):
    heading = looks_like_heading(_block(text), body_font_size=9.5)
    assert heading is not None
    assert heading.is_canonical


def test_does_not_mistake_a_sentence_for_a_heading():
    body = _block(
        "We evaluate on NarrativeQA and QuALITY, reporting test F1 for each split.",
        size=9.5,
        bold=False,
    )
    assert looks_like_heading(body, body_font_size=9.5) is None


def test_author_line_is_flagged_as_alpha_numbered():
    """"A. Researcher, B. Scientist" has the exact shape of an appendix heading."""
    heading = looks_like_heading(_block("A. Researcher, B. Scientist"), body_font_size=9.5)
    assert heading is not None and heading.is_alpha_numbered


def test_subsection_level_is_detected():
    heading = looks_like_heading(_block("4.2 Main Results"), body_font_size=9.5)
    assert heading.level == 2


@pytest.mark.parametrize(
    "text",
    [
        "L = L_read + lambda * sum_i ( f_i * P_i )        (1)",
        "s_i = ( W_q q )^T ( W_b b_i ) / sqrt(d)        (3)",
        r"\frac{1}{N} \sum_{i=1}^{N} x_i",
    ],
)
def test_detects_equations(text):
    assert looks_like_equation(text) is True


def test_prose_is_not_an_equation():
    assert looks_like_equation(
        "The model was trained for 12 epochs on a single machine with standard settings."
    ) is False


def test_detects_tabular_layout():
    table = (
        "Model        NarrativeQA F1   QuALITY Acc\n"
        "Dense-Large          76.2          61.4\n"
        "SRT-Large            81.4          68.7"
    )
    assert looks_like_table(table) is True


def test_prose_mentioning_numbers_is_not_a_table():
    assert looks_like_table(
        "We improve by 5.2 F1 on NarrativeQA while using 3.1x fewer FLOPs."
    ) is False


@pytest.mark.parametrize(
    ("text", "element_type", "element_id"),
    [
        ("Table 1: Main results", ElementType.TABLE, "table-1"),
        ("Table 3. Ablation study", ElementType.TABLE, "table-3"),
        ("Figure 2: Architecture diagram", ElementType.FIGURE, "figure-2"),
        ("Algorithm 1: Routing", ElementType.FIGURE, "algorithm-1"),
    ],
)
def test_caption_ids(text, element_type, element_id):
    assert extract_element_id(text) == (element_type, element_id)


def test_sentence_referring_to_a_table_is_not_a_caption():
    """"Table 1 reports..." is prose about a table, not the table itself."""
    element_type, element_id = extract_element_id("Table 1 reports the main comparison.")
    assert element_type is ElementType.UNKNOWN
    assert element_id is None


def test_reference_entries_are_classified_inside_the_references_section():
    block = _block("[1] Beltagy, I. Longformer. arXiv, 2020.", size=8.6, bold=False)
    element_type, _ = classify_block(block, section="References", equation_counter=1)
    assert element_type is ElementType.REFERENCE


def test_reference_like_numbers_outside_references_are_not_references():
    block = _block("[1] Beltagy, I. Longformer. arXiv, 2020.", size=8.6, bold=False)
    element_type, _ = classify_block(block, section="1 Introduction", equation_counter=1)
    assert element_type is not ElementType.REFERENCE


# --- chunking --------------------------------------------------------------


def test_chunks_carry_full_metadata(ingested):
    for chunk in ingested.chunks:
        assert chunk.document_id == "paper-test"
        assert chunk.page >= 1
        assert chunk.text.strip()
        assert chunk.chunk_id.startswith("paper-test-page-")
        assert isinstance(chunk.element_type, ElementType)
        assert 0.0 <= chunk.confidence <= 1.0


def test_no_chunk_spans_a_page_boundary(ingested):
    """The page number in a chunk id must match the chunk's page field."""
    for chunk in ingested.chunks:
        assert f"-page-{chunk.page}-" in chunk.chunk_id


def test_sections_are_detected_in_document_order(ingested, expected):
    detected = ingested.metadata.sections
    for section in expected["sections"]:
        assert section in detected, f"missing section: {section}"
    assert detected.index("1 Introduction") < detected.index("References")


def test_front_matter_is_not_treated_as_a_section(ingested):
    labels = " ".join(ingested.metadata.sections)
    assert "Researcher" not in labels, "the author line must not become a section"
    assert "Front Matter" in {c.section for c in ingested.chunks}


def test_subsections_are_preserved_for_citations(ingested):
    subsections = {c.subsection for c in ingested.chunks if c.subsection}
    assert "4.2 Main Results" in subsections
    chunk = next(c for c in ingested.chunks if c.subsection == "4.2 Main Results")
    assert chunk.citation_section == "4.2 Main Results"
    assert chunk.section == "4 Experiments"


def test_elements_are_detected_on_the_expected_pages(ingested, expected):
    by_type: dict[ElementType, set[int]] = {}
    for chunk in ingested.chunks:
        by_type.setdefault(chunk.element_type, set()).add(chunk.page)

    assert expected["table_pages"][0] in by_type[ElementType.TABLE]
    assert expected["figure_pages"][0] in by_type[ElementType.FIGURE]
    assert expected["equation_pages"][0] in by_type[ElementType.EQUATION]
    assert by_type[ElementType.REFERENCE] == {6}


def test_table_body_is_linked_to_its_caption(ingested):
    """Caption and numeric body must share one element id so both are citable."""
    table_chunks = [c for c in ingested.chunks if c.element_type is ElementType.TABLE]
    assert len(table_chunks) >= 2
    assert all(c.element_id == "table-1" for c in table_chunks)
    assert any("76.2" in c.text for c in table_chunks), "the numeric body must be indexed"


def test_equations_are_atomic_chunks(ingested):
    equations = [c for c in ingested.chunks if c.element_type is ElementType.EQUATION]
    assert len(equations) == 3
    assert all(len(c.text) < 200 for c in equations), "equations must not absorb prose"


def test_long_text_is_split_with_overlap():
    from paperpilot.ingestion.chunker import _split_long_text

    text = ". ".join(f"Sentence number {i} about routing" for i in range(200))
    pieces = _split_long_text(text, max_chars=400, overlap=80)
    assert len(pieces) > 1
    assert all(len(p) <= 400 for p in pieces)
    assert "".join(pieces).count("Sentence") >= text.count("Sentence")


def test_chunking_an_empty_document_is_safe():
    from paperpilot.ingestion.pdf_parser import ParsedDocument

    chunks, sections = chunk_document(
        ParsedDocument(pages=[], page_count=0), document_id="empty"
    )
    assert chunks == [] and sections == []
