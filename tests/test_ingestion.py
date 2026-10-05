"""PDF extraction, OCR decisions, structure detection and chunk metadata."""

from __future__ import annotations

import pytest

from paperpilot.domain.enums import ElementType, ExtractionMethod
from paperpilot.errors import FileTooLargeError, UnsupportedFileError, ValidationError
from paperpilot.ingestion.chunker import _SectionCursor, chunk_document
from paperpilot.ingestion.elements import (
    DetectedHeading,
    classify_block,
    extract_element_id,
    looks_like_equation,
    looks_like_heading,
    looks_like_table,
    split_embedded_headings,
)
from paperpilot.ingestion.ocr import NullOCREngine, OCRResult, build_ocr_engine
from paperpilot.ingestion.pdf_parser import (
    ParsedPage,
    TextBlock,
    _authors_from_layout,
    _publication_year,
    _strip_running_heads,
    extraction_failures,
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
    text = "word " * 20
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


def test_the_ocr_reading_of_a_failed_page_is_taken_as_is(paper_pdf):
    """A page that failed the check has no trustworthy text to compare against."""

    class ShortEngine(NullOCREngine):
        @property
        def available(self) -> bool:
            return True

        def extract_page(self, page, *, page_number: int):
            return OCRResult(text="Short but correct.", confidence=0.6, engine="paddleocr_vl")

    doc = parse_pdf(paper_pdf, ocr_engine=ShortEngine(), ocr_min_chars=100_000)
    assert all(p.source is ExtractionMethod.OCR for p in doc.pages)
    assert all(p.text == "Short but correct." for p in doc.pages)


def test_an_empty_ocr_reading_keeps_the_native_text_at_low_confidence(paper_pdf):
    class EmptyEngine(NullOCREngine):
        @property
        def available(self) -> bool:
            return True

        def extract_page(self, page, *, page_number: int):
            return OCRResult(text="  \n", confidence=0.6, engine="paddleocr_vl")

    native = parse_pdf(paper_pdf)
    doc = parse_pdf(paper_pdf, ocr_engine=EmptyEngine(), ocr_min_chars=100_000)
    assert all(p.source is ExtractionMethod.NATIVE_TEXT for p in doc.pages)
    assert [p.text for p in doc.pages] == [p.text for p in native.pages]
    assert all(p.confidence == 0.4 for p in doc.pages)


def test_pages_needing_ocr_without_an_engine_lose_confidence(paper_pdf):
    doc = parse_pdf(paper_pdf, ocr_engine=None, ocr_min_chars=100_000)
    assert all(p.confidence < 1.0 for p in doc.pages)
    assert all(p.source is ExtractionMethod.NATIVE_TEXT for p in doc.pages)


_PROSE = (
    "The encoder maps an input sequence of symbol representations to a sequence "
    "of continuous representations, and the decoder then generates an output "
    "sequence one element at a time, consuming the previous symbols as input. "
) * 4


def test_clean_prose_passes_the_extraction_check():
    assert extraction_failures(_PROSE) == []


def test_unmapped_glyphs_fail_the_extraction_check():
    text = " ".join(f"(cid:{i}) token" for i in range(80))
    assert any(r.startswith("broken glyphs") for r in extraction_failures(text))
    private_use = _PROSE.replace("e", "\ue001")
    assert any(r.startswith("broken glyphs") for r in extraction_failures(private_use))


def test_words_run_together_fail_the_extraction_check():
    """A text layer with no space glyphs: each line becomes one long token."""
    text = "\n".join(line.replace(" ", "") for line in _PROSE.split(", "))
    assert any(r.startswith("garbled words") for r in extraction_failures(text))


def test_mojibake_fails_the_extraction_check():
    text = " ".join(["qxkzt", "brrwt", "pfftk", "zxcvb", "the"] * 20)
    assert any(r.startswith("garbled words") for r in extraction_failures(text))


def test_acronyms_are_not_mistaken_for_garbled_words():
    text = (_PROSE + " ") + " ".join(["CNN", "LSTMs", "BLEU", "GPT", "HTTP", "NLP"] * 30)
    assert extraction_failures(text) == []


def test_a_page_covered_by_images_fails_whatever_its_text_layer_holds():
    reasons = extraction_failures(_PROSE, image_coverage=0.97)
    assert reasons and reasons[0].startswith("scanned page")
    assert extraction_failures(_PROSE, image_coverage=0.5) == []


def test_a_scanned_page_with_a_hidden_text_layer_is_read_by_ocr(paper_pdf, tmp_path):
    """Scans often carry an earlier, invisible OCR layer; it is not trusted."""
    import pymupdf

    with pymupdf.open(paper_pdf) as src:
        pixmap = src[1].get_pixmap(dpi=72)
        scan = pymupdf.open()
        page = scan.new_page(width=src[1].rect.width, height=src[1].rect.height)
    page.insert_image(page.rect, pixmap=pixmap)
    page.insert_textbox(page.rect + (40, 40, -40, -40), _PROSE * 2, render_mode=3)
    path = tmp_path / "scan.pdf"
    scan.save(path)
    scan.close()

    calls = []

    class Engine(NullOCREngine):
        @property
        def available(self) -> bool:
            return True

        def extract_page(self, page, *, page_number: int):
            calls.append(page_number)
            return OCRResult(text="Read from the image.", confidence=0.6, engine="paddleocr_vl")

    doc = parse_pdf(path, ocr_engine=Engine())
    assert calls == [1]
    assert (doc.pages[0].text, doc.pages[0].source) == ("Read from the image.", ExtractionMethod.OCR)


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


def test_a_number_and_title_on_separate_lines_is_one_heading():
    heading = looks_like_heading(_block("3.2.1\nScaled Dot-Product Attention", bold=False, size=9.5), body_font_size=9.5)
    assert heading is not None
    assert (heading.label, heading.level) == ("3.2.1 Scaled Dot-Product Attention", 3)


def test_rotated_margin_text_is_never_a_heading():
    stamp = TextBlock(
        page=1, text="arXiv:1706.03762v7  [cs.CL]  2 Aug 2023", bbox=(10, 200, 40, 600),
        font_size=20.0, rotated=True,
    )
    assert looks_like_heading(stamp, body_font_size=9.5) is None
    stamp.rotated = False
    assert looks_like_heading(stamp, body_font_size=9.5) is None  # lowercase start


def test_a_bare_table_header_word_is_not_a_section():
    assert looks_like_heading(_block("Model", bold=False, size=9.5), body_font_size=9.5) is None
    assert looks_like_heading(_block("References", bold=False, size=9.5), body_font_size=9.5)


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


ATTENTION_FIRST_PAGE = """Attention Is All You Need
Ashish Vaswani Google Brain
Abstract
Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task.
31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
arXiv:1706.03762v7 [cs.CL] 2 Aug 2023"""


@pytest.mark.parametrize(
    ("text", "year"),
    [
        (ATTENTION_FIRST_PAGE, 2017),
        ("A Title\nJane Doe\narXiv:1909.01234v2 [cs.LG] 3 Mar 2021\nAbstract\nWMT 2014", 2019),
        ("A Title\nJane Doe, 2020\nAbstract\nWe use ImageNet 2012.", 2020),
        ("A Title\nAbstract\nWe train on WMT 2014 data.", None),
    ],
)
def test_publication_year_ignores_benchmark_years(text, year):
    from paperpilot.ingestion.pdf_parser import _publication_year

    assert _publication_year(text) == year


def test_paddleocr_vl_engine_sends_the_page_image_with_the_ocr_prompt(paper_pdf):
    import json

    import httpx
    import pymupdf

    from paperpilot.ingestion.ocr import PaddleOCRVLEngine

    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "PaddleOCR-VL-1.6"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Scanned text"}}]})

    engine = PaddleOCRVLEngine(base_url="http://ocr.test/v1", transport=httpx.MockTransport(handler))
    with pymupdf.open(paper_pdf) as doc:
        assert engine.available
        result = engine.extract_page(doc[0], page_number=1)

    assert (result.text, result.engine) == ("Scanned text", "paddleocr_vl")
    content = json.loads(requests[-1].content)["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "OCR:"}


def test_paddleocr_vl_engine_degrades_when_the_server_is_down(paper_pdf):
    import httpx
    import pymupdf

    from paperpilot.ingestion.ocr import PaddleOCRVLEngine

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    engine = build_ocr_engine(enabled=True)
    assert isinstance(engine, PaddleOCRVLEngine)

    engine = PaddleOCRVLEngine(transport=httpx.MockTransport(refuse))
    assert engine.available is False
    with pymupdf.open(paper_pdf) as doc:
        assert engine.extract_page(doc[0], page_number=1) is None


# --- charts and diagrams on text pages ---------------------------------------


def _diagram_png(label: str) -> bytes:
    import pymupdf

    scratch = pymupdf.open()
    canvas = scratch.new_page(width=300, height=120)
    canvas.draw_rect(pymupdf.Rect(20, 30, 280, 90), color=(0, 0, 0), fill=(0.9, 0.9, 1))
    canvas.insert_text((40, 65), label, fontsize=12)
    return canvas.get_pixmap(dpi=100).tobytes("png")


@pytest.fixture
def figure_pdf(tmp_path):
    """One text page with two image figures, each captioned below, and a ruled table."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    assert page.insert_textbox(
        pymupdf.Rect(60, 40, 552, 120),
        "We compare the router against three baselines on long-document QA. " * 3,
        fontsize=10,
    ) >= 0, "body text must fit its box, or PyMuPDF silently writes nothing"
    page.insert_image(pymupdf.Rect(150, 130, 450, 250), stream=_diagram_png("Encoder"))
    page.insert_text((60, 270), "Figure 1: The encoder.", fontsize=9)
    page.insert_image(pymupdf.Rect(150, 290, 450, 410), stream=_diagram_png("Decoder"))
    page.insert_text((60, 430), "Figure 2: The decoder.", fontsize=9)

    table = pymupdf.Rect(80, 470, 530, 700)
    for row in range(9):
        y = table.y0 + row * 25
        page.draw_line((table.x0, y), (table.x1, y))
        for col, x in enumerate((90, 240, 390)):
            page.insert_text((x, y + 17), f"model {row} value {col}.5 score", fontsize=9)
    for x in (table.x0, 230, 380, table.x1):
        page.draw_line((x, table.y0), (x, table.y0 + 200))
    path = tmp_path / "figures.pdf"
    doc.save(path)
    return path


class _FigureEngine(NullOCREngine):
    supports_figures = True

    def __init__(self):
        self.regions = []

    @property
    def available(self) -> bool:
        return True

    def extract_figure(self, page, rect, *, page_number, printed_text=""):
        self.regions.append(rect)
        return OCRResult(text=f"Figure content {len(self.regions)}", engine="fake")


def test_figure_regions_find_images_but_not_ruled_tables(figure_pdf):
    import pymupdf

    from paperpilot.ingestion.pdf_parser import find_figure_regions

    with pymupdf.open(figure_pdf) as doc:
        regions = find_figure_regions(doc[0])

    assert [tuple(round(v) for v in r) for r in regions] == [(150, 130, 450, 250), (150, 290, 450, 410)]


def test_figures_on_text_pages_are_read_and_linked_to_their_captions(figure_pdf):
    engine = _FigureEngine()
    doc = parse_pdf(figure_pdf, ocr_engine=engine)
    chunks, _ = chunk_document(doc, document_id="doc", max_chars=1200, overlap=100)

    figures = [c for c in chunks if c.source is ExtractionMethod.OCR]
    assert [(c.text, c.element_type, c.element_id) for c in figures] == [
        ("Figure content 1", ElementType.FIGURE, "figure-1"),
        ("Figure content 2", ElementType.FIGURE, "figure-2"),
    ]
    assert doc.pages[0].source is ExtractionMethod.NATIVE_TEXT
    assert any("We compare the router" in c.text for c in chunks)


def test_figure_ocr_can_be_switched_off(figure_pdf):
    engine = _FigureEngine()
    parse_pdf(figure_pdf, ocr_engine=engine, figure_ocr=False)
    assert engine.regions == []


def test_one_caption_covers_side_by_side_figure_panels():
    from paperpilot.ingestion.pdf_parser import ParsedDocument, ParsedPage

    def figure(text, x0):
        return TextBlock(
            page=1,
            text=text,
            bbox=(x0, 100, x0 + 200, 300),
            source=ExtractionMethod.OCR,
            element_hint=ElementType.FIGURE,
            confidence=0.6,
        )

    caption = TextBlock(page=1, text="Figure 2: Two panels.", bbox=(50, 310, 500, 320), font_size=9)
    page = ParsedPage(page=1, text="", blocks=[figure("left", 50), figure("right", 300), caption])
    chunks, _ = chunk_document(ParsedDocument(pages=[page]), document_id="d", max_chars=1200, overlap=100)

    assert [(c.text, c.element_id, c.confidence) for c in chunks] == [
        ("left", "figure-2", 0.6),
        ("right", "figure-2", 0.6),
        ("Figure 2: Two panels.", "figure-2", 1.0),
    ]


def test_chart_rows_are_kept_only_when_their_numbers_are_printed():
    from paperpilot.ingestion.ocr import grounded_chart_rows

    chart = "Category | Accuracy\nLongformer | 61.2\nBigBird | 64.8\nAttention | 1\nLegend | n/a"
    rows, dropped = grounded_chart_rows(chart, printed="Longformer 61.2 BigBird 64.80")
    assert rows == ["Category | Accuracy", "Longformer | 61.2", "BigBird | 64.8"]
    assert dropped == 2

    # A diagram: every number the model reports is invented.
    assert grounded_chart_rows("Question | Router\nQuestion | 5", printed="Question Router") == ([], 1)


# --- tables re-read by the OCR engine ------------------------------------------

_TABLE_OTSL = (
    "<fcel>Model<fcel>BLEU<lcel><fcel>Cost<lcel><nl>"
    "<ucel><fcel>EN-DE<fcel>EN-FR<fcel>EN-DE<fcel>EN-FR<nl>"
    "<fcel>ByteNet [18]<fcel>23.75<ecel><ecel><ecel><nl>"
    "<fcel>Transformer (base)<fcel>27.3<fcel>38.1<fcel>\\(3.3 \\cdot 10^{18}\\)<lcel><nl>"
)


def test_otsl_tables_become_markdown_with_merged_headers_named():
    from paperpilot.ingestion.ocr import grid_to_markdown, otsl_to_grid

    grid, spanned = otsl_to_grid(_TABLE_OTSL)

    assert grid_to_markdown(grid, spanned).split("\n") == [
        "| Model | BLEU EN-DE | BLEU EN-FR | Cost EN-DE | Cost EN-FR |",
        "| --- | --- | --- | --- | --- |",
        "| ByteNet [18] | 23.75 |  |  |  |",
        # A value spanning two body columns is shown once, not as two values.
        "| Transformer (base) | 27.3 | 38.1 | 3.3 · 10^18 |  |",
    ]


def test_a_recognised_table_must_agree_with_the_text_layer():
    from paperpilot.ingestion.ocr import otsl_to_grid, verify_table

    grid, _ = otsl_to_grid(_TABLE_OTSL)
    # Superscripts flatten in the text layer ("10^18" -> "1018"); digits still agree.
    native = "Model BLEU Cost EN-DE EN-FR ByteNet [18] 23.75 Transformer (base) 27.3 38.1 3.3 · 1018"

    assert verify_table(grid, native).ok
    assert not verify_table(grid, native.replace("23.75", "23.76")).ok, "a misread digit"
    assert not verify_table(grid, "Completely different words 23.75 27.3 38.1 3.3 1018 18").ok
    assert not verify_table([["one cell"]], "one cell").ok, "prose is not a table"


@pytest.fixture
def table_pdf(tmp_path):
    """A captioned table with top/mid/bottom rules, then a captioned unruled one."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 60), "Table 1: Ruled results.", fontsize=9)
    for y in (75, 95, 160):
        page.draw_line((60, y), (400, y), width=0.8)
    page.insert_text((70, 90), "Model      Score", fontsize=9)
    for i, y in enumerate((115, 135, 150)):
        page.insert_text((70, y), f"System {i}      {70 + i}.5", fontsize=9)

    page.insert_text((60, 220), "Table 2: Unruled results.", fontsize=9)
    page.insert_textbox(
        pymupdf.Rect(60, 232, 400, 280), "Model   Score\nAlpha   61.2\nBeta    64.8", fontsize=9
    )
    assert page.insert_textbox(
        pymupdf.Rect(60, 320, 552, 400), "Discussion of the results follows here.", fontsize=10
    ) >= 0
    path = tmp_path / "tables.pdf"
    doc.save(path)
    return path


class _TableEngine(NullOCREngine):
    supports_tables = True

    def __init__(self, accept=True):
        self.accept = accept
        self.regions = []

    @property
    def available(self) -> bool:
        return True

    def extract_table(self, page, rect, *, page_number, printed_text=""):
        self.regions.append((rect, printed_text))
        if not self.accept:
            return None
        return OCRResult(text=f"| Model | Score |\n| --- | --- |\n| T{len(self.regions)} | 1 |", confidence=0.9)


def test_table_regions_follow_captions_with_or_without_rules(table_pdf):
    import pymupdf

    from paperpilot.ingestion.pdf_parser import find_table_regions, parse_pdf

    doc = parse_pdf(table_pdf)
    with pymupdf.open(table_pdf) as pdf:
        regions = find_table_regions(pdf[0], doc.pages[0].blocks)

    ruled, unruled = regions
    assert (round(ruled.y0), round(ruled.y1)) == (72, 163)
    assert 225 < unruled.y0 < 240 and unruled.y1 < 300, "must stop before the prose below"


def test_verified_tables_replace_native_text_and_keep_their_caption_ids(table_pdf):
    doc = parse_pdf(table_pdf, ocr_engine=_TableEngine())
    chunks, _ = chunk_document(doc, document_id="doc", max_chars=1200, overlap=100)

    tables = [c for c in chunks if c.source is ExtractionMethod.OCR]
    assert [(c.element_type, c.element_id, c.confidence) for c in tables] == [
        (ElementType.TABLE, "table-1", 0.9),
        (ElementType.TABLE, "table-2", 0.9),
    ]
    assert not any("System 0" in c.text for c in chunks), "native cells are replaced"
    assert any("Discussion of the results" in c.text for c in chunks)


def test_a_rejected_table_keeps_its_native_text(table_pdf):
    engine = _TableEngine(accept=False)
    doc = parse_pdf(table_pdf, ocr_engine=engine)
    chunks, _ = chunk_document(doc, document_id="doc", max_chars=1200, overlap=100)

    assert len(engine.regions) == 2
    assert "70.5" in engine.regions[0][1], "the engine is given the native text to verify against"
    assert any("System 0" in c.text for c in chunks)
    assert not any(c.source is ExtractionMethod.OCR for c in chunks)


def test_table_ocr_can_be_switched_off(table_pdf):
    engine = _TableEngine()
    parse_pdf(table_pdf, ocr_engine=engine, table_ocr=False)
    assert engine.regions == []


def test_long_tables_split_between_rows_with_the_header_repeated():
    from paperpilot.ingestion.chunker import _split_markdown_table

    table = "\n".join(["| Model | Score |", "| --- | --- |"] + [f"| model-{i} | {i}.0 |" for i in range(40)])
    pieces = _split_markdown_table(table, max_chars=200)

    assert len(pieces) > 1
    assert all(p.startswith("| Model | Score |\n| --- | --- |\n| model-") for p in pieces)
    assert sum(p.count("| model-") for p in pieces) == 40


# --- display equations split across layout blocks ---------------------------


def _row(text: str, x0: float, y0: float, x1: float, y1: float) -> TextBlock:
    return TextBlock(page=4, text=text, bbox=(x0, y0, x1, y1), font_size=10.9)


def _bm25_page_blocks() -> list[TextBlock]:
    """Blocks exactly as PyMuPDF returns BM25 (Eq. 2) of a real paper, page 4."""
    return [
        _row("Sparse Retrieval.\nIn parallel, a BM25 scorer computes term-frequency-based relevance:", 72, 672, 467, 683),
        _row("BM25(q, d) =", 176, 698, 240, 710),
        _row("X", 243, 696, 258, 707),
        _row("t∈q", 244, 713, 257, 722),
        _row("IDF(t) ·\ntf(t, d)(k1 + 1)", 260, 691, 401, 710),
        _row("tf(t, d) + k1", 299, 710, 352, 722),
        _row("\x10", 353, 706, 359, 717),
        _row("1 -b + b ·\n|d|\navgdl", 359, 707, 427, 724),
        _row("\x11\n(2)", 428, 699, 540, 717),
        _row("4", 303, 742, 309, 753),
    ]


def test_an_equation_split_into_side_by_side_blocks_is_reassembled():
    from paperpilot.ingestion.elements import assemble_display_equations

    blocks = assemble_display_equations(_bm25_page_blocks())

    assert [b.text for b in blocks][0].startswith("Sparse Retrieval.")
    equation = blocks[1]
    assert equation.element_hint is ElementType.EQUATION
    assert equation.text.startswith("BM25(q, d) =") and equation.text.endswith("avgdl\n(2)")
    assert "\x10" not in equation.text and "\x11" not in equation.text
    assert blocks[2].text == "4", "the page number below the row is not part of the equation"
    assert classify_block(equation, section="3", equation_counter=1) == (
        ElementType.EQUATION,
        "equation-2",
    )


def test_a_reassembled_equation_becomes_one_numbered_chunk():
    from paperpilot.ingestion.pdf_parser import ParsedDocument, ParsedPage

    page = ParsedPage(page=4, text="", blocks=_bm25_page_blocks())
    chunks, _ = chunk_document(ParsedDocument(pages=[page]), document_id="d", max_chars=1200, overlap=100)
    equations = [c for c in chunks if c.element_type is ElementType.EQUATION]

    assert [(c.element_id, c.text.count("\n")) for c in equations] == [("equation-2", 9)]


def test_short_prose_blocks_on_one_row_are_not_merged_into_an_equation():
    from paperpilot.ingestion.elements import assemble_display_equations

    blocks = [_row("Model", 72, 100, 110, 110), _row("Accuracy (%)", 200, 100, 260, 110)]
    assert assemble_display_equations(blocks) == blocks


def test_norm_bars_mark_a_numbered_block_as_an_equation():
    assert looks_like_equation("sdense(q, d) =\neq · ed\n∥eq∥∥ed∥\n(1)")


def test_a_run_of_table_numbers_is_not_an_equation():
    assert not looks_like_equation("GNMT + RL [38]\n24.6\n39.92\n2.3 · 1019\n1.4 · 1020")


def test_a_table_of_words_after_its_caption_keeps_the_caption_id():
    from paperpilot.ingestion.pdf_parser import ParsedDocument, ParsedPage

    blocks = [
        TextBlock(page=8, text="Table 1: Proposed ablation configurations.", bbox=(220, 258, 391, 268), font_size=10),
        TextBlock(page=8, text="Configuration\nDescription", bbox=(110, 285, 304, 296), font_size=10.9),
        TextBlock(
            page=8,
            text="Full System\nComplete pipeline (text + image)\nDense Only\nSemantic search without BM25",
            bbox=(110, 304, 502, 382),
            font_size=10.9,
        ),
        TextBlock(
            page=8,
            text="We compare against a naive retrieval baseline that uses fixed-size chunks and no validation.",
            bbox=(72, 420, 540, 440),
            font_size=10.9,
        ),
    ]
    chunks, _ = chunk_document(
        ParsedDocument(pages=[ParsedPage(page=8, text="", blocks=blocks)]),
        document_id="d",
        max_chars=1200,
        overlap=100,
    )

    assert [(c.element_type, c.element_id) for c in chunks] == [
        (ElementType.TABLE, "table-1"),
        (ElementType.TABLE, "table-1"),
        (ElementType.PARAGRAPH, None),
    ]
    assert chunks[1].text.startswith("Configuration\nDescription\nFull System")


# --- structure recognition in single-block journal layouts -------------------


def _column_block(lines: list[tuple[str, float]], *, left: float = 56.0) -> TextBlock:
    """A text column whose lines carry real right-edge geometry.

    ``lines`` pairs each line's text with its right edge, which is what tells a
    wrapped line (flush to the column edge) from one that ends a paragraph.
    """
    geometry = [
        (text, (left, 100.0 + position * 12.0, right, 110.0 + position * 12.0))
        for position, (text, right) in enumerate(lines)
    ]
    return TextBlock(
        page=1,
        text="\n".join(text for text, _ in lines),
        bbox=(left, 100.0, max(right for _, right in lines), 110.0 + 12.0 * len(lines)),
        font_size=12.0,
        is_bold=True,
        lines=geometry,
    )


def test_a_heading_glued_to_its_paragraph_is_split_back_out():
    block = _column_block(
        [
            ("ChatGPT.", 370.0),
            ("2 Number of the ChatGPT related", 541.5),
            ("publications", 384.7),
            ("2.1 Publications on Scopus", 458.1),
            ("A search of the term ChatGPT among Scopus", 541.6),
            ("materials found 59 materials published in the", 541.5),
        ]
    )
    pieces = split_embedded_headings([block])

    headings = [p.text for p in pieces if p.is_heading]
    assert headings == [
        "2 Number of the ChatGPT related\npublications",
        "2.1 Publications on Scopus",
    ]
    # The wrapped heading is read whole, not truncated at the line break.
    detected = looks_like_heading(pieces[1], body_font_size=12.0)
    assert detected is not None
    assert detected.label == "2 Number of the ChatGPT related publications"


def test_justified_word_spacing_does_not_invent_headings():
    """Some PDFs report one justified line as several one-word lines."""
    block = _column_block(
        [
            ("students' evaluation", 541.5),
            ("method", 300.0),
            ("and", 340.0),
            ("the", 380.0),
            ("assignment of homeworks was considered.", 480.0),
        ]
    )
    block.lines = [
        (text, (left, bbox[1], bbox[2], bbox[3]))
        for (text, bbox), left in zip(
            block.lines, [56.0, 180.0, 260.0, 300.0, 56.0], strict=True
        )
    ]
    assert [p.is_heading for p in split_embedded_headings([block])] == [False]


def test_a_heading_filling_the_column_is_still_a_heading():
    block = _column_block(
        [
            ("ChatGPT.", 342.0),
            ("3.1 ChatGPT and Education environment", 507.9),
            ("ChatGPT has influenced the educational field,", 513.1),
        ]
    )
    pieces = split_embedded_headings([block])
    assert [p.text for p in pieces if p.is_heading] == [
        "3.1 ChatGPT and Education environment"
    ]


def test_repeated_page_furniture_is_not_a_section():
    """A running head is heading-shaped on every page; repetition gives it away."""
    pages = [
        ParsedPage(
            page=number,
            text="body",
            blocks=[
                TextBlock(
                    page=number,
                    text=f"{number} Informatica Economica vol. 27, no. 1/2023",
                    bbox=(56.0, 20.0, 520.0, 32.0),
                    font_size=9.96,
                    is_bold=True,
                ),
                TextBlock(
                    page=number,
                    text="Body text that differs on every page " + "x" * number,
                    bbox=(56.0, 100.0, 520.0, 600.0),
                    font_size=12.0,
                ),
            ],
            source=ExtractionMethod.NATIVE_TEXT,
            confidence=1.0,
            width=595.0,
            height=842.0,
        )
        for number in range(1, 7)
    ]
    assert _strip_running_heads(pages) == 6
    assert all(len(page.blocks) == 1 for page in pages)
    assert all("Informatica" not in page.blocks[0].text for page in pages)


def test_an_unnumbered_heading_cannot_open_a_section_in_a_numbered_paper():
    """Chart titles and table column headers are set apart like headings."""
    cursor = _SectionCursor()
    assert cursor.apply(
        DetectedHeading(1, "1", "Introduction", "1 Introduction", 1, 0, is_canonical=True)
    )
    assert not cursor.apply(DetectedHeading(2, None, "Publisher", "Publisher", 1, 3))
    assert cursor.section == "1 Introduction"
    # A canonical name still opens one: papers leave "References" unnumbered.
    assert cursor.apply(
        DetectedHeading(9, None, "References", "References", 1, 0, is_canonical=True)
    )


# --- front-matter metadata ---------------------------------------------------


def _front_matter_page(blocks: list[TextBlock], text: str) -> ParsedPage:
    return ParsedPage(
        page=1,
        text=text,
        blocks=blocks,
        source=ExtractionMethod.NATIVE_TEXT,
        confidence=1.0,
        width=595.0,
        height=842.0,
    )


def test_every_author_on_the_byline_is_read_not_just_the_embedded_one():
    page = _front_matter_page(
        [
            TextBlock(page=1, text="ChatGPT - A Systematic Review", bbox=(100, 60, 500, 80), font_size=14.04),
            TextBlock(
                page=1,
                text="Alin ZAMFIROIU^1,2, Denisa VASILE^1, Daniel SAVU^2",
                bbox=(100, 90, 500, 104),
                font_size=10.87,
            ),
            TextBlock(page=1, text="1Bucharest University of Economic Studies", bbox=(100, 110, 500, 122), font_size=10.42),
        ],
        "ChatGPT - A Systematic Review\n",
    )
    assert _authors_from_layout(page) == ["Alin Zamfiroiu", "Denisa Vasile", "Daniel Savu"]


def test_authors_set_one_per_line_are_read_as_separate_authors():
    page = _front_matter_page(
        [
            TextBlock(page=1, text="A Multimodal MedCoPilot", bbox=(100, 60, 500, 80), font_size=17.22),
            TextBlock(page=1, text="Saurabh Kadam^1,*\nPrem Patil^1", bbox=(200, 90, 400, 115), font_size=9.96),
        ],
        "A Multimodal MedCoPilot\n",
    )
    assert _authors_from_layout(page) == ["Saurabh Kadam", "Prem Patil"]


def test_a_wrapped_title_line_is_not_mistaken_for_an_author_list():
    page = _front_matter_page(
        [
            TextBlock(page=1, text="A Multimodal MedCoPilot", bbox=(100, 60, 500, 80), font_size=17.22),
            TextBlock(page=1, text="Large Language and Vision Models", bbox=(100, 85, 500, 105), font_size=17.22),
        ],
        "A Multimodal MedCoPilot\n",
    )
    assert _authors_from_layout(page) == []


def test_a_year_in_body_prose_is_not_read_as_the_publication_year():
    """"published ... <year>" is an ordinary English sentence, not a venue line."""
    first_page = (
        "Informatica Economica vol. 27, no. 1/2023\n"
        "ChatGPT - A Systematic Review of Published Research Papers\n"
        "Introduction\n"
        "A search of the term ChatGPT among Scopus materials found 59 materials. "
        "Of these, 2 materials were published in 2022 (December).\n"
    )
    assert _publication_year(first_page) == 2023
