"""Layout analysis: the interface, the region algebra, and determinism.

These tests never import paddle. The engine is exercised through the interface
and through the pure functions that turn regions into blocks, so the suite runs
on a machine that has never installed the optional dependency. The one test that
does need PP-Structure is skipped unless it is present.
"""

from __future__ import annotations

import json

import pytest

from paperpilot.ingestion.elements import apply_page_layout, looks_like_heading
from paperpilot.ingestion.layout import (
    COORD_PRECISION,
    LayoutRegion,
    NullLayoutEngine,
    PageLayout,
    PPStructureLayoutEngine,
    _region_sort_key,
    build_layout_engine,
)
from paperpilot.ingestion.pdf_parser import TextBlock


def _region(kind: str, y0: float, y1: float, *, x0: float = 50.0, x1: float = 300.0, order: int = 0):
    return LayoutRegion(kind=kind, bbox=(x0, y0, x1, y1), score=0.9, order=order)


def _block(lines: list[tuple[str, float]]) -> TextBlock:
    """A block whose lines are placed at the given vertical positions."""
    geometry = [(text, (60.0, y, 280.0, y + 10.0)) for text, y in lines]
    return TextBlock(
        page=1,
        text="\n".join(text for text, _ in lines),
        bbox=(60.0, lines[0][1], 280.0, lines[-1][1] + 10.0),
        font_size=12.0,
        is_bold=False,
        lines=geometry,
    )


# --- the interface -----------------------------------------------------------


def test_layout_disabled_yields_a_null_engine():
    engine = build_layout_engine(enabled=False)
    assert isinstance(engine, NullLayoutEngine)
    assert engine.available is False
    assert engine.analyze_page(None, page_number=1) is None


def test_an_unknown_engine_name_degrades_to_null_rather_than_raising():
    assert isinstance(build_layout_engine(enabled=True, engine="not-an-engine"), NullLayoutEngine)


# --- regions into blocks -----------------------------------------------------


def test_a_heading_region_splits_the_paragraph_that_swallowed_it():
    block = _block(
        [
            ("2 Number of the ChatGPT related publications", 100.0),
            ("A search of the term ChatGPT among Scopus", 120.0),
            ("materials found 59 materials published in the", 132.0),
        ]
    )
    layout = PageLayout(
        page=1,
        regions=[
            _region("title", 95.0, 115.0, order=0),
            _region("text", 118.0, 150.0, order=1),
        ],
    )
    pieces = apply_page_layout([block], layout)

    assert [p.is_heading for p in pieces] == [True, False]
    assert pieces[0].text == "2 Number of the ChatGPT related publications"
    detected = looks_like_heading(pieces[0], body_font_size=12.0)
    assert detected is not None and detected.label == (
        "2 Number of the ChatGPT related publications"
    )


def test_lines_inside_a_running_header_are_dropped():
    block = _block(
        [
            ("6 Informatica Economica vol. 27, no. 1/2023", 20.0),
            ("Body text that belongs to the document.", 120.0),
        ]
    )
    layout = PageLayout(
        page=1,
        regions=[_region("header", 10.0, 40.0, order=0), _region("text", 110.0, 160.0, order=1)],
    )
    pieces = apply_page_layout([block], layout)
    assert [p.text for p in pieces] == ["Body text that belongs to the document."]


def test_a_block_the_model_says_nothing_about_is_returned_untouched():
    """A page the model failed on must degrade, never lose text."""
    block = _block([("Body text nowhere near a region.", 500.0)])
    layout = PageLayout(page=1, regions=[_region("text", 10.0, 40.0)])
    assert apply_page_layout([block], layout) == [block]
    assert apply_page_layout([block], PageLayout(page=1, regions=[])) == [block]


def test_overlapping_regions_are_resolved_by_coverage_then_reading_order():
    """Two regions claim the same line; the answer must not depend on order."""
    block = _block([("Heading line", 100.0)])
    wide = LayoutRegion(kind="text", bbox=(0.0, 0.0, 600.0, 800.0), score=0.9, order=1)
    tight = LayoutRegion(kind="title", bbox=(55.0, 95.0, 290.0, 115.0), score=0.9, order=0)

    forward = apply_page_layout([block], PageLayout(page=1, regions=[wide, tight]))
    reverse = apply_page_layout([block], PageLayout(page=1, regions=[tight, wide]))
    assert [p.is_heading for p in forward] == [p.is_heading for p in reverse] == [True]


# --- determinism -------------------------------------------------------------


def test_regions_sort_into_a_total_order_independent_of_detection_order():
    regions = [
        _region("text", 300.0, 340.0, x0=320.0, x1=560.0),
        _region("title", 100.0, 120.0),
        _region("text", 300.0, 340.0, x0=50.0, x1=300.0),
        _region("figure", 100.0, 120.0, x0=320.0, x1=560.0),
    ]
    first = sorted(regions, key=_region_sort_key)
    second = sorted(reversed(regions), key=_region_sort_key)
    assert [r.bbox for r in first] == [r.bbox for r in second]
    # Top-to-bottom, then left-to-right.
    assert [r.bbox[1] for r in first] == [100.0, 100.0, 300.0, 300.0]
    assert [r.bbox[0] for r in first] == [50.0, 320.0, 50.0, 320.0]


def test_the_fingerprint_covers_every_setting_that_changes_the_output():
    base = PPStructureLayoutEngine()
    assert base.fingerprint() == PPStructureLayoutEngine().fingerprint()
    for changed in (
        PPStructureLayoutEngine(dpi=300),
        PPStructureLayoutEngine(layout_model="PP-DocLayout-S"),
        PPStructureLayoutEngine(min_score=0.7),
        PPStructureLayoutEngine(lang="ch"),
        PPStructureLayoutEngine(device="gpu"),
    ):
        assert changed.fingerprint() != base.fingerprint()


def test_the_cache_replays_exactly_what_was_written(tmp_path):
    engine = PPStructureLayoutEngine(cache_dir=tmp_path)
    regions = [
        LayoutRegion(kind="title", bbox=(1.0, 2.0, 3.0, 4.0), score=0.91, order=0),
        LayoutRegion(kind="text", bbox=(1.0, 5.0, 3.0, 9.0), score=0.88, order=1),
    ]
    engine._cache_write(b"page-image-bytes", regions)
    assert engine._cache_read(b"page-image-bytes") == regions
    # A different page, and the same page under a different configuration, miss.
    assert engine._cache_read(b"other-image") is None
    assert PPStructureLayoutEngine(dpi=300, cache_dir=tmp_path)._cache_read(b"page-image-bytes") is None


def test_a_corrupt_cache_entry_is_ignored_rather_than_raising(tmp_path):
    engine = PPStructureLayoutEngine(cache_dir=tmp_path)
    path = engine._cache_path(b"page")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert engine._cache_read(b"page") is None


def test_cached_coordinates_keep_the_precision_they_were_written_with(tmp_path):
    """A float that differs in its last bit must not reorder two regions."""
    engine = PPStructureLayoutEngine(cache_dir=tmp_path)
    engine._cache_write(
        b"page", [LayoutRegion(kind="text", bbox=(1.23456789, 2.0, 3.0, 4.0), score=0.5)]
    )
    restored = engine._cache_read(b"page")
    assert restored[0].bbox[0] == round(1.23456789, COORD_PRECISION)


def test_the_cache_file_is_written_whole(tmp_path):
    """A killed run must not leave a half-written entry a later run reads back."""
    engine = PPStructureLayoutEngine(cache_dir=tmp_path)
    engine._cache_write(b"page", [LayoutRegion(kind="text", bbox=(0.0, 0.0, 1.0, 1.0))])
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(written[0].read_text())["fingerprint"] == engine.fingerprint()


# --- the real model ----------------------------------------------------------


@pytest.mark.slow
def test_pp_structure_returns_the_same_regions_on_every_run(paper_pdf):
    """The guarantee that matters: same PDF in, same regions out."""
    pytest.importorskip("paddleocr", reason="PP-Structure is an optional dependency")
    import pymupdf

    document = pymupdf.open(paper_pdf)
    page = document.load_page(0)

    runs = []
    for _ in range(2):
        engine = PPStructureLayoutEngine(cache_dir=None)  # no cache: real inference twice
        if not engine.available:
            pytest.skip("PP-Structure models are not available on this machine")
        layout = engine.analyze_page(page, page_number=1)
        runs.append([(r.kind, r.bbox, r.score, r.order) for r in layout.regions])

    assert runs[0] == runs[1]
    assert [r[3] for r in runs[0]] == list(range(len(runs[0])))


def test_a_tightly_boxed_folio_is_still_recognised_as_page_furniture():
    """Detectors box glyphs tightly; PyMuPDF line boxes carry the full leading.

    The area overlap of a one-digit folio against its footer region falls just
    under the threshold, so centre containment is what catches it.
    """
    block = TextBlock(
        page=2,
        text="6\nInformatica Economica vol. 27, no. 1/2023",
        bbox=(56.64, 48.73, 512.83, 59.75),
        font_size=9.96,
        is_bold=True,
        lines=[
            ("6", (56.64, 48.73, 64.19, 59.75)),
            ("Informatica Economica vol. 27, no. 1/2023", (337.51, 48.73, 512.83, 59.75)),
        ],
    )
    layout = PageLayout(
        page=2,
        regions=[
            LayoutRegion(kind="header", bbox=(335.4, 48.66, 508.98, 58.37), order=0),
            LayoutRegion(kind="footer", bbox=(55.47, 49.83, 61.59, 57.71), order=1),
        ],
    )
    assert apply_page_layout([block], layout) == []


def test_a_figure_caption_is_never_read_as_a_section_heading():
    """Short, set apart, mostly capitalised — a caption passes every heading test."""
    caption = TextBlock(
        page=2,
        text="Fig. 1. Authors on Scopus",
        bbox=(218.0, 357.0, 347.0, 370.0),
        font_size=12.0,
        is_bold=True,
    )
    assert looks_like_heading(caption, body_font_size=12.0) is None

    table = TextBlock(
        page=4,
        text="Table 1. Number of citations per paper",
        bbox=(100.0, 100.0, 400.0, 113.0),
        font_size=12.0,
        is_bold=True,
    )
    assert looks_like_heading(table, body_font_size=12.0) is None
