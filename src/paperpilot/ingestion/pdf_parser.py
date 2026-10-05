"""PDF text extraction with page boundaries preserved.

Page numbering contract
-----------------------
``page`` is always the **1-based PDF page index**. When the document declares
page labels (roman-numeral front matter, offset printed numbers) the label is
carried alongside as ``printed_label`` — it is never substituted for the index,
because every downstream citation and every deterministic range check is defined
against the PDF index.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from ..domain.enums import ElementType, ExtractionMethod
from ..domain.models import PageText
from ..errors import IngestionError
from ..observability import debug, info, warning
from .layout import LayoutEngine, PageLayout

PDF_MAGIC = b"%PDF-"

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}


@dataclass(slots=True)
class TextBlock:
    """One layout block from PyMuPDF, with the geometry agents may cite."""

    page: int
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float = 0.0
    is_bold: bool = False
    block_no: int = 0
    source: ExtractionMethod = ExtractionMethod.NATIVE_TEXT
    #: Set when the element type is known from how the block was produced (a
    #: figure read by OCR), so text-shape heuristics must not reclassify it.
    element_hint: ElementType | None = None
    #: Overrides the page confidence for this block's chunks.
    confidence: float | None = None
    #: Text not running left-to-right, e.g. the arXiv identifier stamped up the
    #: page margin. Never a heading, however large its font.
    rotated: bool = False
    #: Each text line with its bounding box, for splitting display equations
    #: that the PDF laid out inside a paragraph block.
    lines: list[tuple[str, tuple[float, float, float, float]]] = field(default_factory=list)
    #: Set when layout analysis cut this block out of a larger one because the
    #: line reads as a section heading. Many PDFs emit the heading and the
    #: paragraph beneath it as a single block, so the heading carries none of
    #: the typographic emphasis of its own; this records what layout established.
    is_heading: bool = False


@dataclass(slots=True)
class ParsedPage:
    page: int
    text: str
    blocks: list[TextBlock] = field(default_factory=list)
    source: ExtractionMethod = ExtractionMethod.NATIVE_TEXT
    confidence: float = 1.0
    printed_label: str | None = None
    width: float = 0.0
    height: float = 0.0
    #: Regions a layout model found on this page, when one is configured. The
    #: chunker re-cuts the blocks along them; ``None`` means the geometric
    #: heuristics are on their own.
    layout: PageLayout | None = None
    #: Why the native text layer failed the extraction check (empty: it passed).
    extraction_failures: list[str] = field(default_factory=list)

    def to_model(self) -> PageText:
        return PageText(
            page=self.page,
            text=self.text,
            source=self.source,
            confidence=self.confidence,
            printed_label=self.printed_label,
            char_count=len(self.text),
            width=self.width or None,
            height=self.height or None,
        )


@dataclass(slots=True)
class ParsedDocument:
    pages: list[ParsedPage]
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    doi: str | None = None
    year: int | None = None
    page_count: int = 0
    file_size_bytes: int = 0
    sha256: str | None = None
    filename: str | None = None

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.source == ExtractionMethod.OCR)

    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


def normalize_text(text: str) -> str:
    """Normalise ligatures, quotes and hyphenated line breaks.

    Runs of spaces are deliberately **preserved**: in extracted PDF text they are
    how column alignment survives, and the table detector depends on them. The
    aggressive whitespace collapse needed for quote comparison lives in
    :mod:`paperpilot.validation.quotes`, so indexed text and agent quotes are
    still compared like with like.
    """
    if not text:
        return ""
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    # Join words split across a line break: "atten-\ntion" -> "attention".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t", "    ", text)
    text = re.sub(r"[ ]{2,}", lambda m: " " * min(len(m.group(0)), 8), text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_pdf_file(path: Path, *, max_bytes: int) -> int:
    """Deterministic file checks run before a single byte is parsed.

    Returns the file size. Raises :class:`ValidationError` subclasses on a bad
    extension, a bad magic header, an empty file, or an oversized upload.
    """
    from ..errors import FileTooLargeError, UnsupportedFileError, ValidationError

    if not path.exists() or not path.is_file():
        raise ValidationError(f"file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise UnsupportedFileError(
            f"expected a .pdf file, got '{path.suffix or 'no extension'}'",
            details={"filename": path.name},
        )
    size = path.stat().st_size
    if size == 0:
        raise ValidationError("uploaded file is empty", details={"filename": path.name})
    if size > max_bytes:
        raise FileTooLargeError(
            f"file is {size / 1024 / 1024:.1f} MB; limit is {max_bytes / 1024 / 1024:.1f} MB",
            details={"size_bytes": size, "max_bytes": max_bytes},
        )
    with path.open("rb") as handle:
        header = handle.read(len(PDF_MAGIC))
    if header != PDF_MAGIC:
        raise UnsupportedFileError(
            "file content is not a PDF (missing %PDF- header)",
            details={"filename": path.name},
        )
    return size


def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


#: PyMuPDF span flag set on text raised above the baseline.
_SUPERSCRIPT_FLAG = 1
#: A superscript is set smaller than the text it follows.
_SUPERSCRIPT_SCALE = 0.9
#: A bare folio ("7", "xii") inside this fraction of the page height, top or
#: bottom, is a running page number, not content.
_FOLIO_MARGIN = 0.1
_FOLIO = re.compile(r"^\s*(?:\d{1,4}|[ivxlcdm]{1,7})\s*$", re.IGNORECASE)


def _span_text(spans: list[dict]) -> str:
    """Join a line's spans, marking superscripts with ``^``.

    "10" followed by a raised "20" is 10^20, not the number 1020; the marker is
    only added where the raised text is attached to the text before it, so a
    footnote mark after a space stays as written.
    """
    parts: list[str] = []
    previous_raised = False
    previous_size = 0.0
    for span in spans:
        text = span.get("text", "")
        size = float(span.get("size") or 0.0)
        # A fraction's numerator is raised too, but set at full size.
        smaller = not previous_size or not size or size < previous_size * _SUPERSCRIPT_SCALE
        raised = (
            bool(int(span.get("flags", 0)) & _SUPERSCRIPT_FLAG) and bool(text.strip()) and smaller
        )
        attached = bool(parts) and parts[-1][-1:].strip() != "" and text[:1].strip() != ""
        if raised and not previous_raised and attached:
            parts.append("^")
        parts.append(text)
        previous_raised = raised
        if not raised and text.strip():
            previous_size = size
    return "".join(parts)


def _is_folio(block: dict, text: str, page_height: float) -> bool:
    if not page_height or not _FOLIO.match(text):
        return False
    _, y0, _, y1 = block.get("bbox", (0, 0, 0, 0))
    return y0 >= page_height * (1 - _FOLIO_MARGIN) or y1 <= page_height * _FOLIO_MARGIN


#: A block lying within this fraction of the page height, top or bottom, is in
#: the margin where running heads and feet are set.
_MARGIN_BAND = 0.12
#: Repetition below this many pages is coincidence, not a running head.
_RUNNING_HEAD_MIN_PAGES = 3
_RUNNING_HEAD_MIN_FRACTION = 0.25
_DIGIT_RUN = re.compile(r"\d+")


def _running_head_key(block: TextBlock, page_height: float) -> str | None:
    """Signature of a margin block, with the folio masked out.

    The page number is the only part of a running head that changes, so masking
    digits collapses "6 Informatica Economica vol. 27, no. 1/2023" and its
    counterpart on page 8 onto the same signature. Top and bottom margins are
    kept apart so a header never cancels out an unrelated footer.
    """
    if not page_height:
        return None
    top = block.bbox[3] <= page_height * _MARGIN_BAND
    bottom = block.bbox[1] >= page_height * (1 - _MARGIN_BAND)
    if top == bottom:  # neither margin, or a block tall enough to span both
        return None
    masked = _DIGIT_RUN.sub("#", " ".join(block.text.split())).casefold()
    if len(masked) < 4:
        return None
    return f"{'top' if top else 'bottom'}:{masked}"


def _strip_running_heads(pages: list[ParsedPage]) -> int:
    """Remove repeated page headers and footers from every page's blocks.

    A running head is heading-*shaped* — short, often bold, set apart at the top
    of the page — so it beats the real section titles at every typographic test,
    and on a journal template whose headings are glued to the paragraph below
    them it can be the only heading-shaped block on the page. Repetition is what
    separates the two: body content does not recur verbatim in the same margin
    page after page, and a title that does is furniture, not a section.
    """
    if len(pages) < _RUNNING_HEAD_MIN_PAGES:
        return 0

    keys: list[dict[int, str]] = []
    counts: Counter[str] = Counter()
    for page in pages:
        found = {
            position: key
            for position, block in enumerate(page.blocks)
            if (key := _running_head_key(block, page.height))
        }
        keys.append(found)
        counts.update(set(found.values()))

    threshold = max(_RUNNING_HEAD_MIN_PAGES, round(len(pages) * _RUNNING_HEAD_MIN_FRACTION))
    repeated = {key for key, seen in counts.items() if seen >= threshold}
    if not repeated:
        return 0

    removed = 0
    for page, found in zip(pages, keys, strict=True):
        drop = {position for position, key in found.items() if key in repeated}
        if drop:
            page.blocks = [b for i, b in enumerate(page.blocks) if i not in drop]
            removed += len(drop)
    if removed:
        info("ingestion.running_heads_removed", blocks=removed, distinct=len(repeated))
    return removed


def _extract_blocks(page: pymupdf.Page, page_number: int) -> list[TextBlock]:
    """Pull layout blocks with font size, used later for heading detection."""
    blocks: list[TextBlock] = []
    try:
        raw = page.get_text("dict")
    except Exception as exc:  # noqa: BLE001 - a damaged page must not kill the run
        debug("pdf.block_extraction_failed", page=page_number, error=str(exc)[:200])
        return blocks

    page_height = float(page.rect.height)
    for index, block in enumerate(raw.get("blocks", [])):
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        lines: list[str] = []
        geometry: list[tuple[str, tuple[float, float, float, float]]] = []
        sizes: list[float] = []
        bold = False
        rotated = False
        for line in block.get("lines", []):
            direction = line.get("dir") or (1.0, 0.0)
            if abs(direction[0]) < 0.9:
                rotated = True
            spans = line.get("spans", [])
            line_text = _span_text(spans)
            if line_text.strip():
                lines.append(line_text)
                geometry.append(
                    (normalize_text(line_text), tuple(float(v) for v in line["bbox"]))  # type: ignore[misc]
                )
            for span in spans:
                if span.get("size"):
                    sizes.append(float(span["size"]))
                font = (span.get("font") or "").lower()
                if "bold" in font or "black" in font or "semibold" in font:
                    bold = True
        text = normalize_text("\n".join(lines))
        if not text or _is_folio(block, text, page_height):
            continue
        bbox = tuple(float(v) for v in block.get("bbox", (0, 0, 0, 0)))  # type: ignore[assignment]
        blocks.append(
            TextBlock(
                page=page_number,
                text=text,
                bbox=bbox,  # type: ignore[arg-type]
                font_size=round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
                is_bold=bold,
                block_no=index,
                rotated=rotated,
                lines=geometry,
            )
        )
    return blocks


#: A figure region smaller than this on either side is an icon, logo or rule.
_FIGURE_MIN_SIDE = 50.0
#: ... or smaller than this fraction of the page.
_FIGURE_MIN_PAGE_FRACTION = 0.015
#: A region covering most of the page is a background or a scanned-page image
#: behind a text layer, not a figure.
_FIGURE_MAX_PAGE_FRACTION = 0.85
#: Vector regions this text-dense are ruled tables, which PyMuPDF already reads.
_TABLE_MIN_WORDS = 60
_TABLE_MIN_TEXT_COVERAGE = 0.12
#: Axis tick labels of a vector chart sit just outside its drawn lines.
_FIGURE_LABEL_MARGIN = 28.0
#: A native block this much inside a figure region is one of its labels.
_FIGURE_BLOCK_OVERLAP = 0.6
_MAX_FIGURES_PER_PAGE = 8


def find_figure_regions(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Locate charts and diagrams: embedded images and clusters of vector drawings."""
    page_rect = page.rect
    page_area = page_rect.get_area() or 1.0

    candidates: list[tuple[pymupdf.Rect, bool]] = []
    try:
        candidates += [(pymupdf.Rect(i["bbox"]), False) for i in page.get_image_info()]
    except Exception as exc:  # noqa: BLE001 - damaged image streams must not kill the run
        debug("pdf.image_info_failed", page=page.number + 1, error=str(exc)[:200])
    try:
        candidates += [(rect, True) for rect in page.cluster_drawings()]
    except Exception as exc:  # noqa: BLE001
        debug("pdf.cluster_drawings_failed", page=page.number + 1, error=str(exc)[:200])

    words = [pymupdf.Rect(w[:4]) for w in page.get_text("words")]
    regions: list[pymupdf.Rect] = []
    for rect, is_vector in candidates:
        rect = rect & page_rect
        if rect.is_empty or min(rect.width, rect.height) < _FIGURE_MIN_SIDE:
            continue
        fraction = rect.get_area() / page_area
        if not _FIGURE_MIN_PAGE_FRACTION <= fraction <= _FIGURE_MAX_PAGE_FRACTION:
            continue
        if is_vector:
            inside = [w for w in words if rect.contains(w)]
            coverage = sum(w.get_area() for w in inside) / rect.get_area()
            if len(inside) >= _TABLE_MIN_WORDS or coverage >= _TABLE_MIN_TEXT_COVERAGE:
                continue
            margin = _FIGURE_LABEL_MARGIN
            rect = (rect + (-margin, -margin, margin, margin)) & page_rect
        regions.append(rect)

    # An image inside a drawn frame, or two clusters of one chart, is one figure.
    merged: list[pymupdf.Rect] = []
    for rect in sorted(regions, key=lambda r: (r.y0, r.x0)):
        for index, existing in enumerate(merged):
            if existing.intersects(rect):
                merged[index] = existing | rect
                break
        else:
            merged.append(rect)
    return merged[:_MAX_FIGURES_PER_PAGE]


def _overlap_fraction(bbox: tuple[float, float, float, float], rect: pymupdf.Rect) -> float:
    block = pymupdf.Rect(bbox)
    area = block.get_area()
    return (block & rect).get_area() / area if area else 0.0


#: Rules of one table (top, mid, bottom) are closer than this to each other...
_TABLE_RULE_MAX_GAP = 150.0
#: ... and the first of them sits this close to the caption.
_TABLE_CAPTION_MAX_GAP = 40.0
#: Without rules, the table body is the block(s) this close below the caption.
_TABLE_BODY_MAX_GAP = 30.0
_TABLE_BODY_LINE_GAP = 6.0


def _horizontal_rules(page: pymupdf.Page) -> list[pymupdf.Rect]:
    try:
        drawings = page.get_drawings()
    except Exception as exc:  # noqa: BLE001 - damaged drawing streams must not kill the run
        debug("pdf.drawings_failed", page=page.number + 1, error=str(exc)[:200])
        return []
    rules = [d["rect"] for d in drawings if d["rect"].height < 2 and d["rect"].width >= 60]
    return sorted(rules, key=lambda r: r.y0)


def _ruled_table_region(
    rules: list[pymupdf.Rect], caption: pymupdf.Rect, *, below: bool
) -> pymupdf.Rect | None:
    """Chain the horizontal rules starting next to a caption into one table."""

    def overlaps(rule):
        return rule.x0 < caption.x1 and rule.x1 > caption.x0

    ordered = rules if below else list(reversed(rules))
    gap = (lambda r: r.y0 - caption.y1) if below else (lambda r: caption.y0 - r.y1)
    start = next(
        (r for r in ordered if overlaps(r) and 0 <= gap(r) <= _TABLE_CAPTION_MAX_GAP), None
    )
    if start is None:
        return None
    chain = [start]
    for rule in ordered[ordered.index(start) + 1 :]:
        distance = rule.y0 - chain[-1].y1 if below else chain[-1].y0 - rule.y1
        if distance > _TABLE_RULE_MAX_GAP:
            break
        if distance > 1 and overlaps(rule):
            chain.append(rule)
    if len(chain) < 2:  # a single rule is a separator, not a table
        return None
    return pymupdf.Rect(
        min(r.x0 for r in chain),
        min(r.y0 for r in chain) - 3,
        max(r.x1 for r in chain),
        max(r.y1 for r in chain) + 3,
    )


def find_table_regions(page: pymupdf.Page, blocks: list[TextBlock]) -> list[pymupdf.Rect]:
    """Locate tables from their "Table N:" captions and the rules drawn around them.

    Anchoring on captions keeps prose out: PyMuPDF's own table finder only sees
    fully ruled grids and misses the common top/mid/bottom-rule layout.
    """
    from .elements import extract_element_id  # elements imports this module

    rules = _horizontal_rules(page)
    regions: list[pymupdf.Rect] = []
    for index, block in enumerate(blocks):
        if extract_element_id(block.text)[0] is not ElementType.TABLE:
            continue
        caption = pymupdf.Rect(block.bbox)
        region = _ruled_table_region(rules, caption, below=True) or _ruled_table_region(
            rules, caption, below=False
        )
        if region is None:
            body: list[TextBlock] = []
            for candidate in blocks[index + 1 :]:
                limit = _TABLE_BODY_MAX_GAP if not body else _TABLE_BODY_LINE_GAP
                top = caption.y1 if not body else body[-1].bbox[3]
                if not 0 < candidate.bbox[1] - top <= limit:
                    break
                if extract_element_id(candidate.text)[0] is not ElementType.UNKNOWN:
                    break
                body.append(candidate)
            if not body:
                continue
            region = pymupdf.Rect(body[0].bbox)
            for b in body[1:]:
                region |= pymupdf.Rect(b.bbox)
        if not any(region.intersects(r) for r in regions):
            regions.append(region)
    return regions


def _add_table_blocks(page: pymupdf.Page, parsed: ParsedPage, ocr_engine) -> list[pymupdf.Rect]:
    """Replace each table's native blocks with the engine's verified reading.

    Returns every table region found, including those whose reading was
    rejected, so figure detection does not treat a table's rules as a chart.
    """
    regions = find_table_regions(page, parsed.blocks)
    for rect in regions:
        inside = [b for b in parsed.blocks if _overlap_fraction(b.bbox, rect) >= _FIGURE_BLOCK_OVERLAP]
        if not inside:
            continue
        try:
            result = ocr_engine.extract_table(
                page,
                rect + (-4, -4, 4, 4),
                page_number=parsed.page,
                printed_text=page.get_text("text", clip=rect),
            )
        except Exception as exc:  # noqa: BLE001 - a failed table keeps its native text
            warning("ingestion.table_ocr_failed", page=parsed.page, error=str(exc)[:200])
            continue
        if result is None or not result.text.strip():
            continue

        position = parsed.blocks.index(inside[0])
        replaced = {id(b) for b in inside}
        blocks = [b for b in parsed.blocks if id(b) not in replaced]
        blocks.insert(
            position,
            TextBlock(
                page=parsed.page,
                text=result.text.strip(),
                bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                source=ExtractionMethod.OCR,
                element_hint=ElementType.TABLE,
                confidence=result.confidence,
            ),
        )
        parsed.blocks = blocks
    return regions


#: Spans of one line further apart than this many font sizes are separate cells.
_CELL_GAP_EM = 0.8
#: A reconstructed grid needs this many rows with two or more cells, else the
#: region was prose that happened to sit under a caption.
_MIN_GRID_ROWS = 2
_MAX_HEADER_ROWS = 3
#: "28.4", "100K", "−0.5", "92%": a cell that is a value rather than a label.
_NUMERIC_CELL = re.compile(r"^[-+−]?\d[\d.,]*\s*[%KMBkmb]?$")


@dataclass(slots=True)
class _Cell:
    text: str
    x0: float
    x1: float
    y: float
    height: float


def _table_cells(page: pymupdf.Page, rect: pymupdf.Rect) -> list[_Cell]:
    """Cells inside a table region: line spans split wherever a column gap opens."""
    try:
        raw = page.get_text("dict", clip=rect)
    except Exception as exc:  # noqa: BLE001 - a damaged page must not kill the run
        debug("pdf.table_cells_failed", page=page.number + 1, error=str(exc)[:200])
        return []
    cells: list[_Cell] = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            x0, y0, x1, y1 = line["bbox"]
            if not rect.y0 <= (y0 + y1) / 2 <= rect.y1:
                continue  # descenders of the caption above, clipped into the region
            group: list[dict] = []
            for span in line.get("spans", []):
                if not span.get("text", "").strip() and not group:
                    continue
                if group:
                    gap = span["bbox"][0] - group[-1]["bbox"][2]
                    if gap > _CELL_GAP_EM * max(float(span.get("size") or 0), 1.0):
                        cells.append(_cell_from_spans(group))
                        group = []
                group.append(span)
            if group:
                cells.append(_cell_from_spans(group))
    return [c for c in cells if c.text]


def _cell_from_spans(spans: list[dict]) -> _Cell:
    x0 = min(s["bbox"][0] for s in spans)
    x1 = max(s["bbox"][2] for s in spans)
    y0 = min(s["bbox"][1] for s in spans)
    y1 = max(s["bbox"][3] for s in spans)
    return _Cell(
        text=" ".join(normalize_text(_span_text(spans)).split()),
        x0=x0,
        x1=x1,
        y=(y0 + y1) / 2,
        height=y1 - y0,
    )


def _grid_rows(cells: list[_Cell]) -> list[list[_Cell]]:
    """Group cells into visual rows by vertical centre."""
    if not cells:
        return []
    heights = sorted(c.height for c in cells)
    tolerance = max(heights[len(heights) // 2] * 0.5, 2.0)
    rows: list[list[_Cell]] = []
    for cell in sorted(cells, key=lambda c: (c.y, c.x0)):
        if rows and abs(cell.y - sum(c.y for c in rows[-1]) / len(rows[-1])) <= tolerance:
            rows[-1].append(cell)
        else:
            rows.append([cell])
    return [sorted(row, key=lambda c: c.x0) for row in rows]


def table_grid_text(cells: list[_Cell]) -> str | None:
    """Lay table cells out as a Markdown table, or ``None`` if they are not a grid.

    Column bands come from the fullest rows, which carry one cell per column;
    every other cell goes to the band it overlaps most (a spanning header to
    the first of its columns), so a row with blanks keeps its values under the
    right headers instead of shifting left.
    """
    rows = _grid_rows(cells)
    if sum(1 for row in rows if len(row) >= 2) < _MIN_GRID_ROWS:
        return None
    width = max(len(row) for row in rows)
    full = [row for row in rows if len(row) == width]
    bands = [
        (min(row[i].x0 for row in full), max(row[i].x1 for row in full)) for i in range(width)
    ]

    def columns(cell: _Cell, *, spanning: bool) -> list[int]:
        overlaps = [min(cell.x1, b1) - max(cell.x0, b0) for b0, b1 in bands]
        if spanning and sum(1 for o in overlaps if o > 0) > 1:
            return [i for i in range(width) if overlaps[i] > 0]
        best = max(range(width), key=lambda i: overlaps[i])  # ties: the first column
        if overlaps[best] > 0:
            return [best]
        centre = (cell.x0 + cell.x1) / 2
        return [min(range(width), key=lambda i: abs(centre - (bands[i][0] + bands[i][1]) / 2))]

    # Header rows are the rows above the first row holding a number; a header
    # cell spanning several columns ("BLEU" over EN-DE and EN-FR) names each.
    first_numeric = next(
        (i for i, row in enumerate(rows) if any(_NUMERIC_CELL.match(c.text) for c in row)),
        1,
    )
    header_rows = rows[: max(1, min(first_numeric, _MAX_HEADER_ROWS))]
    header = [""] * width
    for row in header_rows:
        for cell in row:
            for index in columns(cell, spanning=True):
                header[index] = f"{header[index]} {cell.text}".strip()

    def render(slots: list[str]) -> str:
        return "| " + " | ".join(s.replace("|", "\\|") for s in slots) + " |"

    lines = [render(header), "|" + " --- |" * width]
    for row in rows[len(header_rows) :]:
        slots = [""] * width
        for cell in row:
            index = columns(cell, spanning=False)[0]
            slots[index] = f"{slots[index]} {cell.text}".strip()
        lines.append(render(slots))
    return "\n".join(lines)


def _add_native_table_blocks(
    page: pymupdf.Page, parsed: ParsedPage, regions: list[pymupdf.Rect]
) -> None:
    """Replace each table's one-cell-per-block native text with its grid.

    Runs for every table region an engine did not already read. The grid comes
    from the text layer's own coordinates, so no value is changed, only placed
    in its row and column.
    """
    for rect in regions:
        if any(
            b.element_hint is ElementType.TABLE and pymupdf.Rect(b.bbox).intersects(rect)
            for b in parsed.blocks
        ):
            continue
        inside = [
            b for b in parsed.blocks if _overlap_fraction(b.bbox, rect) >= _FIGURE_BLOCK_OVERLAP
        ]
        if not inside:
            continue
        text = table_grid_text(_table_cells(page, rect))
        if text is None:
            continue
        position = parsed.blocks.index(inside[0])
        replaced = {id(b) for b in inside}
        blocks = [b for b in parsed.blocks if id(b) not in replaced]
        blocks.insert(
            position,
            TextBlock(
                page=parsed.page,
                text=text,
                bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                font_size=inside[0].font_size,
                source=ExtractionMethod.NATIVE_TEXT,
                element_hint=ElementType.TABLE,
            ),
        )
        parsed.blocks = blocks


def _add_figure_blocks(
    page: pymupdf.Page,
    parsed: ParsedPage,
    ocr_engine,
    *,
    exclude: list[pymupdf.Rect] = (),
) -> None:
    """Replace the native fragments of each chart or diagram with its OCR reading.

    The native blocks inside a figure are axis ticks and box labels, meaningless
    as paragraphs; they are handed to the engine as printed text instead. The
    figure block is placed just before the first block below it (normally its
    caption), so the caption's id attaches to it during chunking.
    """
    for rect in find_figure_regions(page):
        if any(rect.intersects(table) for table in exclude):
            continue
        inside = [b for b in parsed.blocks if _overlap_fraction(b.bbox, rect) >= _FIGURE_BLOCK_OVERLAP]
        try:
            result = ocr_engine.extract_figure(
                page,
                rect,
                page_number=parsed.page,
                printed_text="\n".join(b.text for b in inside),
            )
        except Exception as exc:  # noqa: BLE001 - a failed figure must not kill the run
            warning("ingestion.figure_ocr_failed", page=parsed.page, error=str(exc)[:200])
            continue
        if result is None or not result.text.strip():
            continue

        replaced = {id(b) for b in inside}
        blocks = [b for b in parsed.blocks if id(b) not in replaced]
        figure = TextBlock(
            page=parsed.page,
            text=normalize_text(result.text),
            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
            source=ExtractionMethod.OCR,
            element_hint=ElementType.FIGURE,
            confidence=result.confidence,
        )
        position = next(
            (
                i
                for i, b in enumerate(blocks)
                if b.bbox[1] >= rect.y1 - 2 and b.bbox[0] < rect.x1 and b.bbox[2] > rect.x0
            ),
            len(blocks),
        )
        blocks.insert(position, figure)
        parsed.blocks = blocks


def _title_block(page: ParsedPage) -> TextBlock | None:
    """The block holding the paper title: the largest type on the first page."""
    candidates = [
        b
        for b in page.blocks
        if b.font_size > 0 and 8 <= len(b.text.strip()) <= 300 and b.text.count("\n") <= 3
    ]
    if not candidates:
        return None
    largest = max(candidates, key=lambda b: b.font_size)
    text = _joined(largest)
    if text.lower().startswith(("arxiv", "preprint", "under review", "published as")):
        remaining = [b for b in candidates if b is not largest]
        if remaining:
            return max(remaining, key=lambda b: b.font_size)
    return largest


def _joined(block: TextBlock) -> str:
    return " ".join(part.strip() for part in block.text.split("\n") if part.strip())


def _title_from_layout(page: ParsedPage) -> str:
    """Guess the title from page 1 typography.

    The title is normally the largest type on the first page and often wraps
    across two or three lines, so the whole block is taken and its internal line
    breaks collapsed — truncating at the first newline loses half the title.
    """
    block = _title_block(page)
    if block is not None:
        return _joined(block)
    for line in page.text.split("\n"):
        stripped = line.strip()
        if 8 <= len(stripped) <= 200:
            return stripped
    return ""


#: Affiliation footnote markers attached to a name: "Denisa VASILE^1", "J. Doe*",
#: "Ashish Vaswani^∗". Typesetters reach for several different asterisks.
_MARK_CHARS = r"\d*∗⋆★†‡§¶"
_AUTHOR_MARKS = re.compile(rf"\^?[{_MARK_CHARS}]+(?:\s*,\s*[{_MARK_CHARS}]+)*")
_AUTHOR_SPLIT = re.compile(r"\s*(?:,|;|\band\b|&)\s*", re.IGNORECASE)
#: Lines that sit where an author list sits but are not one.
_NOT_AUTHORS = re.compile(
    r"@|\bhttps?://|\b(?:university|institute|department|faculty|college|school|"
    r"laborator|abstract|keywords|doi)\b",
    re.IGNORECASE,
)
#: How many blocks after the title to look in before giving up.
_AUTHOR_SEARCH_DEPTH = 3
_MAX_AUTHORS = 60


_PARTICLES = frozenset({"van", "von", "de", "del", "della", "der", "den", "da", "di", "du", "la"})


def _looks_like_person(name: str) -> bool:
    """True for "Alin ZAMFIROIU", "E.A. van Dis", "J. Doe" — not for prose."""
    words = name.split()
    if not 1 < len(words) <= 6:
        return False
    if any(c.isdigit() for c in name):
        return False
    # Every word is either capitalised, an initial, or a lowercase nobiliary
    # particle ("van", "de"); a sentence fails on its first ordinary verb.
    return all(w[:1].isupper() or w.lower() in _PARTICLES for w in words) and any(
        w[:1].isupper() for w in words
    )


def _normalise_name(name: str) -> str:
    """Undo the all-caps surnames journals set: "ZAMFIROIU" -> "Zamfiroiu"."""
    return " ".join(
        word.title() if word.isupper() and len(word) > 1 and word.isalpha() else word
        for word in name.split()
    )


def _authors_from_layout(page: ParsedPage) -> list[str]:
    """Read the author list from the block under the title.

    A PDF's embedded ``author`` metadata is whatever the submitting tool wrote
    there: frequently one name for a paper with several, or the name of whoever
    exported the file. The author list under the title is the paper's own
    statement of authorship, so it is preferred whenever it parses.

    Two or more names are required. A layout that gives each author a block of
    their own — the grid of names across the top of a conference paper — offers
    no way to tell "Noam Shazeer" from "Google Brain" on the line beneath it, so
    reading one name out of eight is the likely outcome. Reporting no authors
    and leaving the embedded field to stand is the better failure.
    """
    title = _title_block(page)
    if title is None:
        return []
    # Set smaller than the title, which also excludes the title's own wrapped
    # continuation lines — they sit below it and read as capitalised phrases.
    below = [
        b
        for b in page.blocks
        if b.bbox[1] >= title.bbox[3]
        and b.text.strip()
        and not b.rotated
        and 0 < b.font_size < title.font_size
    ]
    below.sort(key=lambda b: (b.bbox[1], b.bbox[0]))

    for block in below[:_AUTHOR_SEARCH_DEPTH]:
        if len(block.text) > 400 or _NOT_AUTHORS.search(block.text):
            continue
        names: list[str] = []
        for line in block.text.split("\n"):
            # Authors are separated by commas on one line, or set one per line.
            for part in _AUTHOR_SPLIT.split(_AUTHOR_MARKS.sub("", line)):
                cleaned = part.strip(" .,")
                if cleaned:
                    names.append(_normalise_name(cleaned))
        if 1 < len(names) <= _MAX_AUTHORS and all(_looks_like_person(n) for n in names):
            return names
    return []


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_YEAR = r"((?:19|20)\d{2})"
#: A year on a venue or copyright line is the publication year.
_VENUE_YEAR_RE = re.compile(
    r"(?:\b(?:conference|proceedings|workshop|symposium|journal|transactions|published"
    r"|copyright)\b[^\n]{0,120}?|©\s*|\b(?:NeurIPS|NIPS|ICML|ICLR|ACL|NAACL|EMNLP|CVPR"
    r"|ICCV|ECCV|AAAI|IJCAI|KDD|SIGIR|COLING|WWW)\s*)\b" + _YEAR + r"\b",
    re.IGNORECASE,
)
#: arXiv identifiers encode the first submission's year and month (YYMM.NNNNN).
_ARXIV_ID_RE = re.compile(r"arXiv:\s*(\d{2})(0[1-9]|1[0-2])\.\d{4,5}", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b" + _YEAR + r"\b")
_BODY_START_RE = re.compile(r"\n\s*(?:abstract|\d?\.?\s*introduction)\b", re.IGNORECASE)


def _publication_year(first_page_text: str) -> int | None:
    """Best-effort publication year from the first page.

    Years in the abstract usually name datasets or benchmarks ("WMT 2014"), so
    only venue lines, arXiv identifiers and the header above the abstract count.
    Returning nothing beats reporting a wrong year.

    The venue pattern is confined to that header for the same reason. Its cue
    words are ordinary English — a body sentence such as "2 materials were
    published in 2022" satisfies "published … <year>" and would otherwise be
    read as a publication line, which on a paper whose first page carries body
    text is a wrong year rather than a missing one.
    """
    body = _BODY_START_RE.search(first_page_text)
    header = first_page_text[: body.start()] if body else first_page_text

    venue = _VENUE_YEAR_RE.search(header)
    if venue:
        return int(venue.group(1))
    arxiv = _ARXIV_ID_RE.search(first_page_text)
    if arxiv:
        return 2000 + int(arxiv.group(1))
    year = _YEAR_RE.search(header) if body else None
    return int(year.group(1)) if year else None


def _document_metadata(doc: pymupdf.Document, pages: list[ParsedPage]) -> dict:
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip() or None
    author_field = (meta.get("author") or "").strip()
    authors = [a.strip() for a in re.split(r",| and |;", author_field) if a.strip()]

    first_page_text = pages[0].text if pages else ""
    if pages:
        if not title:
            title = _title_from_layout(pages[0]) or None
        # The embedded field routinely names one author of several, so it only
        # stands when the page itself offers nothing fuller.
        from_page = _authors_from_layout(pages[0])
        if len(from_page) > len(authors):
            authors = from_page

    doi_match = _DOI_RE.search(first_page_text)
    return {
        "title": title,
        "authors": authors,
        "doi": doi_match.group(0) if doi_match else None,
        "year": _publication_year(first_page_text),
    }


def parse_pdf(
    path: Path,
    *,
    ocr_engine=None,
    ocr_min_chars: int = 180,
    max_bytes: int = 50 * 1024 * 1024,
    figure_ocr: bool = True,
    table_ocr: bool = True,
    layout_engine: LayoutEngine | None = None,
) -> ParsedDocument:
    """Extract every page of ``path``, falling back to OCR where needed.

    ``ocr_engine`` is any object implementing :class:`~paperpilot.ingestion.ocr.OCREngine`.
    Passing ``None`` disables OCR entirely; pages that need it keep whatever
    native text exists and record a reduced confidence.

    Pages with a native text layer keep it. When ``figure_ocr`` is set and the
    engine supports it, the charts and diagrams on those pages are additionally
    read by the engine, since their content is not in the text layer. When
    ``table_ocr`` is set, tables are re-read by the engine to recover their grid,
    and the reading replaces the native text only once verified against it.

    ``layout_engine`` is any object implementing
    :class:`~paperpilot.ingestion.layout.LayoutEngine`. When one is supplied and
    available, each page's regions are recorded on the page for the chunker to
    re-cut the blocks along; passing ``None`` leaves the geometric heuristics to
    recover the structure on their own.
    """
    size = validate_pdf_file(path, max_bytes=max_bytes)
    digest = sha256_of(path)

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"could not open PDF: {exc}", details={"file": path.name}) from exc

    pages: list[ParsedPage] = []
    try:
        if doc.page_count == 0:
            raise IngestionError("PDF contains no pages", details={"file": path.name})

        try:
            labels = doc.get_page_labels()
        except Exception:  # noqa: BLE001 - labels are optional metadata
            labels = []
        has_labels = bool(labels)

        for index in range(doc.page_count):
            page_number = index + 1
            page = doc.load_page(index)
            native_text = normalize_text(page.get_text("text"))
            blocks = _extract_blocks(page, page_number)

            parsed = ParsedPage(
                page=page_number,
                text=native_text,
                blocks=blocks,
                source=ExtractionMethod.NATIVE_TEXT,
                confidence=1.0,
                width=float(page.rect.width),
                height=float(page.rect.height),
            )

            if has_labels:
                try:
                    label = page.get_label()
                    parsed.printed_label = label or None
                except Exception:  # noqa: BLE001
                    parsed.printed_label = None

            failures = extraction_failures(
                native_text, min_chars=ocr_min_chars, image_coverage=_image_coverage(page)
            )
            parsed.extraction_failures = failures
            if failures:
                info("ingestion.page_flagged", page=page_number, reasons=failures)
                ocr_result = (
                    ocr_engine.extract_page(page, page_number=page_number)
                    if ocr_engine is not None
                    else None
                )
                if ocr_result is not None and ocr_result.text.strip():
                    # The model's reading is taken as-is: the native layer
                    # already failed the check, so it is not a yardstick.
                    parsed.text = normalize_text(ocr_result.text)
                    parsed.source = ExtractionMethod.OCR
                    parsed.confidence = ocr_result.confidence
                    parsed.blocks = [
                        TextBlock(
                            page=page_number,
                            text=parsed.text,
                            bbox=(0.0, 0.0, parsed.width, parsed.height),
                            source=ExtractionMethod.OCR,
                        )
                    ]
                    info(
                        "ingestion.ocr_page",
                        page=page_number,
                        chars=len(parsed.text),
                        engine=ocr_result.engine,
                    )
                else:
                    # No reading: be explicit that the kept native text is suspect.
                    parsed.confidence = 0.4
            else:
                if ocr_engine is not None and table_ocr and ocr_engine.supports_tables:
                    tables = _add_table_blocks(page, parsed, ocr_engine)
                else:
                    tables = find_table_regions(page, parsed.blocks)
                # Tables the engine did not read (or could not) still get their grid.
                _add_native_table_blocks(page, parsed, tables)
                if ocr_engine is not None and figure_ocr and ocr_engine.supports_figures:
                    _add_figure_blocks(page, parsed, ocr_engine, exclude=tables)

            if (
                layout_engine is not None
                and parsed.source is ExtractionMethod.NATIVE_TEXT
                and layout_engine.available
            ):
                parsed.layout = layout_engine.analyze_page(page, page_number=page_number)

            pages.append(parsed)

        # The layout model reports running heads as header/footer regions, and
        # the chunker drops the blocks inside them. The repetition heuristic is
        # what covers the pages it did not analyse.
        _strip_running_heads([p for p in pages if p.layout is None])

        meta = _document_metadata(doc, pages)
    finally:
        doc.close()

    return ParsedDocument(
        pages=pages,
        page_count=len(pages),
        file_size_bytes=size,
        sha256=digest,
        filename=path.name,
        **meta,
    )


#: A text layer below this share of letters is symbol soup, not prose.
_MIN_LETTER_RATIO = 0.5
#: Broken glyph units per word above which the font encoding is unreadable.
_MAX_BROKEN_GLYPHS_PER_WORD = 0.05
#: Share of word characters in non-words above which the text layer is garbled.
#: Weighted by length, because one run-together token can hold a whole line.
_MAX_GARBLED_WORD_RATIO = 0.3
#: Less word text than this is too small a sample to judge garbling. Counted in
#: characters, since a garbled layer may hold only a few (very long) tokens.
_MIN_WORD_CHARS_TO_JUDGE = 100
#: A page this covered by raster images is a scan, whatever its text layer holds.
_SCANNED_IMAGE_COVERAGE = 0.85

#: Glyphs a font could not map to Unicode: PyMuPDF's ``(cid:N)`` placeholders,
#: the replacement character, private-use code points and stray control codes.
_BROKEN_GLYPH = re.compile(r"\(cid:\d+\)|[\ufffd\ue000-\uf8ff\x00-\x08\x0b\x0c\x0e-\x1f]")
_WORD_TOKEN = re.compile(r"[^\W\d_]{4,}")
_VOWEL = re.compile(r"[aeiouyAEIOUY]")
#: Words run together when the text layer has no space glyphs.
_RUN_TOGETHER_LENGTH = 25


def _is_garbled_word(word: str) -> bool:
    if word.isupper() or (word[:-1].isupper() and word[-1] == "s"):
        return False  # acronyms (CNN, LSTMs) legitimately have no vowels
    if not word.isascii():
        return False  # non-Latin scripts are not judged by Latin vowels
    return len(word) >= _RUN_TOGETHER_LENGTH or not _VOWEL.search(word)


def extraction_failures(
    text: str, *, min_chars: int = 180, image_coverage: float = 0.0
) -> list[str]:
    """Why a page's native text layer cannot be trusted; empty when it can.

    Deliberately non-LLM and cheap. Each check targets a different way a PDF
    text layer goes wrong, and any one failing sends the page to the OCR model:

    - **thin**: fewer than ``min_chars`` usable characters (a scan, or a page
      that is one big image);
    - **symbols**: mostly non-letters (scan artefacts, stray glyphs);
    - **broken glyphs**: characters the font could not map to Unicode;
    - **garbled words**: tokens that are not words — no vowels, or several
      words run together because the layer has no space glyphs;
    - **scanned**: raster images cover nearly the whole page, so whatever text
      layer exists was laid over a scan by some earlier OCR pass.
    """
    stripped = (text or "").strip()
    failures: list[str] = []
    if image_coverage >= _SCANNED_IMAGE_COVERAGE:
        failures.append(f"scanned page ({image_coverage:.0%} covered by images)")
    if len(stripped) < min_chars:
        failures.append(f"thin text layer ({len(stripped)} < {min_chars} chars)")
        return failures
    letters = sum(1 for c in stripped if c.isalpha())
    if letters / len(stripped) < _MIN_LETTER_RATIO:
        failures.append(f"mostly non-letters ({letters / len(stripped):.0%} letters)")

    words = stripped.split()
    broken = len(_BROKEN_GLYPH.findall(stripped))
    if broken / max(len(words), 1) > _MAX_BROKEN_GLYPHS_PER_WORD:
        failures.append(f"broken glyphs ({broken} unmapped characters)")

    candidates = _WORD_TOKEN.findall(stripped)
    word_chars = sum(len(w) for w in candidates)
    if word_chars >= _MIN_WORD_CHARS_TO_JUDGE:
        garbled = sum(len(w) for w in candidates if _is_garbled_word(w))
        share = garbled / word_chars
        if share > _MAX_GARBLED_WORD_RATIO:
            failures.append(f"garbled words ({share:.0%} of word text is not words)")
    return failures


def needs_ocr(text: str, *, min_chars: int = 180, image_coverage: float = 0.0) -> bool:
    """Whether a page's text layer fails :func:`extraction_failures`."""
    return bool(extraction_failures(text, min_chars=min_chars, image_coverage=image_coverage))


def _image_coverage(page: pymupdf.Page) -> float:
    """Share of the page area covered by raster images (overlaps counted once each)."""
    area = abs(page.rect)
    if not area:
        return 0.0
    try:
        infos = page.get_image_info()
    except Exception:  # noqa: BLE001 - coverage is a hint, never a failure
        return 0.0
    covered = sum(abs(pymupdf.Rect(i["bbox"]) & page.rect) for i in infos if i.get("bbox"))
    return min(covered / area, 1.0)
