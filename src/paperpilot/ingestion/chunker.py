"""Page-aware chunking.

Two rules are absolute:

1. A chunk never spans a page boundary, so every chunk can carry an honest page
   number into a citation.
2. Metadata (page, section, subsection, element type, element id, extraction
   method) is attached at construction and travels with the text everywhere
   afterwards.

Structural elements — tables, figures, equations — are kept atomic rather than
merged into a running paragraph chunk, because they are exactly what agents need
to cite precisely. Captions are linked to the adjacent body block so that
"Table 1" in a citation points at the numbers, not only at the caption text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..domain.enums import ElementType, ExtractionMethod
from ..domain.models import PaperChunk
from .elements import (
    DetectedHeading,
    apply_page_layout,
    assemble_display_equations,
    body_font_size,
    classify_block,
    extract_element_id,
    looks_like_heading,
    looks_like_table_body,
    split_display_equations,
    split_embedded_headings,
)
from .pdf_parser import ParsedDocument, ParsedPage, TextBlock

#: Element types that are never merged with neighbouring text.
ATOMIC_ELEMENTS = frozenset({ElementType.TABLE, ElementType.FIGURE, ElementType.EQUATION})

#: Label used for the title/author block that precedes the first real section.
FRONT_MATTER = "Front Matter"


@dataclass
class _SectionCursor:
    """Tracks where in the document's section hierarchy we currently are."""

    section: str | None = FRONT_MATTER
    subsection: str | None = None
    seen_canonical: bool = False
    seen_numbered: bool = False
    ordered: list[str] = field(default_factory=list)

    def apply(self, heading: DetectedHeading) -> bool:
        """Advance the cursor. Returns False when the heading is front matter."""
        if not self.seen_canonical and (
            not heading.is_canonical or heading.is_alpha_numbered
        ):
            # Title, author list and affiliation all look like headings; a real
            # section (numbered, or a canonical name such as "Abstract") has to
            # appear before letter-numbered appendix headings are believable.
            return False
        if self.seen_numbered and not heading.is_canonical:
            # A paper that numbers its sections numbers all of them. Anything
            # unnumbered that still reads as a heading is set-apart text inside
            # the page — a chart title, a table's column header — and naming a
            # section after it puts every citation below it under that name.
            return False
        self.seen_canonical = True
        self.seen_numbered = self.seen_numbered or heading.number is not None
        if heading.level >= 2 and self.section not in (None, FRONT_MATTER):
            self.subsection = heading.label
        else:
            self.section = heading.label
            self.subsection = None
        if heading.label not in self.ordered:
            self.ordered.append(heading.label)
        return True


def _page_blocks(page: ParsedPage) -> list[TextBlock]:
    if page.blocks:
        return page.blocks
    if not page.text.strip():
        return []
    # OCR pages (and damaged native pages) arrive as one undifferentiated string;
    # split on blank lines so chunking still has something to work with.
    blocks: list[TextBlock] = []
    for index, part in enumerate(page.text.split("\n\n")):
        if part.strip():
            blocks.append(
                TextBlock(
                    page=page.page,
                    text=part.strip(),
                    bbox=(0.0, 0.0, page.width, page.height),
                    block_no=index,
                    source=page.source,
                )
            )
    return blocks


def _split_markdown_table(text: str, *, max_chars: int) -> list[str]:
    """Split a Markdown table between rows, repeating the header in every piece.

    A table cut mid-row, or a piece without its header, is unreadable: the
    numbers in it no longer say which column they belong to.
    """
    lines = text.split("\n")
    if len(text) <= max_chars or len(lines) <= 3:
        return [text]
    header, body = lines[:2], lines[2:]
    pieces: list[str] = []
    current: list[str] = []
    for row in body:
        if current and len("\n".join(header + current + [row])) > max_chars:
            pieces.append("\n".join(header + current))
            current = []
        current.append(row)
    pieces.append("\n".join(header + current))
    return pieces


def _split_long_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Split oversized text on sentence-ish boundaries with a small overlap."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in (". ", ".\n", "; ", "\n", " "):
                cut = window.rfind(sep)
                if cut > max_chars * 0.5:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = _overlap_start(text, max(end - overlap, start + 1), end)
    return pieces


def _overlap_start(text: str, earliest: int, end: int) -> int:
    """Where the overlap of the next piece begins: a sentence, else a word.

    Starting at a fixed character offset cuts words in half ("led
    intra-attention" for "called intra-attention"), and a quote copied from
    such a chunk no longer reads as the paper's sentence.
    """
    window = text[earliest:end]
    for sep in (". ", ".\n"):
        cut = window.find(sep)
        if cut >= 0 and earliest + cut + len(sep) < end:
            return earliest + cut + len(sep)
    if earliest > 0 and not text[earliest - 1].isspace():
        space = next((i for i, c in enumerate(window) if c.isspace()), None)
        if space is not None and earliest + space + 1 < end:
            return earliest + space + 1
    return earliest


class _PageChunker:
    """Builds the chunks for exactly one page.

    Extracted from :func:`chunk_document` so that page state (the buffer, the
    running chunk index, the pending caption) is explicit instance state rather
    than variables closed over by nested functions inside a loop.
    """

    def __init__(
        self,
        page: ParsedPage,
        *,
        document_id: str,
        cursor: _SectionCursor,
        baseline_font: float,
        max_chars: int,
        overlap: int,
    ) -> None:
        self.page = page
        self.document_id = document_id
        self.cursor = cursor
        self.baseline_font = baseline_font
        self.max_chars = max_chars
        self.overlap = overlap

        self.chunks: list[PaperChunk] = []
        self.buffer: list[TextBlock] = []
        self.chunk_index = 0
        self.page_offset = 0
        self.equation_counter = 0
        #: A caption seen but not yet attached to a body block.
        self.pending_caption: tuple[ElementType, str] | None = None

    # --- emit helpers -----------------------------------------------------

    def _emit(
        self,
        text: str,
        *,
        element_type: ElementType,
        element_id: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        source: ExtractionMethod | None = None,
        confidence: float | None = None,
    ) -> PaperChunk:
        self.chunk_index += 1
        chunk = PaperChunk(
            chunk_id=f"{self.document_id}-page-{self.page.page}-chunk-{self.chunk_index:02d}",
            document_id=self.document_id,
            page=self.page.page,
            section=self.cursor.section,
            subsection=self.cursor.subsection,
            element_type=element_type,
            element_id=element_id,
            text=text,
            source=source or self.page.source,
            char_start=self.page_offset,
            char_end=self.page_offset + len(text),
            bbox=bbox,
            confidence=self.page.confidence if confidence is None else confidence,
            printed_label=self.page.printed_label,
        )
        self.page_offset += len(text)
        self.chunks.append(chunk)
        return chunk

    def _flush(self) -> None:
        """Emit the buffered paragraph text, splitting it if it grew too long."""
        if not self.buffer:
            return
        text = "\n".join(b.text for b in self.buffer).strip()
        self.buffer = []
        if not text:
            return
        for piece in _split_long_text(text, max_chars=self.max_chars, overlap=self.overlap):
            self._emit(piece, element_type=ElementType.PARAGRAPH)

    def _backfill_caption(self, element_type: ElementType, element_id: str) -> bool:
        """Attach a caption's id to the body block(s) that appeared *before* it.

        Consecutive pieces share the caption: a figure made of side-by-side
        panels arrives as one OCR block per image.
        """
        attached = False
        for previous in reversed(self.chunks[:-1]):
            if previous.element_type is element_type and previous.element_id is None:
                previous.element_id = element_id
                attached = True
            elif attached or previous.element_type is ElementType.PARAGRAPH:
                break
        return attached

    def _emit_table_body(self, block: TextBlock, element_id: str) -> None:
        """Emit a table body block, joining it to the body already emitted for it."""
        text = block.text.strip()
        last = self.chunks[-1] if self.chunks else None
        is_body = (
            last is not None
            and last.element_type is ElementType.TABLE
            and last.element_id == element_id
            and extract_element_id(last.text)[1] is None  # not the caption itself
        )
        if is_body and len(last.text) + len(text) + 1 <= self.max_chars * 3:
            last.text = f"{last.text}\n{text}"
            last.char_end = (last.char_end or 0) + len(text) + 1
            self.page_offset += len(text) + 1
            if last.bbox and block.bbox:
                last.bbox = (
                    min(last.bbox[0], block.bbox[0]),
                    min(last.bbox[1], block.bbox[1]),
                    max(last.bbox[2], block.bbox[2]),
                    max(last.bbox[3], block.bbox[3]),
                )
            return
        self._emit(
            text[: self.max_chars * 3],
            element_type=ElementType.TABLE,
            element_id=element_id,
            bbox=block.bbox,
        )

    # --- main loop --------------------------------------------------------

    def run(self, blocks: list[TextBlock]) -> list[PaperChunk]:
        # The layout model is the authority where it ran: it was asked what the
        # page contains, rather than inferring it from how the text is set. The
        # geometric splitter is the fallback for pages it did not analyse, and
        # is still applied after it to catch headings inside a region the model
        # returned as one undivided block of text.
        prepared = blocks
        if self.page.layout is not None:
            prepared = apply_page_layout(prepared, self.page.layout)
        prepared = split_embedded_headings(prepared)
        for block in assemble_display_equations(split_display_equations(prepared)):
            self._handle(block)
        self._flush()
        return self.chunks

    def _handle(self, block: TextBlock) -> None:
        if block.element_hint in (ElementType.FIGURE, ElementType.TABLE):
            # Read from the page image: the type is known, and text heuristics
            # would misclassify a chart's data rows as a table.
            self._flush()
            element_type = block.element_hint
            element_id = None
            if self.pending_caption and self.pending_caption[0] is element_type:
                element_id = self.pending_caption[1]
                self.pending_caption = None
            pieces = (
                _split_markdown_table(block.text.strip(), max_chars=self.max_chars * 3)
                if element_type is ElementType.TABLE
                else [block.text.strip()[: self.max_chars * 3]]
            )
            for piece in pieces:
                self._emit(
                    piece,
                    element_type=element_type,
                    element_id=element_id,
                    bbox=block.bbox,
                    source=block.source,
                    confidence=block.confidence,
                )
            return

        heading = looks_like_heading(block, body_font_size=self.baseline_font)
        if heading is not None:
            self._flush()
            if self.cursor.apply(heading):
                self.pending_caption = None
            else:
                # Front matter: keep the text indexed, just not as a section.
                self._emit(block.text.strip(), element_type=ElementType.PARAGRAPH)
            return

        element_type, element_id = classify_block(
            block, section=self.cursor.section, equation_counter=self.equation_counter + 1
        )
        if element_type is ElementType.EQUATION:
            self.equation_counter += 1

        is_caption = element_id is not None and element_type in (
            ElementType.TABLE,
            ElementType.FIGURE,
        )
        if is_caption:
            self._flush()
            self._emit(
                block.text.strip()[: self.max_chars * 3],
                element_type=element_type,
                element_id=element_id,
                bbox=block.bbox,
            )
            attached = self._backfill_caption(element_type, element_id)  # type: ignore[arg-type]
            if attached and element_type is ElementType.FIGURE:
                # Figure captions sit below their figure: once attached, a figure
                # that follows belongs to the next caption, not this one.
                self.pending_caption = None
            else:
                self.pending_caption = (element_type, element_id)  # type: ignore[assignment]
            return

        table_caption = (
            self.pending_caption[1]
            if self.pending_caption and self.pending_caption[0] is ElementType.TABLE
            else None
        )
        if element_type is ElementType.TABLE and table_caption and element_id is None:
            # A table often arrives as several blocks (header, then rows): the
            # caption stays open so every block of the body carries its id.
            self._flush()
            self._emit_table_body(block, table_caption)
            return

        if element_type in ATOMIC_ELEMENTS or element_type is ElementType.REFERENCE:
            self._flush()
            if (
                self.pending_caption
                and element_type is self.pending_caption[0]
                and element_id is None
            ):
                element_id = self.pending_caption[1]
                self.pending_caption = None
            self._emit(
                block.text.strip()[: self.max_chars * 3],
                element_type=element_type,
                element_id=element_id,
                bbox=block.bbox,
            )
            return

        # A caption followed by text the numeric detector does not recognise as a
        # table — a table of words, one cell per line — is still that table.
        if table_caption and looks_like_table_body(block.text):
            self._flush()
            self._emit_table_body(block, table_caption)
            return

        self.pending_caption = None
        self.buffer.append(block)
        if sum(len(b.text) for b in self.buffer) >= self.max_chars:
            self._flush()


def chunk_document(
    document: ParsedDocument,
    *,
    document_id: str,
    max_chars: int = 1200,
    overlap: int = 150,
) -> tuple[list[PaperChunk], list[str]]:
    """Turn a parsed document into page-aware chunks.

    Returns the chunks and the ordered list of detected section labels. The
    section cursor is shared across pages so a section that starts on page 3
    still labels its continuation on page 4.
    """
    all_blocks = [b for page in document.pages for b in _page_blocks(page)]
    baseline_font = body_font_size(all_blocks)

    chunks: list[PaperChunk] = []
    cursor = _SectionCursor()

    for page in document.pages:
        chunker = _PageChunker(
            page,
            document_id=document_id,
            cursor=cursor,
            baseline_font=baseline_font,
            max_chars=max_chars,
            overlap=overlap,
        )
        chunks.extend(chunker.run(_page_blocks(page)))

    attribute_floats(chunks)
    return chunks, cursor.ordered


_FLOAT_REFERENCE = {
    ElementType.TABLE: r"\btables?\s+{number}\b",
    ElementType.FIGURE: r"\b(?:figures?|figs?\.)\s+{number}\b",
}


def attribute_floats(chunks: list[PaperChunk]) -> None:
    """Give each table and figure the section whose text discusses it.

    A float is placed wherever it fits on the page, which is often the top of
    the next page, under a heading it has nothing to do with. The section that
    refers to it by number ("see Table 2") is where it belongs; of the
    references within one page of the float, the nearest in reading order wins
    (the later one on a tie). A float nobody refers to keeps its position.
    """
    positions: dict[str, list[int]] = {}
    for position, chunk in enumerate(chunks):
        if chunk.element_type in _FLOAT_REFERENCE and chunk.element_id:
            positions.setdefault(chunk.element_id, []).append(position)

    for element_id, members in positions.items():
        first = chunks[members[0]]
        number = element_id.split("-", 1)[-1]
        pattern = re.compile(
            _FLOAT_REFERENCE[first.element_type].format(number=re.escape(number)), re.IGNORECASE
        )
        candidates = [
            position
            for position, chunk in enumerate(chunks)
            if chunk.element_type is ElementType.PARAGRAPH
            and abs(chunk.page - first.page) <= 1
            and pattern.search(chunk.text)
        ]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda p: (abs(p - members[0]), p < members[0]))
        for position in members:
            chunks[position].section = chunks[nearest].section
            chunks[position].subsection = chunks[nearest].subsection


def element_counts(chunks: list[PaperChunk]) -> dict[str, int]:
    """Elements per type: a table split over several chunks is still one table."""
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.element_id:
            if chunk.element_id in seen:
                continue
            seen.add(chunk.element_id)
        counts[chunk.element_type.value] = counts.get(chunk.element_type.value, 0) + 1
    return counts


def ocr_chunk_count(chunks: list[PaperChunk]) -> int:
    return sum(1 for c in chunks if c.source == ExtractionMethod.OCR)
