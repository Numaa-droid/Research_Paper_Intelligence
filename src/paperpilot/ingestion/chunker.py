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

from dataclasses import dataclass, field

from ..domain.enums import ElementType, ExtractionMethod
from ..domain.models import PaperChunk
from .elements import (
    DetectedHeading,
    body_font_size,
    classify_block,
    looks_like_heading,
    looks_like_table,
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
        self.seen_canonical = True
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
        start = max(end - overlap, start + 1)
    return pieces


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
            source=self.page.source,
            char_start=self.page_offset,
            char_end=self.page_offset + len(text),
            bbox=bbox,
            confidence=self.page.confidence,
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

    def _backfill_caption(self, element_type: ElementType, element_id: str) -> None:
        """Attach a caption's id to a body block that appeared *before* it."""
        for previous in reversed(self.chunks[:-1]):
            if previous.element_type is element_type and previous.element_id is None:
                previous.element_id = element_id
                return
            if previous.element_type is ElementType.PARAGRAPH:
                return

    # --- main loop --------------------------------------------------------

    def run(self, blocks: list[TextBlock]) -> list[PaperChunk]:
        for block in blocks:
            self._handle(block)
        self._flush()
        return self.chunks

    def _handle(self, block: TextBlock) -> None:
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
            self._backfill_caption(element_type, element_id)  # type: ignore[arg-type]
            self.pending_caption = (element_type, element_id)  # type: ignore[assignment]
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

        # A caption followed by a block the detector did not recognise as a
        # table: trust the caption when the block still looks tabular.
        if (
            self.pending_caption
            and self.pending_caption[0] is ElementType.TABLE
            and looks_like_table(block.text)
        ):
            self._flush()
            self._emit(
                block.text.strip()[: self.max_chars * 3],
                element_type=ElementType.TABLE,
                element_id=self.pending_caption[1],
                bbox=block.bbox,
            )
            self.pending_caption = None
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

    return chunks, cursor.ordered


def element_counts(chunks: list[PaperChunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.element_type.value] = counts.get(chunk.element_type.value, 0) + 1
    return counts


def ocr_chunk_count(chunks: list[PaperChunk]) -> int:
    return sum(1 for c in chunks if c.source == ExtractionMethod.OCR)
