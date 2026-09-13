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
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from ..domain.enums import ExtractionMethod
from ..domain.models import PageText
from ..errors import IngestionError
from ..observability import debug, info

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


def _extract_blocks(page: pymupdf.Page, page_number: int) -> list[TextBlock]:
    """Pull layout blocks with font size, used later for heading detection."""
    blocks: list[TextBlock] = []
    try:
        raw = page.get_text("dict")
    except Exception as exc:  # noqa: BLE001 - a damaged page must not kill the run
        debug("pdf.block_extraction_failed", page=page_number, error=str(exc)[:200])
        return blocks

    for index, block in enumerate(raw.get("blocks", [])):
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        lines: list[str] = []
        sizes: list[float] = []
        bold = False
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            if line_text.strip():
                lines.append(line_text)
            for span in spans:
                if span.get("size"):
                    sizes.append(float(span["size"]))
                font = (span.get("font") or "").lower()
                if "bold" in font or "black" in font or "semibold" in font:
                    bold = True
        text = normalize_text("\n".join(lines))
        if not text:
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
            )
        )
    return blocks


def _title_from_layout(page: ParsedPage) -> str:
    """Guess the title from page 1 typography.

    The title is normally the largest type on the first page and often wraps
    across two or three lines, so the whole block is taken and its internal line
    breaks collapsed — truncating at the first newline loses half the title.
    """
    candidates = [
        b
        for b in page.blocks
        if b.font_size > 0 and 8 <= len(b.text.strip()) <= 300 and b.text.count("\n") <= 3
    ]
    if not candidates:
        for line in page.text.split("\n"):
            stripped = line.strip()
            if 8 <= len(stripped) <= 200:
                return stripped
        return ""

    largest = max(candidates, key=lambda b: b.font_size)
    text = " ".join(part.strip() for part in largest.text.split("\n") if part.strip())
    if text.lower().startswith(("arxiv", "preprint", "under review", "published as")):
        remaining = [b for b in candidates if b is not largest]
        if remaining:
            nxt = max(remaining, key=lambda b: b.font_size)
            text = " ".join(part.strip() for part in nxt.text.split("\n") if part.strip())
    return text


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _document_metadata(doc: pymupdf.Document, pages: list[ParsedPage]) -> dict:
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip() or None
    author_field = (meta.get("author") or "").strip()
    authors = [a.strip() for a in re.split(r",| and |;", author_field) if a.strip()]

    first_page_text = pages[0].text if pages else ""
    if not title and pages:
        title = _title_from_layout(pages[0]) or None

    doi_match = _DOI_RE.search(first_page_text)
    year_match = _YEAR_RE.search(first_page_text)
    return {
        "title": title,
        "authors": authors,
        "doi": doi_match.group(0) if doi_match else None,
        "year": int(year_match.group(0)) if year_match else None,
    }


def parse_pdf(
    path: Path,
    *,
    ocr_engine=None,
    ocr_min_chars: int = 180,
    max_bytes: int = 50 * 1024 * 1024,
) -> ParsedDocument:
    """Extract every page of ``path``, falling back to OCR where needed.

    ``ocr_engine`` is any object implementing :class:`~paperpilot.ingestion.ocr.OCREngine`.
    Passing ``None`` disables OCR entirely; pages that need it keep whatever
    native text exists and record a reduced confidence.
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

            if needs_ocr(native_text, min_chars=ocr_min_chars) and ocr_engine is not None:
                ocr_result = ocr_engine.extract_page(page, page_number=page_number)
                if ocr_result is not None and len(ocr_result.text) > len(native_text):
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
                    info("ingestion.ocr_page", page=page_number, chars=len(parsed.text))
            elif needs_ocr(native_text, min_chars=ocr_min_chars):
                # No OCR available: be explicit rather than pretending the page is empty.
                parsed.confidence = 0.4

            pages.append(parsed)

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


def needs_ocr(text: str, *, min_chars: int = 180) -> bool:
    """Decide whether a page's native text layer is too thin to trust.

    Deliberately non-LLM and cheap: a page is a candidate for OCR when it holds
    fewer than ``min_chars`` usable characters, or when what it does hold is
    mostly non-alphabetic (a symptom of a scanned page with stray artefacts).
    """
    stripped = (text or "").strip()
    if len(stripped) < min_chars:
        return True
    letters = sum(1 for c in stripped if c.isalpha())
    return (letters / len(stripped)) < 0.5
