"""Ingestion layer: PDF parsing, OCR fallback, structure detection, chunking."""

from .chunker import chunk_document, element_counts
from .elements import classify_block, looks_like_equation, looks_like_heading, looks_like_table
from .layout import (
    LayoutEngine,
    LayoutRegion,
    NullLayoutEngine,
    PageLayout,
    PPStructureLayoutEngine,
    build_layout_engine,
)
from .ocr import (
    NullOCREngine,
    OCREngine,
    OCRResult,
    PaddleOCRVLEngine,
    build_ocr_engine,
)
from .pdf_parser import (
    ParsedDocument,
    ParsedPage,
    TextBlock,
    extraction_failures,
    needs_ocr,
    normalize_text,
    parse_pdf,
    validate_pdf_file,
)
from .service import IngestionResult, IngestionService, new_document_id

__all__ = [
    "IngestionResult",
    "IngestionService",
    "LayoutEngine",
    "LayoutRegion",
    "NullLayoutEngine",
    "NullOCREngine",
    "OCREngine",
    "OCRResult",
    "PPStructureLayoutEngine",
    "PaddleOCRVLEngine",
    "PageLayout",
    "ParsedDocument",
    "ParsedPage",
    "TextBlock",
    "build_layout_engine",
    "build_ocr_engine",
    "chunk_document",
    "classify_block",
    "element_counts",
    "extraction_failures",
    "looks_like_equation",
    "looks_like_heading",
    "looks_like_table",
    "needs_ocr",
    "new_document_id",
    "normalize_text",
    "parse_pdf",
    "validate_pdf_file",
]
