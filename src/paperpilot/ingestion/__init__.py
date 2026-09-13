"""Ingestion layer: PDF parsing, OCR fallback, structure detection, chunking."""

from .chunker import chunk_document, element_counts
from .elements import classify_block, looks_like_equation, looks_like_heading, looks_like_table
from .ocr import NullOCREngine, OCREngine, OCRResult, TesseractOCREngine, build_ocr_engine
from .pdf_parser import (
    ParsedDocument,
    ParsedPage,
    TextBlock,
    needs_ocr,
    normalize_text,
    parse_pdf,
    validate_pdf_file,
)
from .service import IngestionResult, IngestionService, new_document_id

__all__ = [
    "IngestionResult",
    "IngestionService",
    "NullOCREngine",
    "OCREngine",
    "OCRResult",
    "ParsedDocument",
    "ParsedPage",
    "TesseractOCREngine",
    "TextBlock",
    "build_ocr_engine",
    "chunk_document",
    "classify_block",
    "element_counts",
    "looks_like_equation",
    "looks_like_heading",
    "looks_like_table",
    "needs_ocr",
    "new_document_id",
    "normalize_text",
    "parse_pdf",
    "validate_pdf_file",
]
