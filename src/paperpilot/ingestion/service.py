"""Ingestion service: PDF in, queryable :class:`PaperIndex` out."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..domain.enums import ElementType
from ..domain.models import PaperChunk, PaperMetadata
from ..observability import info, timed
from ..providers.embeddings import EmbeddingProvider
from ..retrieval.index import PaperIndex
from .chunker import chunk_document, element_counts
from .ocr import OCREngine, build_ocr_engine
from .pdf_parser import ParsedDocument, parse_pdf

_ABSTRACT_RE = re.compile(
    r"abstract\b[\s:.—-]*(.{60,2500}?)(?:\n\s*(?:1\s|I\.|introduction|keywords|index terms)\b)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True)
class IngestionResult:
    document_id: str
    index: PaperIndex
    metadata: PaperMetadata
    chunks: list[PaperChunk]
    parsed: ParsedDocument

    @property
    def page_count(self) -> int:
        return self.metadata.page_count


def new_document_id(prefix: str = "paper") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _extract_abstract(parsed: ParsedDocument, chunks: list[PaperChunk]) -> str | None:
    """Prefer an explicitly detected Abstract section; fall back to a regex."""
    for chunk in chunks:
        if chunk.section and chunk.section.strip().lower().startswith("abstract"):
            return chunk.text.strip()[:2500]
    head = "\n".join(p.text for p in parsed.pages[:2])
    match = _ABSTRACT_RE.search(head)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return None


def build_metadata(
    *,
    document_id: str,
    parsed: ParsedDocument,
    chunks: list[PaperChunk],
    sections: list[str],
) -> PaperMetadata:
    counts = element_counts(chunks)
    return PaperMetadata(
        document_id=document_id,
        title=parsed.title,
        authors=parsed.authors,
        abstract=_extract_abstract(parsed, chunks),
        year=parsed.year,
        doi=parsed.doi,
        filename=parsed.filename,
        page_count=parsed.page_count,
        chunk_count=len(chunks),
        ocr_page_count=parsed.ocr_page_count,
        page_numbering="pdf_index",
        sections=sections,
        element_counts=counts,
        equation_count=counts.get(ElementType.EQUATION.value, 0),
        table_count=counts.get(ElementType.TABLE.value, 0),
        figure_count=counts.get(ElementType.FIGURE.value, 0),
        has_equations=counts.get(ElementType.EQUATION.value, 0) > 0,
        has_tables=counts.get(ElementType.TABLE.value, 0) > 0,
        has_figures=counts.get(ElementType.FIGURE.value, 0) > 0,
        file_size_bytes=parsed.file_size_bytes,
        sha256=parsed.sha256,
    )


class IngestionService:
    """Parse, structure, chunk and index one PDF."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: EmbeddingProvider | None = None,
        ocr_engine: OCREngine | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.ocr_engine = ocr_engine or build_ocr_engine(
            enabled=settings.ocr_enabled, language=settings.ocr_language
        )

    def ingest(self, file_path: str | Path, *, document_id: str | None = None) -> IngestionResult:
        path = Path(file_path)
        document_id = document_id or new_document_id()

        with timed("ingestion", document_id=document_id, filename=path.name):
            parsed = parse_pdf(
                path,
                ocr_engine=self.ocr_engine if self.ocr_engine.available else None,
                ocr_min_chars=self.settings.ocr_min_chars_per_page,
                max_bytes=self.settings.max_upload_bytes,
            )
            chunks, sections = chunk_document(
                parsed,
                document_id=document_id,
                max_chars=self.settings.chunk_max_chars,
                overlap=self.settings.chunk_overlap_chars,
            )
            metadata = build_metadata(
                document_id=document_id, parsed=parsed, chunks=chunks, sections=sections
            )
            index = PaperIndex.build(
                document_id=document_id,
                metadata=metadata,
                chunks=chunks,
                embedder=self.embedder,
            )

        info(
            "ingestion.summary",
            document_id=document_id,
            pages=metadata.page_count,
            chunks=metadata.chunk_count,
            ocr_pages=metadata.ocr_page_count,
            sections=len(sections),
            equations=metadata.equation_count,
            tables=metadata.table_count,
            figures=metadata.figure_count,
        )
        return IngestionResult(
            document_id=document_id,
            index=index,
            metadata=metadata,
            chunks=chunks,
            parsed=parsed,
        )
