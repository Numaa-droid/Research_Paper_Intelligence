"""The paper intelligence index: the single source of truth about one paper.

Everything downstream — agents, the verifier, the citation formatter, the
deterministic validators — reads the paper through this object. Nothing else
touches the PDF again after ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.enums import ElementType
from ..domain.models import PaperChunk, PaperMetadata
from ..errors import RetrievalError
from ..observability import info
from ..providers.embeddings import EmbeddingProvider
from .bm25 import BM25Index
from .vector import VectorIndex, build_vector_index


@dataclass
class PaperIndex:
    """Chunk store + lexical index + dense index for a single document."""

    document_id: str
    metadata: PaperMetadata
    chunks: list[PaperChunk] = field(default_factory=list)
    bm25: BM25Index = field(default_factory=BM25Index)
    vectors: VectorIndex | None = None
    _by_id: dict[str, PaperChunk] = field(default_factory=dict, repr=False)
    _by_page: dict[int, list[PaperChunk]] = field(default_factory=dict, repr=False)

    # --- construction -----------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        document_id: str,
        metadata: PaperMetadata,
        chunks: list[PaperChunk],
        embedder: EmbeddingProvider | None = None,
        vector_backend: str = "numpy",
    ) -> PaperIndex:
        index = cls(document_id=document_id, metadata=metadata, chunks=list(chunks))
        index._reindex()

        pairs = [(c.chunk_id, c.text) for c in index.chunks]
        index.bm25 = BM25Index().fit(pairs)

        if embedder is not None and pairs:
            index.vectors = build_vector_index(embedder, backend=vector_backend)
            index.vectors.build([c for c, _ in pairs], [t for _, t in pairs])

        info(
            "index.built",
            document_id=document_id,
            chunks=len(index.chunks),
            pages=metadata.page_count,
            lexical_docs=index.bm25.size,
            vector_docs=index.vectors.size if index.vectors else 0,
        )
        return index

    def _reindex(self) -> None:
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self._by_page = {}
        for chunk in self.chunks:
            self._by_page.setdefault(chunk.page, []).append(chunk)

    # --- lookups ----------------------------------------------------------

    @property
    def page_count(self) -> int:
        return self.metadata.page_count or (max(self._by_page) if self._by_page else 0)

    def get(self, chunk_id: str) -> PaperChunk | None:
        return self._by_id.get(chunk_id)

    def page(self, page: int) -> list[PaperChunk]:
        return list(self._by_page.get(page, []))

    def page_text(self, page: int) -> str:
        return "\n".join(c.text for c in self._by_page.get(page, []))

    def full_text(self) -> str:
        return "\n".join(c.text for c in self.chunks)

    def sections(self) -> list[str]:
        """Every distinct section and subsection label, in document order."""
        seen: list[str] = []
        for chunk in self.chunks:
            for label in (chunk.section, chunk.subsection):
                if label and label not in seen:
                    seen.append(label)
        return seen

    def by_element_type(self, element_type: ElementType) -> list[PaperChunk]:
        return [c for c in self.chunks if c.element_type is element_type]

    def by_element_id(self, element_id: str) -> list[PaperChunk]:
        """Every chunk belonging to one element, e.g. a caption and its table body."""
        return [c for c in self.chunks if c.element_id == element_id]

    def has_page(self, page: int) -> bool:
        return 1 <= page <= self.page_count

    def filter(
        self,
        *,
        pages: list[int] | None = None,
        sections: list[str] | None = None,
        element_types: list[ElementType] | None = None,
    ) -> list[PaperChunk]:
        """Metadata-only filtering, with case-insensitive substring section match."""
        results = self.chunks
        if pages:
            wanted = set(pages)
            results = [c for c in results if c.page in wanted]
        if element_types:
            wanted_types = set(element_types)
            results = [c for c in results if c.element_type in wanted_types]
        if sections:
            needles = [s.lower() for s in sections]
            results = [
                c
                for c in results
                if any(
                    n in (label or "").lower()
                    for n in needles
                    for label in (c.section, c.subsection)
                )
            ]
        return list(results)

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for chunk in self.chunks:
            counts[chunk.element_type.value] = counts.get(chunk.element_type.value, 0) + 1
        return {
            "document_id": self.document_id,
            "pages": self.page_count,
            "chunks": len(self.chunks),
            "sections": len(self.sections()),
            "element_counts": counts,
        }

    def require_chunk(self, chunk_id: str) -> PaperChunk:
        chunk = self.get(chunk_id)
        if chunk is None:
            raise RetrievalError(f"unknown chunk_id: {chunk_id}")
        return chunk
