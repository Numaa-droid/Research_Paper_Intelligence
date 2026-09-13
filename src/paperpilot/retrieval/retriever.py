"""Hybrid retriever: dense + lexical, fused, filtered, optionally reranked.

Agents never see the whole PDF. They ask this retriever targeted questions and
receive a bounded set of chunks that already carry page, section and element
metadata, which is what makes the citations in the final report checkable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..domain.enums import ElementType
from ..domain.models import PaperChunk
from ..observability import debug
from .index import PaperIndex

#: Reciprocal-rank-fusion constant. 60 is the value from the original RRF paper
#: and behaves well when one of the two rankers returns nothing.
RRF_K = 60


@dataclass(slots=True)
class RetrievalResult:
    """One retrieved chunk with its fused score and per-ranker provenance."""

    chunk: PaperChunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None
    matched_by: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "page": self.chunk.page,
            # The most specific heading, so a citation reads "§4.2 Main Results"
            # rather than the parent "§4 Experiments".
            "section": self.chunk.citation_section,
            "element_type": self.chunk.element_type.value,
            "element_id": self.chunk.element_id,
            "text": self.chunk.text,
            "source": self.chunk.source.value,
            "score": round(self.score, 6),
            "matched_by": self.matched_by,
        }


@dataclass(slots=True)
class RetrievalQuery:
    """A targeted request for evidence."""

    text: str
    top_k: int = 8
    pages: list[int] | None = None
    sections: list[str] | None = None
    element_types: list[ElementType] | None = None
    mode: str = "hybrid"  # "hybrid" | "dense" | "lexical"
    #: Pull in sibling chunks of any retrieved element (a table's caption and its
    #: body are separate chunks, and an agent asking about results needs both).
    expand_elements: bool = True


class Reranker(ABC):
    """Optional second-stage ranking."""

    name = "abstract"

    @abstractmethod
    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]: ...


class FlashRankReranker(Reranker):  # pragma: no cover - optional dependency
    """Cross-encoder reranking via FlashRank."""

    name = "flashrank"

    def __init__(self, model: str | None = None) -> None:
        from flashrank import Ranker

        self._ranker = Ranker(model_name=model) if model else Ranker()

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        from flashrank import RerankRequest

        passages = [{"id": r.chunk_id, "text": r.chunk.text} for r in results]
        ranked = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        order = {item["id"]: float(item["score"]) for item in ranked}
        for result in results:
            result.rerank_score = order.get(result.chunk_id)
            if result.rerank_score is not None:
                result.score = result.rerank_score
        return sorted(results, key=lambda r: -r.score)


class Retriever(ABC):
    """Retrieval interface. Agents depend on this, never on a concrete index."""

    @abstractmethod
    def search(self, query: RetrievalQuery) -> list[RetrievalResult]: ...

    @abstractmethod
    def semantic_search(self, text: str, *, top_k: int = 8) -> list[RetrievalResult]: ...

    @abstractmethod
    def keyword_search(self, text: str, *, top_k: int = 8) -> list[RetrievalResult]: ...


class HybridRetriever(Retriever):
    """Dense + BM25 with reciprocal rank fusion and metadata filtering.

    Filters are applied *after* ranking but *before* truncation, so asking for
    "tables only" returns the top-k tables rather than whatever tables happened
    to survive a global top-k cut.
    """

    def __init__(
        self,
        index: PaperIndex,
        *,
        default_top_k: int = 8,
        reranker: Reranker | None = None,
        candidate_multiplier: int = 4,
    ) -> None:
        self.index = index
        self.default_top_k = default_top_k
        self.reranker = reranker
        self.candidate_multiplier = candidate_multiplier

    # --- public API -------------------------------------------------------

    def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        top_k = query.top_k or self.default_top_k
        pool = max(top_k * self.candidate_multiplier, top_k)

        dense = self._dense(query.text, pool) if query.mode in ("hybrid", "dense") else []
        lexical = self._lexical(query.text, pool) if query.mode in ("hybrid", "lexical") else []

        fused = self._fuse(dense, lexical)
        allowed = self._allowed_ids(query)
        if allowed is not None:
            fused = [r for r in fused if r.chunk_id in allowed]

        if not fused and allowed:
            # The query matched nothing lexically or semantically but the filter
            # itself is meaningful (e.g. "give me every equation"); fall back to
            # returning the filtered chunks in document order rather than [].
            fused = [
                RetrievalResult(chunk=self.index.require_chunk(cid), score=0.0, matched_by=["filter"])
                for cid in allowed
            ]
            fused.sort(key=lambda r: (r.chunk.page, r.chunk.chunk_id))

        if query.expand_elements:
            fused = self._expand_elements(fused, limit=top_k)

        if self.reranker is not None and fused:
            fused = self.reranker.rerank(query.text, fused[:pool])

        debug(
            "retrieval.query",
            query=query.text[:120],
            mode=query.mode,
            top_k=top_k,
            dense_hits=len(dense),
            lexical_hits=len(lexical),
            returned=min(len(fused), top_k),
        )
        return fused[:top_k]

    def semantic_search(self, text: str, *, top_k: int = 8) -> list[RetrievalResult]:
        return self.search(RetrievalQuery(text=text, top_k=top_k, mode="dense"))

    def keyword_search(self, text: str, *, top_k: int = 8) -> list[RetrievalResult]:
        return self.search(RetrievalQuery(text=text, top_k=top_k, mode="lexical"))

    def exact_term(self, term: str, *, top_k: int = 8) -> list[RetrievalResult]:
        """Chunks literally containing every token of ``term``."""
        ids = self.index.bm25.exact_term_search(term)[:top_k]
        return [
            RetrievalResult(chunk=self.index.require_chunk(cid), score=1.0, matched_by=["exact"])
            for cid in ids
        ]

    # --- internals --------------------------------------------------------

    def _dense(self, text: str, pool: int) -> list[tuple[str, float]]:
        if self.index.vectors is None:
            return []
        return self.index.vectors.search(text, top_k=pool)

    def _lexical(self, text: str, pool: int) -> list[tuple[str, float]]:
        return self.index.bm25.search(text, top_k=pool)

    def _fuse(
        self, dense: list[tuple[str, float]], lexical: list[tuple[str, float]]
    ) -> list[RetrievalResult]:
        merged: dict[str, RetrievalResult] = {}

        def ensure(chunk_id: str) -> RetrievalResult | None:
            if chunk_id in merged:
                return merged[chunk_id]
            chunk = self.index.get(chunk_id)
            if chunk is None:
                return None
            merged[chunk_id] = RetrievalResult(chunk=chunk, score=0.0)
            return merged[chunk_id]

        for rank, (chunk_id, score) in enumerate(dense, start=1):
            result = ensure(chunk_id)
            if result is None:
                continue
            result.dense_score = score
            result.dense_rank = rank
            result.score += 1.0 / (RRF_K + rank)
            result.matched_by.append("dense")

        for rank, (chunk_id, score) in enumerate(lexical, start=1):
            result = ensure(chunk_id)
            if result is None:
                continue
            result.lexical_score = score
            result.lexical_rank = rank
            result.score += 1.0 / (RRF_K + rank)
            result.matched_by.append("lexical")

        ordered = sorted(merged.values(), key=lambda r: (-r.score, r.chunk.page, r.chunk_id))
        return ordered

    def _expand_elements(
        self, results: list[RetrievalResult], *, limit: int
    ) -> list[RetrievalResult]:
        """Insert sibling chunks of retrieved elements next to their sibling.

        A table is split across a caption chunk and a body chunk. The caption
        wins on lexical similarity (it is prose) while the body holds the
        numbers, so retrieving one without the other hands the agent a citation
        it cannot actually read a value out of.
        """
        seen = {r.chunk_id for r in results}
        expanded: list[RetrievalResult] = []

        for result in results:
            expanded.append(result)
            element_id = result.chunk.element_id
            if not element_id or len(expanded) >= limit * 2:
                continue
            for sibling in self.index.by_element_id(element_id):
                if sibling.chunk_id in seen:
                    continue
                seen.add(sibling.chunk_id)
                expanded.append(
                    RetrievalResult(
                        chunk=sibling,
                        # Just below its sibling, so ordering stays intuitive.
                        score=result.score * 0.999,
                        matched_by=[*result.matched_by, "element_sibling"],
                    )
                )
        return expanded

    def _allowed_ids(self, query: RetrievalQuery) -> set[str] | None:
        if not (query.pages or query.sections or query.element_types):
            return None
        filtered = self.index.filter(
            pages=query.pages,
            sections=query.sections,
            element_types=query.element_types,
        )
        return {c.chunk_id for c in filtered}


def build_retriever(
    index: PaperIndex, *, top_k: int = 8, rerank_enabled: bool = False
) -> HybridRetriever:
    reranker: Reranker | None = None
    if rerank_enabled:
        try:
            reranker = FlashRankReranker()
        except Exception as exc:  # noqa: BLE001 - reranking is a nice-to-have
            from ..observability import warning

            warning("retrieval.rerank_unavailable", error=str(exc)[:200])
            reranker = None
    return HybridRetriever(index, default_top_k=top_k, reranker=reranker)
