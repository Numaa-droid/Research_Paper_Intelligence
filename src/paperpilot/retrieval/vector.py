"""Dense vector index.

The default backend is a small NumPy cosine-similarity store: for a single paper
(hundreds of chunks, not millions) an exact brute-force search is faster than
building an ANN structure, has no extra dependency, and returns exactly correct
neighbours. :class:`FaissVectorIndex` implements the same interface for anyone
who wants to scale past one paper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..errors import RetrievalError
from ..providers.embeddings import EmbeddingProvider


class VectorIndex(ABC):
    """Interface for dense retrieval backends."""

    name = "abstract"

    @abstractmethod
    def build(self, doc_ids: list[str], texts: list[str]) -> None: ...

    @abstractmethod
    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]: ...

    @property
    @abstractmethod
    def size(self) -> int: ...


class NumpyVectorIndex(VectorIndex):
    """Exact cosine similarity over L2-normalised embeddings."""

    name = "numpy"

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder
        self._doc_ids: list[str] = []
        self._matrix: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def build(self, doc_ids: list[str], texts: list[str]) -> None:
        if len(doc_ids) != len(texts):
            raise RetrievalError("doc_ids and texts must be the same length")
        self._doc_ids = list(doc_ids)
        if not texts:
            self._matrix = None
            return
        vectors = self.embedder.embed(texts)
        self._matrix = np.asarray(vectors, dtype=np.float32)

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        if self._matrix is None or not self._doc_ids:
            return []
        query_vector = np.asarray(self.embedder.embed_one(query), dtype=np.float32)
        if query_vector.shape[0] != self._matrix.shape[1]:
            raise RetrievalError(
                "query embedding dimension does not match the index; "
                "the index must be rebuilt after changing embedding provider"
            )
        # Both sides are already L2-normalised, so a dot product is the cosine.
        scores = self._matrix @ query_vector
        top_k = min(top_k, len(self._doc_ids))
        best = np.argpartition(-scores, range(top_k))[:top_k]
        return [(self._doc_ids[int(i)], float(scores[int(i)])) for i in best if scores[int(i)] > 0]


class FaissVectorIndex(VectorIndex):  # pragma: no cover - optional dependency
    """FAISS-backed index for corpora large enough to need one."""

    name = "faiss"

    def __init__(self, embedder: EmbeddingProvider) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:
            raise RetrievalError(
                "faiss-cpu is not installed; install the 'faiss' extra or use the numpy index"
            ) from exc
        self.embedder = embedder
        self._doc_ids: list[str] = []
        self._index = None

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def build(self, doc_ids: list[str], texts: list[str]) -> None:
        import faiss

        self._doc_ids = list(doc_ids)
        if not texts:
            self._index = None
            return
        matrix = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        if self._index is None:
            return []
        vector = np.asarray([self.embedder.embed_one(query)], dtype=np.float32)
        scores, indices = self._index.search(vector, min(top_k, len(self._doc_ids)))
        return [
            (self._doc_ids[int(i)], float(s))
            for s, i in zip(scores[0], indices[0], strict=False)
            if i >= 0 and s > 0
        ]


def build_vector_index(embedder: EmbeddingProvider, *, backend: str = "numpy") -> VectorIndex:
    if backend == "numpy":
        return NumpyVectorIndex(embedder)
    if backend == "faiss":
        return FaissVectorIndex(embedder)
    raise RetrievalError(f"unknown vector backend: {backend}")
