"""Okapi BM25 over the paper's chunks.

Implemented directly rather than pulled in as a dependency: it is ~60 lines,
deterministic, and easy to unit-test, and it shares its tokenizer with the
hashing embedder so lexical and dense scores see the same vocabulary.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..providers.embeddings import tokenize


@dataclass(slots=True)
class BM25Index:
    """A fitted BM25 index over a fixed corpus."""

    k1: float = 1.5
    b: float = 0.75
    doc_ids: list[str] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    term_freqs: list[Counter] = field(default_factory=list)
    doc_freqs: Counter = field(default_factory=Counter)
    avg_length: float = 0.0

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def fit(self, documents: list[tuple[str, str]]) -> BM25Index:
        """Fit over ``(doc_id, text)`` pairs."""
        self.doc_ids = []
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = Counter()

        for doc_id, text in documents:
            tokens = tokenize(text)
            counts = Counter(tokens)
            self.doc_ids.append(doc_id)
            self.doc_lengths.append(len(tokens))
            self.term_freqs.append(counts)
            self.doc_freqs.update(counts.keys())

        total = sum(self.doc_lengths)
        self.avg_length = total / len(self.doc_lengths) if self.doc_lengths else 0.0
        return self

    def _idf(self, term: str) -> float:
        n = self.size
        df = self.doc_freqs.get(term, 0)
        # BM25+ style floor keeps very common terms from going negative.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[tuple[str, float]]:
        """Score every document against ``query``; highest first."""
        terms = tokenize(query)
        if not terms or self.size == 0:
            return []

        idfs = {t: self._idf(t) for t in set(terms)}
        results: list[tuple[str, float]] = []
        for index, doc_id in enumerate(self.doc_ids):
            counts = self.term_freqs[index]
            length = self.doc_lengths[index] or 1
            score = 0.0
            for term in set(terms):
                freq = counts.get(term, 0)
                if not freq:
                    continue
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * (length / (self.avg_length or 1))
                )
                score += idfs[term] * (freq * (self.k1 + 1)) / denominator
            if score > 0:
                results.append((doc_id, score))

        results.sort(key=lambda item: (-item[1], item[0]))
        return results

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        return self.score(query)[:top_k]

    def exact_term_search(self, term: str) -> list[str]:
        """Documents literally containing ``term`` — used for numeric lookups."""
        needles = tokenize(term)
        if not needles:
            return []
        hits = []
        for index, doc_id in enumerate(self.doc_ids):
            counts = self.term_freqs[index]
            if all(counts.get(n, 0) > 0 for n in needles):
                hits.append(doc_id)
        return hits
