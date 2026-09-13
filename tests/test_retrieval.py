"""BM25, dense retrieval, hybrid fusion and metadata filtering."""

from __future__ import annotations

import pytest

from paperpilot.domain.enums import ElementType
from paperpilot.errors import RetrievalError
from paperpilot.providers.embeddings import HashingEmbeddingProvider
from paperpilot.retrieval.bm25 import BM25Index
from paperpilot.retrieval.retriever import HybridRetriever, RetrievalQuery
from paperpilot.retrieval.vector import NumpyVectorIndex

CORPUS = [
    ("a", "sparse routing transformers for long document question answering"),
    ("b", "the router is trained with a load balancing loss and a straight through estimator"),
    ("c", "we evaluate on NarrativeQA and QuALITY reporting F1 and accuracy"),
    ("d", "unrelated text about photosynthesis in tropical plants"),
]


# --- BM25 ------------------------------------------------------------------


def test_bm25_ranks_the_relevant_document_first():
    index = BM25Index().fit(CORPUS)
    results = index.search("load balancing loss router", top_k=3)
    assert results[0][0] == "b"


def test_bm25_ignores_documents_with_no_shared_terms():
    index = BM25Index().fit(CORPUS)
    ids = [doc_id for doc_id, _ in index.search("NarrativeQA QuALITY", top_k=4)]
    assert "d" not in ids


def test_bm25_scores_are_positive_and_ordered():
    index = BM25Index().fit(CORPUS)
    scores = [score for _, score in index.search("routing transformers", top_k=4)]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_bm25_empty_query_returns_nothing():
    assert BM25Index().fit(CORPUS).search("", top_k=3) == []


def test_bm25_on_empty_corpus_is_safe():
    assert BM25Index().fit([]).search("anything", top_k=3) == []


def test_bm25_exact_term_search_requires_every_token():
    index = BM25Index().fit(CORPUS)
    assert index.exact_term_search("photosynthesis") == ["d"]
    assert index.exact_term_search("photosynthesis routing") == []


def test_bm25_is_deterministic():
    first = BM25Index().fit(CORPUS).search("routing loss", top_k=4)
    second = BM25Index().fit(CORPUS).search("routing loss", top_k=4)
    assert first == second


# --- dense retrieval -------------------------------------------------------


def test_hashing_embeddings_are_normalised_and_deterministic():
    embedder = HashingEmbeddingProvider(dim=128)
    first = embedder.embed_one("sparse routing transformer")
    second = embedder.embed_one("sparse routing transformer")
    assert first == second
    assert pytest.approx(sum(v * v for v in first), rel=1e-6) == 1.0


def test_vector_index_finds_the_nearest_document():
    index = NumpyVectorIndex(HashingEmbeddingProvider(dim=256))
    index.build([i for i, _ in CORPUS], [t for _, t in CORPUS])
    results = index.search("photosynthesis tropical plants", top_k=2)
    assert results and results[0][0] == "d"


def test_vector_index_reports_its_size():
    index = NumpyVectorIndex(HashingEmbeddingProvider(dim=64))
    index.build([i for i, _ in CORPUS], [t for _, t in CORPUS])
    assert index.size == 4


def test_vector_index_on_empty_corpus_is_safe():
    index = NumpyVectorIndex(HashingEmbeddingProvider(dim=64))
    index.build([], [])
    assert index.search("anything") == []


def test_vector_index_rejects_mismatched_inputs():
    index = NumpyVectorIndex(HashingEmbeddingProvider(dim=64))
    with pytest.raises(RetrievalError):
        index.build(["a", "b"], ["only one text"])


# --- the paper index -------------------------------------------------------


def test_index_reports_accurate_stats(index, expected):
    stats = index.stats()
    assert stats["pages"] == expected["page_count"]
    assert stats["chunks"] == len(index.chunks)
    assert stats["element_counts"]["equation"] == 3


def test_index_page_bounds(index, expected):
    assert index.has_page(1) is True
    assert index.has_page(expected["page_count"]) is True
    assert index.has_page(expected["page_count"] + 1) is False
    assert index.has_page(0) is False


def test_index_lookup_by_element_id(index):
    chunks = index.by_element_id("table-1")
    assert len(chunks) >= 2
    assert all(c.element_id == "table-1" for c in chunks)


def test_index_require_chunk_raises_for_unknown_id(index):
    with pytest.raises(RetrievalError):
        index.require_chunk("does-not-exist")


# --- filtering -------------------------------------------------------------


def test_filter_by_page(index):
    assert all(c.page == 3 for c in index.filter(pages=[3]))


def test_filter_by_element_type(index):
    tables = index.filter(element_types=[ElementType.TABLE])
    assert tables and all(c.element_type is ElementType.TABLE for c in tables)


def test_filter_by_section_matches_subsections_too(index):
    results = index.filter(sections=["4.2"])
    assert results
    assert all("4.2" in (c.subsection or c.section or "") for c in results)


def test_filters_combine(index):
    results = index.filter(pages=[4], element_types=[ElementType.TABLE])
    assert results and all(c.page == 4 for c in results)


# --- hybrid retrieval ------------------------------------------------------


def test_hybrid_retrieval_finds_the_results_table(retriever):
    results = retriever.search(
        RetrievalQuery(text="main results comparison", element_types=[ElementType.TABLE], top_k=5)
    )
    assert results
    assert any("76.2" in r.chunk.text for r in results)


def test_element_cohesion_pulls_in_the_table_body(retriever):
    """Retrieving a caption must also surface the numbers it describes."""
    results = retriever.search(RetrievalQuery(text="Table 1 main results", top_k=6))
    ids = {r.chunk.element_id for r in results}
    assert "table-1" in ids
    assert any("81.4" in r.chunk.text for r in results)


def test_retrieval_respects_top_k(retriever):
    assert len(retriever.search(RetrievalQuery(text="routing", top_k=3))) <= 3


def test_page_filter_constrains_results(retriever):
    results = retriever.search(RetrievalQuery(text="anything at all", pages=[6], top_k=5))
    assert results and all(r.chunk.page == 6 for r in results)


def test_filter_with_no_lexical_match_still_returns_the_filtered_set(retriever):
    """Asking for 'every equation' must work even when the query text matches nothing."""
    results = retriever.search(
        RetrievalQuery(
            text="zzzz nonexistent query tokens",
            element_types=[ElementType.EQUATION],
            top_k=10,
        )
    )
    assert len(results) == 3
    assert all(r.chunk.element_type is ElementType.EQUATION for r in results)


def test_results_record_which_ranker_matched(retriever):
    results = retriever.search(RetrievalQuery(text="load balancing loss", top_k=5))
    assert results
    assert all(r.matched_by for r in results)
    assert any("lexical" in r.matched_by or "dense" in r.matched_by for r in results)


def test_semantic_and_keyword_search_are_available(retriever):
    assert retriever.semantic_search("routing objective", top_k=3) is not None
    assert retriever.keyword_search("NarrativeQA", top_k=3) is not None


def test_exact_term_lookup_finds_a_number(retriever):
    results = retriever.exact_term("81.4", top_k=5)
    assert results and any("81.4" in r.chunk.text for r in results)


def test_retrieval_result_dict_uses_the_specific_section(retriever):
    results = retriever.search(RetrievalQuery(text="main results table", top_k=4))
    payload = next(r.to_dict() for r in results if r.chunk.element_id == "table-1")
    assert payload["section"] == "4.2 Main Results"


def test_retriever_without_a_vector_index_still_works(index):
    index.vectors = None
    retriever = HybridRetriever(index, default_top_k=5)
    results = retriever.search(RetrievalQuery(text="load balancing loss", top_k=5))
    assert results and all("lexical" in r.matched_by for r in results)
