"""Symbol definitions are read from the paper's text, so "not defined" is checked."""

from __future__ import annotations

import pytest

from paperpilot.domain.models import PaperChunk, PaperMetadata
from paperpilot.retrieval.index import PaperIndex
from paperpilot.validation.symbols import PLACEHOLDER, SymbolDefinitions, symbol_key

PAGES = {
    3: "Sub-layers produce outputs of dimension dmodel = 512.",
    4: "The input consists of queries and keys of dimension dk, and values of dimension dv.",
    5: (
        "Where the projections are parameter matrices W Q i ∈R^dmodel×dk, W K i ∈R^dmodel×dk "
        "and W O ∈R^hdv×dmodel. In this work we employ h = 8 parallel attention layers. "
        "For each of these we use dk = dv = dmodel/h = 64. The dimensionality of input and "
        "output is dmodel = 512. We convert tokens to vectors of dimension dmodel. "
        "Their dot product, q · k = P^dk i=1 q^ik^i, has mean 0."
    ),
    9: "We trained a 4-layer transformer with dmodel = 1024 on the WSJ.",
}


@pytest.fixture(scope="module")
def defs():
    chunks = [
        PaperChunk(chunk_id=f"c{p}", document_id="d", page=p, text=text) for p, text in PAGES.items()
    ]
    index = PaperIndex(document_id="d", metadata=PaperMetadata(document_id="d"), chunks=chunks)
    index._reindex()
    return SymbolDefinitions(index)


@pytest.mark.parametrize(
    ("symbol", "meaning", "expected"),
    [
        ("WQ_i", "Projection matrix for queries of head i", "∈ ℝ^dmodel×dk (p. 5)"),
        ("W^{O}", "Output projection matrix", "∈ ℝ^hdv×dmodel (p. 5)"),
        ("h", "Number of attention heads", "h = 8 (p. 5)"),
        ("d_k", "Dimension of the keys", "dk = dv = dmodel/h = 64 (p. 5)"),
        ("Q", "Queries matrix", "dimension dk (p. 4)"),
        ("V", "Values matrix", "dimension dv (p. 4)"),
        ("x", "Input vector at a position", "dimension dmodel (p. 5)"),
    ],
)
def test_the_papers_own_definition_is_found(defs, symbol, meaning, expected):
    assert defs.lookup(symbol, meaning).describe() == expected


def test_a_symbol_defined_twice_shows_every_value(defs):
    assert defs.lookup("d_model", "Model dimension").describe() == (
        "dmodel = 512 (p. 3); also dmodel = 1024 (p. 9)"
    )


@pytest.mark.parametrize(
    ("symbol", "meaning"),
    [
        ("b1", "First bias vector"),       # "vector" alone ties it to nothing
        ("W1", "First weight matrix"),
        ("i", "Summation index"),          # "i=1" in a sum is not a definition
        ("z", "Latent code"),
    ],
)
def test_symbols_the_paper_does_not_define_stay_undefined(defs, symbol, meaning):
    assert defs.lookup(symbol, meaning) is None


def test_placeholders_and_symbol_keys():
    assert PLACEHOLDER.match("not defined in the paper") and PLACEHOLDER.match("Not explicitly stated")
    assert not PLACEHOLDER.match("scalar") and not PLACEHOLDER.match("(d_model, d_ff)")
    assert symbol_key("W^Q_i") == symbol_key("W Q i") == symbol_key("WQ_{i}") == "wqi"
