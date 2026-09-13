"""Shared fixtures.

The whole suite runs offline against the deterministic mock provider: no network,
no credentials, no external services.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fixtures"))

from synthetic_paper import EXPECTED, build_pdf  # noqa: E402

from paperpilot.config import Settings  # noqa: E402
from paperpilot.ingestion.service import IngestionService  # noqa: E402
from paperpilot.observability import configure_logging  # noqa: E402
from paperpilot.providers.embeddings import HashingEmbeddingProvider  # noqa: E402
from paperpilot.providers.mock_llm import MockLLMProvider  # noqa: E402
from paperpilot.retrieval.retriever import build_retriever  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _quiet_logs():
    configure_logging(level="ERROR", fmt="text")


@pytest.fixture(scope="session")
def expected() -> dict:
    """Ground truth about the synthetic fixture paper."""
    return EXPECTED


@pytest.fixture(scope="session")
def paper_pdf(tmp_path_factory) -> Path:
    """The synthetic paper, built once per test session."""
    target = tmp_path_factory.mktemp("fixtures") / "synthetic_paper.pdf"
    return build_pdf(target)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointed at an isolated temp directory."""
    data_dir = tmp_path / "data"
    return Settings(
        data_dir=data_dir,
        database_url=f"sqlite:///{data_dir / 'paperpilot.sqlite'}",
        llm_provider="mock",
        embedding_provider="hashing",
        embedding_dim=256,
        log_level="ERROR",
        ocr_enabled=False,
        max_retries=1,
    )


@pytest.fixture
def embedder() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(dim=256)


@pytest.fixture
def llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def ingested(settings, embedder, paper_pdf):
    """The fixture paper, parsed, chunked and indexed."""
    return IngestionService(settings, embedder=embedder).ingest(
        paper_pdf, document_id="paper-test"
    )


@pytest.fixture
def index(ingested):
    return ingested.index


@pytest.fixture
def retriever(index):
    return build_retriever(index, top_k=8)


@pytest.fixture
def agent_context(ingested, retriever):
    from paperpilot.agents.base import AgentContext

    return AgentContext(
        document_id=ingested.document_id,
        metadata=ingested.metadata,
        index=ingested.index,
        retriever=retriever,
        priority_sections=[],
    )


@pytest.fixture
def service(settings, embedder, llm):
    """A service wired to in-memory repositories and the mock provider."""
    from paperpilot.service import PaperPilotService
    from paperpilot.storage.sqlite_repo import (
        InMemoryAnalysisRepository,
        InMemoryDocumentRepository,
    )

    return PaperPilotService(
        settings,
        documents=InMemoryDocumentRepository(),
        analyses=InMemoryAnalysisRepository(),
        llm_provider=llm,
        embedding_provider=embedder,
    )


def _index_from_texts(texts: list[str], *, document_id: str = "paper-synthetic"):
    """Build a PaperIndex directly from plain text, bypassing the PDF layer."""
    from paperpilot.domain.models import PaperChunk, PaperMetadata
    from paperpilot.retrieval.index import PaperIndex

    chunks = [
        PaperChunk(
            chunk_id=f"{document_id}-page-{i}-chunk-01",
            document_id=document_id,
            page=i,
            section=f"{i} Section",
            text=text,
        )
        for i, text in enumerate(texts, start=1)
    ]
    metadata = PaperMetadata(
        document_id=document_id,
        title="Synthetic",
        page_count=len(chunks),
        chunk_count=len(chunks),
        sections=[c.section for c in chunks if c.section],
    )
    return PaperIndex.build(
        document_id=document_id,
        metadata=metadata,
        chunks=chunks,
        embedder=HashingEmbeddingProvider(dim=128),
    )


def _context_for(index):
    from paperpilot.agents.base import AgentContext

    return AgentContext(
        document_id=index.document_id,
        metadata=index.metadata,
        index=index,
        retriever=build_retriever(index, top_k=6),
        priority_sections=[],
    )


@pytest.fixture
def non_math_context():
    """A qualitative paper: no equations and no mathematical vocabulary."""
    return _context_for(
        _index_from_texts(
            [
                "This paper reports a qualitative interview study of software teams "
                "adopting code review practices in twelve organisations.",
                "We interviewed forty engineers and coded the transcripts thematically, "
                "reporting the themes that recurred across sites.",
                "Participants described review as a social process shaped by trust and "
                "by team history rather than by tooling.",
            ],
            document_id="paper-qualitative",
        )
    )


@pytest.fixture
def math_prose_context():
    """Equations failed to extract, but the prose is unmistakably mathematical."""
    return _context_for(
        _index_from_texts(
            [
                "We define the loss function over the training distribution and we "
                "minimize it with stochastic gradient descent.",
                "The gradient of the objective function is bounded, and we prove a "
                "theorem giving an upper bound on the optimization error.",
                "Our proof proceeds by taking the derivative of the regularization "
                "term and showing the expectation converges.",
            ],
            document_id="paper-math-prose",
        )
    )
