"""Embedding-provider abstraction.

The default ``hashing`` provider is deterministic, dependency-free and offline:
it is not competitive with a trained encoder, but it makes dense retrieval
reproducible in tests and keeps the system usable with no credentials. Swap in
``openai_compatible`` or ``sentence_transformers`` via configuration for real
semantic search.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from abc import ABC, abstractmethod

from ..config import Settings
from ..errors import ProviderError

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
# Process-wide: the accelerator device is shared by every model instance.
TORCH_INFERENCE_LOCK = threading.Lock()


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer shared by BM25 and the hashing embedder."""
    return _TOKEN_RE.findall((text or "").lower())


class EmbeddingProvider(ABC):
    """Interface for turning text into dense vectors."""

    name = "abstract"
    dim = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; the result is L2-normalised."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashed bag-of-ngrams embedding.

    Tokens are hashed into a fixed number of buckets with a signed hash (so
    collisions cancel rather than accumulate), weighted sub-linearly by term
    frequency, and L2-normalised. Word bigrams are included so short queries
    retain a little word-order signal.
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ProviderError("embedding_dim must be positive")
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            tokens = tokenize(text)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for a, b in zip(tokens, tokens[1:], strict=False):
                bigram = f"{a}_{b}"
                counts[bigram] = counts.get(bigram, 0) + 1

            vector = [0.0] * self.dim
            for token, count in counts.items():
                index, sign = self._bucket(token)
                vector[index] += sign * (1.0 + math.log(count))

            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
            vectors.append(vector)
        return vectors


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embeddings from any OpenAI-compatible ``/embeddings`` endpoint."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        dim: int = 1536,
        timeout: float = 60.0,
        batch_size: int = 64,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout = timeout
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        out: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                response = client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": batch},
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
                for item in sorted(payload["data"], key=lambda d: d.get("index", 0)):
                    out.append(_normalize(item["embedding"]))
        if out:
            self.dim = len(out[0])
        return out


class SentenceTransformersProvider(EmbeddingProvider):
    """Local encoder via ``sentence-transformers`` (optional extra)."""

    name = "sentence_transformers"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderError(
                "sentence-transformers is not installed; "
                "install the 'embeddings' extra or use PAPERPILOT_EMBEDDING_PROVIDER=hashing"
            ) from exc
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - optional
        # Torch's MPS (and CUDA) backends are not safe for concurrent inference
        # from multiple threads; parallel specialist agents otherwise abort the
        # process with a Metal command-buffer assertion.
        with TORCH_INFERENCE_LOCK:
            vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm == 0:
        return [float(v) for v in vector]
    return [float(v) / norm for v in vector]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Instantiate the configured embedding provider."""
    provider = settings.embedding_provider
    if provider == "hashing":
        return HashingEmbeddingProvider(dim=settings.embedding_dim)
    if provider == "openai_compatible":
        base_url = settings.embedding_base_url or settings.llm_base_url
        return OpenAICompatibleEmbeddingProvider(
            base_url=base_url,
            api_key=settings.embedding_api_key or settings.llm_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    if provider == "sentence_transformers":
        return SentenceTransformersProvider(model=settings.embedding_model)
    raise ProviderError(f"unknown embedding provider: {provider}")
