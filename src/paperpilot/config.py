"""Environment-driven configuration.

Secrets are never hard-coded and never logged. See ``.env.example`` for the full
set of variables; every one has a default that keeps the system runnable offline
with the deterministic mock provider.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["mock", "openai_compatible", "huggingface"]
EmbeddingProviderName = Literal["hashing", "openai_compatible", "sentence_transformers"]


class Settings(BaseSettings):
    """Runtime settings loaded from the environment and/or a local ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="PAPERPILOT_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Model provider ---------------------------------------------------
    llm_provider: LLMProviderName = "mock"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    #: Sampling seed sent to ``openai_compatible`` endpoints. With temperature 0
    #: it makes repeated runs on the same paper return the same answers as far as
    #: the endpoint allows. Set empty to send none.
    llm_seed: int | None = 42
    #: Output budget per call. Agents return findings with verbatim quotes, which
    #: routinely exceeds 2K tokens; a response cut off early loses its last sections.
    llm_max_tokens: int = Field(default=8192, gt=0)
    #: ``openai_compatible`` only: a truncated response (``finish_reason ==
    #: "length"``) is re-requested with a doubled budget, up to this ceiling.
    #: Keep it within the model's own output limit (gpt-4o-mini: 16384).
    llm_max_tokens_ceiling: int = Field(default=16384, gt=0)
    llm_timeout_seconds: float = Field(default=90.0, gt=0)
    llm_max_attempts: int = Field(default=5, ge=1, le=8)
    #: Seconds before the first retry of a rate-limited or failed call; doubles
    #: on each further attempt. A server-sent ``Retry-After`` takes precedence.
    llm_backoff_seconds: float = Field(default=2.0, ge=0.0, le=60.0)
    #: Requests in flight at once, across every agent. Hosted APIs rate-limit per
    #: account, so parallel agents must share a budget rather than burst.
    llm_max_concurrency: int = Field(default=2, ge=1, le=16)
    #: ``reasoning_effort`` sent to ``openai_compatible`` endpoints; unset sends
    #: nothing. "none" stops thinking models (e.g. Qwen 3.5 on Ollama) spending
    #: the whole token budget on reasoning and returning empty content.
    llm_reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    #: Where the ``huggingface`` provider runs the model; "auto" picks CUDA, then
    #: Apple MPS, then CPU.
    llm_device: Literal["auto", "cpu", "mps", "cuda"] = "auto"

    # --- Embeddings -------------------------------------------------------
    embedding_provider: EmbeddingProviderName = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_dim: int = Field(default=512, gt=0)

    # --- Ingestion --------------------------------------------------------
    max_upload_mb: float = Field(default=50.0, gt=0)
    ocr_enabled: bool = True
    ocr_min_chars_per_page: int = Field(default=180, ge=0)
    ocr_language: str = "eng"
    #: "tesseract" (local binary) or "paddleocr_vl" (PaddleOCR-VL in llama-server).
    ocr_engine: Literal["tesseract", "paddleocr_vl"] = "tesseract"
    #: OpenAI-compatible endpoint of the llama-server running PaddleOCR-VL.
    ocr_base_url: str = "http://127.0.0.1:8080/v1"
    ocr_model: str = "PaddleOCR-VL-1.6"
    ocr_timeout_seconds: float = Field(default=300.0, gt=0)
    #: Also read charts and diagrams on text pages (engines that support it, i.e.
    #: paddleocr_vl). Text itself always comes from PyMuPDF on those pages.
    ocr_figures_enabled: bool = True
    #: Re-read tables on text pages to recover merged headers and cell order;
    #: the reading replaces the native text only when it agrees with it.
    ocr_tables_enabled: bool = True
    # --- Layout analysis --------------------------------------------------
    #: Run a document-layout model over each page and re-cut the text blocks
    #: along the regions it finds. Off by default: it is an optional dependency
    #: and rasterises every page, so it costs far more than the geometric
    #: heuristics that run otherwise.
    layout_enabled: bool = False
    layout_engine: Literal["pp_structure"] = "pp_structure"
    #: PP-Structure layout detector. Pinned, because a different model finds
    #: different regions and the parse must not change under the caller.
    layout_model: str = "PP-DocLayout_plus-L"
    layout_lang: str = "en"
    #: Rasterisation DPI. Detections are computed in pixels, so changing this
    #: changes the result; it is part of the layout cache key.
    layout_dpi: int = Field(default=200, ge=72, le=600)
    #: Regions scored below this are discarded.
    layout_min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    #: "cpu" keeps inference reproducible; GPU reduction order is not stable.
    layout_device: str = "cpu"
    #: Where analysed pages are cached, keyed by page image and engine
    #: configuration. Empty disables the cache; results are unchanged either way.
    layout_cache_dir: str = "data/layout_cache"

    chunk_max_chars: int = Field(default=1200, ge=200)
    chunk_overlap_chars: int = Field(default=150, ge=0)

    # --- Retrieval --------------------------------------------------------
    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    rerank_enabled: bool = False

    # --- Orchestration ----------------------------------------------------
    max_retries: int = Field(default=1, ge=0, le=3)
    verification_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    parallel_agents: bool = True
    math_min_equations: int = Field(default=3, ge=1)
    #: After the specialists, read every content chunk none of them was shown.
    coverage_enabled: bool = True
    #: Upper bound on sweep prompts; content past it is listed as unread.
    coverage_max_batches: int = Field(default=4, ge=0, le=20)

    # --- Storage ----------------------------------------------------------
    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./data/paperpilot.sqlite"

    # --- Observability ----------------------------------------------------
    log_level: str = "INFO"
    #: "auto" renders readable lines on a terminal and JSON when piped or shipped.
    log_format: Literal["auto", "pretty", "json", "text"] = "auto"
    #: Which HTTP request lines the server prints: "changes" hides the successful
    #: GET polling a browser repeats every second; "errors", "all" or "none".
    log_http: Literal["changes", "errors", "all", "none"] = "changes"
    log_document_text: bool = False

    @field_validator("llm_reasoning_effort", "llm_seed", mode="before")
    @classmethod
    def _blank_reasoning_effort_is_unset(cls, v):
        # `.env.example` ships the variable blank; that must mean "send nothing".
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("chunk_overlap_chars")
    @classmethod
    def _overlap_smaller_than_chunk(cls, v: int, info) -> int:
        max_chars = info.data.get("chunk_max_chars", 1200)
        if v >= max_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_max_chars")
        return v

    def ocr_engine_options(self) -> dict:
        """Keyword arguments for :func:`paperpilot.ingestion.ocr.build_ocr_engine`."""
        return {
            "enabled": self.ocr_enabled,
            "language": self.ocr_language,
            "engine": self.ocr_engine,
            "base_url": self.ocr_base_url,
            "model": self.ocr_model,
            "timeout": self.ocr_timeout_seconds,
        }

    @property
    def layout_engine_options(self) -> dict:
        """Keyword arguments for :func:`paperpilot.ingestion.layout.build_layout_engine`."""
        return {
            "enabled": self.layout_enabled,
            "engine": self.layout_engine,
            "dpi": self.layout_dpi,
            "layout_model": self.layout_model,
            "lang": self.layout_lang,
            "min_score": self.layout_min_score,
            "cache_dir": self.layout_cache_dir or None,
            "device": self.layout_device,
        }

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def report_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def sqlite_path(self) -> Path:
        """Filesystem path behind ``database_url`` for the SQLite MVP backend."""
        url = self.database_url
        if not url.startswith("sqlite"):
            raise ValueError(f"Only sqlite URLs are supported in the MVP, got: {url}")
        _, _, tail = url.partition("///")
        return Path(tail or "./data/paperpilot.sqlite")

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.index_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict:
        """Configuration snapshot safe to log: secrets are masked."""
        data = self.model_dump(mode="json")
        for key in ("llm_api_key", "embedding_api_key"):
            if data.get(key):
                data[key] = "***redacted***"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings; used by tests that patch the environment."""
    get_settings.cache_clear()


def settings_from_env(**overrides) -> Settings:
    """Build a fresh (uncached) Settings instance from the environment.

    Used by the CLI and by tests that need to override individual values without
    disturbing the process-wide cached settings.
    """
    return Settings(**overrides)
