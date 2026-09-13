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

LLMProviderName = Literal["mock", "openai_compatible"]
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
    llm_max_tokens: int = Field(default=2048, gt=0)
    llm_timeout_seconds: float = Field(default=90.0, gt=0)
    llm_max_attempts: int = Field(default=3, ge=1, le=6)

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

    # --- Storage ----------------------------------------------------------
    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./data/paperpilot.sqlite"

    # --- Observability ----------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_document_text: bool = False

    @field_validator("chunk_overlap_chars")
    @classmethod
    def _overlap_smaller_than_chunk(cls, v: int, info) -> int:
        max_chars = info.data.get("chunk_max_chars", 1200)
        if v >= max_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_max_chars")
        return v

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
