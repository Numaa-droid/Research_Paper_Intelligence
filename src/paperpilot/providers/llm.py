"""Model-provider abstraction.

Every model call in PaperPilot goes through :class:`LLMProvider`. That keeps the
whole system runnable — and fully testable — without external credentials, and
lets a local model be swapped in by changing one environment variable.

Requests are *structured*: an agent supplies the system/user prompt **and** the
retrieved evidence it was grounded on. Real providers only need the prompt text;
the deterministic mock uses the evidence to synthesise grounded answers, which is
what makes the offline end-to-end test meaningful rather than decorative.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..errors import ProviderError
from ..observability import debug, warning

#: Rough USD per 1K tokens, used only for the evaluation harness's cost estimate.
#: Unknown models simply report no cost rather than a fabricated one.
_PRICE_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4.1-mini": (0.0004, 0.0016),
}


@dataclass(slots=True)
class LLMRequest:
    """One structured model call."""

    agent: str
    task: str
    system: str
    user: str
    prompt_version: str = "v1"
    schema: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class LLMResponse:
    """A parsed JSON response plus the metadata needed for traces."""

    content: dict[str, Any]
    model: str
    provider: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    raw: str = ""

    def estimated_cost_usd(self) -> float | None:
        price = _PRICE_PER_1K.get(self.model)
        if not price:
            return None
        prompt_rate, completion_rate = price
        return round(
            (self.usage.prompt_tokens / 1000) * prompt_rate
            + (self.usage.completion_tokens / 1000) * completion_rate,
            6,
        )


class LLMProvider(ABC):
    """Interface every model backend must implement."""

    name: str = "abstract"

    @abstractmethod
    def generate_json(self, request: LLMRequest) -> LLMResponse:
        """Return a JSON object obeying ``request.schema``."""

    @property
    def model(self) -> str:
        return "unknown"


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in prose or fences often enough that tolerating it is worth
    a few lines; anything that still fails to parse raises rather than being
    guessed at.
    """
    if not text or not text.strip():
        raise ProviderError("model returned an empty response")
    candidate = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ProviderError(
                "model response did not contain a JSON object",
                details={"preview": text[:400]},
            ) from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"model response was not valid JSON: {exc}",
                details={"preview": text[:400]},
            ) from exc

    if not isinstance(parsed, dict):
        raise ProviderError("model response JSON must be an object")
    return parsed


class OpenAICompatibleProvider(LLMProvider):
    """Chat-completions client for any OpenAI-compatible endpoint.

    Works unchanged against OpenAI, vLLM, Ollama's compatibility layer, LM Studio
    and similar local servers — only ``base_url``/``model`` change.
    """

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 90.0,
        max_attempts: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max_attempts

    @property
    def model(self) -> str:
        return self._model

    def generate_json(self, request: LLMRequest) -> LLMResponse:
        import httpx  # imported lazily so offline runs never touch the transport

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": (
                request.temperature if request.temperature is not None else self.temperature
            ),
            "max_tokens": request.max_tokens or self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
                    response.raise_for_status()
                    body = response.json()
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced
                last_error = exc
                warning(
                    "llm.attempt_failed",
                    agent=request.agent,
                    attempt=attempt,
                    error=str(exc)[:200],
                )
                if attempt < self.max_attempts:
                    time.sleep(min(2**attempt * 0.5, 8.0))
                continue

            latency_ms = (time.perf_counter() - started) * 1000
            try:
                raw = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError("unexpected chat-completions response shape") from exc

            usage_block = body.get("usage") or {}
            return LLMResponse(
                content=extract_json(raw),
                model=body.get("model", self._model),
                provider=self.name,
                usage=TokenUsage(
                    prompt_tokens=int(usage_block.get("prompt_tokens", 0)),
                    completion_tokens=int(usage_block.get("completion_tokens", 0)),
                ),
                latency_ms=latency_ms,
                raw=raw,
            )

        raise ProviderError(
            f"model provider failed after {self.max_attempts} attempts: {last_error}",
            details={"agent": request.agent, "task": request.task},
        )


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate the configured provider."""
    if settings.llm_provider == "mock":
        from .mock_llm import MockLLMProvider

        # Deliberately ignores settings.llm_model: the mock must never report a
        # real model name, or traces and cost estimates would misattribute the run.
        return MockLLMProvider()
    if settings.llm_provider == "openai_compatible":
        debug("llm.provider_selected", provider="openai_compatible", model=settings.llm_model)
        return OpenAICompatibleProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_attempts=settings.llm_max_attempts,
        )
    raise ProviderError(f"unknown LLM provider: {settings.llm_provider}")
