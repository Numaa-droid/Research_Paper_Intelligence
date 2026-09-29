"""Model-provider abstraction.

Every model call in PaperPilot goes through :class:`LLMProvider`. That keeps the
whole system runnable — and fully testable — without external credentials, and
lets a local model be swapped in by changing one environment variable.

Requests are *structured*: an agent supplies the system/user prompt **and** the
retrieved evidence it was grounded on. Real providers only need the prompt text;
the deterministic mock uses the evidence to synthesise grounded answers, which is
what makes the offline end-to-end test meaningful rather than decorative.

Models answer in Markdown, never JSON. Structure is recovered deterministically
by :func:`paperpilot.response.parse_markdown`, so a model can no longer fail a
run by misplacing a comma.
"""

from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
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
    """A model's Markdown answer plus the metadata needed for traces."""

    text: str
    model: str
    provider: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0

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
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Return the model's Markdown answer to ``request``."""

    @property
    def model(self) -> str:
        return "unknown"


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
        max_tokens: int = 8192,
        max_tokens_ceiling: int = 16384,
        timeout: float = 90.0,
        max_attempts: int = 5,
        backoff_seconds: float = 2.0,
        max_concurrency: int = 2,
        reasoning_effort: str | None = None,
        seed: int | None = None,
        transport=None,
        sleep=time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.max_tokens_ceiling = max(max_tokens_ceiling, max_tokens)
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.reasoning_effort = reasoning_effort
        #: Shared by every agent thread using this provider; held only while a
        #: request is in flight, never while backing off.
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._transport = transport
        self._sleep = sleep

    def _backoff(self, attempt: int) -> float:
        base = min(self.backoff_seconds * 2 ** (attempt - 1), _MAX_BACKOFF_SECONDS)
        return base * random.uniform(0.75, 1.25)

    @property
    def model(self) -> str:
        return self._model

    def generate(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": (
                request.temperature if request.temperature is not None else self.temperature
            ),
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            payload["seed"] = self.seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # A response cut off at the token limit is missing its last sections, and
        # asking again with the same budget reproduces the same cut. Grow the budget
        # instead, up to the ceiling, and only then give up with the real reason.
        budget = request.max_tokens or self.max_tokens
        while True:
            payload["max_tokens"] = budget
            try:
                return self._post_with_retries(payload, headers, request)
            except _TruncatedResponse as exc:
                if budget >= self.max_tokens_ceiling:
                    raise ProviderError(
                        f"model response was cut off at the {budget}-token output limit before "
                        "it was complete. Raise PAPERPILOT_LLM_MAX_TOKENS_CEILING "
                        "(the model's own output limit permitting).",
                        details={
                            "agent": request.agent,
                            "task": request.task,
                            "max_tokens": budget,
                            "completion_tokens": exc.completion_tokens,
                        },
                    ) from None
                warning(
                    "llm.response_truncated",
                    agent=request.agent,
                    max_tokens=budget,
                    retry_with=min(budget * 2, self.max_tokens_ceiling),
                )
                budget = min(budget * 2, self.max_tokens_ceiling)

    def _post_with_retries(
        self, payload: dict[str, Any], headers: dict[str, str], request: LLMRequest
    ) -> LLMResponse:
        import httpx  # imported lazily so offline runs never touch the transport

        details: dict[str, Any] = {"agent": request.agent, "task": request.task}
        last_error = "no attempt was made"
        last_status: int | None = None
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                with self._slots, httpx.Client(
                    timeout=self.timeout, transport=self._transport
                ) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
            except httpx.HTTPError as exc:  # timeouts, refused or dropped connections
                last_error, last_status = f"{type(exc).__name__}: {exc}", None
                delay = self._backoff(attempt)
            else:
                if response.status_code < 400:
                    latency_ms = (time.perf_counter() - started) * 1000
                    return self._parse(response, latency_ms)
                last_error, last_status = _describe_http_error(response), response.status_code
                if response.status_code not in _RETRYABLE_STATUS:
                    raise ProviderError(
                        f"model provider rejected the request: {last_error}",
                        details={**details, "status_code": response.status_code},
                    )
                delay = _retry_after_seconds(response)
                if delay is None:
                    delay = self._backoff(attempt)

            warning(
                "llm.attempt_failed",
                agent=request.agent,
                attempt=attempt,
                status_code=last_status,
                error=last_error[:200],
                retry_in_s=round(delay, 2) if attempt < self.max_attempts else None,
            )
            if attempt < self.max_attempts:
                self._sleep(delay)

        if last_status == 429:
            message = (
                f"the model provider kept rate-limiting requests (HTTP 429) after "
                f"{self.max_attempts} attempts. The account's request quota is exhausted: "
                "wait a minute and try again, or lower PAPERPILOT_LLM_MAX_CONCURRENCY."
            )
        else:
            message = f"model provider failed after {self.max_attempts} attempts: {last_error}"
        raise ProviderError(message, details={**details, "status_code": last_status})

    def _parse(self, response, latency_ms: float) -> LLMResponse:
        try:
            body = response.json()
            choice = body["choices"][0]
            text = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("unexpected chat-completions response shape") from exc
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("model returned an empty response")

        usage_block = body.get("usage") or {}
        if choice.get("finish_reason") == "length":
            raise _TruncatedResponse(int(usage_block.get("completion_tokens", 0)))
        return LLMResponse(
            text=text,
            model=body.get("model", self._model),
            provider=self.name,
            usage=TokenUsage(
                prompt_tokens=int(usage_block.get("prompt_tokens", 0)),
                completion_tokens=int(usage_block.get("completion_tokens", 0)),
            ),
            latency_ms=latency_ms,
        )


class HuggingFaceLocalProvider(LLMProvider):
    """Open-weights chat model from the Hugging Face Hub, run in-process.

    Weights are downloaded once into the Hugging Face cache and loaded lazily on
    the first call, so constructing the provider (e.g. at API start-up) is cheap.
    No API key is involved; gated or private repos use the ``HF_TOKEN`` that
    ``huggingface_hub`` already reads from the environment.
    """

    name = "huggingface"

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        device: str = "auto",
    ) -> None:
        self._model_id = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model_id

    def _load(self):
        with self._load_lock:
            if self._model is not None:
                return self._model, self._tokenizer
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ProviderError(
                    "transformers and torch are required for PAPERPILOT_LLM_PROVIDER=huggingface; "
                    "install the 'local-llm' extra"
                ) from exc

            device = _resolve_torch_device(self.device, torch)
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
            debug("llm.loading_model", model=self._model_id, device=device)
            started = time.perf_counter()
            try:
                tokenizer = AutoTokenizer.from_pretrained(self._model_id)
                model = AutoModelForCausalLM.from_pretrained(self._model_id, dtype=dtype)
            except (OSError, ValueError) as exc:
                raise ProviderError(
                    f"could not load Hugging Face model '{self._model_id}': {exc}",
                    details={"model": self._model_id},
                ) from exc
            model.to(device).eval()
            debug(
                "llm.model_loaded",
                model=self._model_id,
                device=device,
                load_s=round(time.perf_counter() - started, 1),
            )
            self._model, self._tokenizer = model, tokenizer
            return model, tokenizer

    def generate(self, request: LLMRequest) -> LLMResponse:
        import torch

        # Shares the embedder's lock: one accelerator, and MPS/CUDA are not safe
        # for concurrent inference from several agent threads.
        from .embeddings import TORCH_INFERENCE_LOCK

        model, tokenizer = self._load()
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        temperature = request.temperature if request.temperature is not None else self.temperature
        generation: dict[str, Any] = {
            "max_new_tokens": request.max_tokens or self.max_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation.update(do_sample=True, temperature=temperature)
        else:
            generation.update(do_sample=False, temperature=None, top_p=None, top_k=None)

        started = time.perf_counter()
        with TORCH_INFERENCE_LOCK:
            try:
                inputs = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
                ).to(model.device)
                with torch.inference_mode():
                    output = model.generate(**inputs, **generation)
            except RuntimeError as exc:  # out of memory, unsupported op on device
                raise ProviderError(
                    f"local model generation failed: {exc}",
                    details={"agent": request.agent, "task": request.task},
                ) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        completion_ids = output[0][prompt_tokens:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        budget = generation["max_new_tokens"]
        generated = int(completion_ids.shape[-1])
        stopped_on_eos = generated > 0 and int(completion_ids[-1]) == tokenizer.eos_token_id
        if generated >= budget and not stopped_on_eos:
            raise ProviderError(
                f"model response was cut off at the {budget}-token output limit before it was "
                "complete. Raise PAPERPILOT_LLM_MAX_TOKENS.",
                details={"agent": request.agent, "task": request.task, "max_tokens": budget},
            )
        if not text.strip():
            raise ProviderError("model returned an empty response")
        return LLMResponse(
            text=text,
            model=self._model_id,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=generated),
            latency_ms=latency_ms,
        )


def _resolve_torch_device(preference: str, torch) -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _TruncatedResponse(Exception):
    """The provider stopped at ``max_tokens`` (``finish_reason == "length"``)."""

    def __init__(self, completion_tokens: int) -> None:
        super().__init__(completion_tokens)
        self.completion_tokens = completion_tokens


#: Statuses worth retrying: rate limits, timeouts and transient server errors.
#: Anything else (bad key, unknown model, malformed request) fails immediately.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_MAX_BACKOFF_SECONDS = 30.0
_MAX_RETRY_AFTER_SECONDS = 120.0


def _retry_after_seconds(response) -> float | None:
    """Honour a server-sent ``Retry-After`` (delta-seconds or HTTP date)."""
    value = (response.headers.get("retry-after") or "").strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        seconds = (when - datetime.now(UTC)).total_seconds()
    return min(max(seconds, 0.0), _MAX_RETRY_AFTER_SECONDS)


def _describe_http_error(response) -> str:
    """``HTTP 401 Unauthorized: <provider message>``, without dumping the whole body."""
    reason = f"HTTP {response.status_code} {response.reason_phrase}".strip()
    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()
        return f"{reason}: {text[:200]}" if text else reason
    error = body.get("error") if isinstance(body, dict) else None
    message = (
        (error.get("message") if isinstance(error, dict) else error)
        or (body.get("detail") if isinstance(body, dict) else None)
        or (body.get("message") if isinstance(body, dict) else None)
    )
    return f"{reason}: {str(message)[:200]}" if message else reason


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
            max_tokens_ceiling=settings.llm_max_tokens_ceiling,
            timeout=settings.llm_timeout_seconds,
            max_attempts=settings.llm_max_attempts,
            backoff_seconds=settings.llm_backoff_seconds,
            max_concurrency=settings.llm_max_concurrency,
            reasoning_effort=settings.llm_reasoning_effort,
            seed=settings.llm_seed,
        )
    if settings.llm_provider == "huggingface":
        debug("llm.provider_selected", provider="huggingface", model=settings.llm_model)
        return HuggingFaceLocalProvider(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            device=settings.llm_device,
        )
    raise ProviderError(f"unknown LLM provider: {settings.llm_provider}")
