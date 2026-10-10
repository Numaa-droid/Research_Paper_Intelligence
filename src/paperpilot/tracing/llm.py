"""Model calls as spans.

Wrapping the provider, rather than each call site, means every model call —
planner, specialists, coverage sweep, verifier, and any added later — is
recorded with its full prompt, answer, token counts and latency.
"""

from __future__ import annotations

from typing import Any

from ..providers.llm import LLMProvider, LLMRequest, LLMResponse
from .tracer import span


class TracedLLMProvider(LLMProvider):
    """Delegates to ``inner`` and records each call as an ``llm`` span."""

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self.name = inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    def __getattr__(self, item: str) -> Any:
        # Only reached for attributes this wrapper does not define.
        return getattr(self.inner, item)

    def generate(self, request: LLMRequest) -> LLMResponse:
        with span(
            self.inner.model,
            kind="llm",
            metadata={
                "provider": self.inner.name,
                "agent": request.agent,
                "task": request.task,
                "prompt_version": request.prompt_version,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "evidence_sources": len(request.evidence),
            },
        ) as s:
            s.set_inputs(
                messages=[
                    {"role": "system", "content": s.content(request.system)},
                    {"role": "user", "content": s.content(request.user)},
                ]
            )
            response = self.inner.generate(request)
            s.set_outputs(
                message={"role": "assistant", "content": s.content(response.text)}
            )
            s.set_metadata(
                model=response.model,
                provider=response.provider,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=round(response.latency_ms, 2),
                cost_usd=response.estimated_cost_usd(),
            )
            return response


def traced_llm(provider: LLMProvider) -> LLMProvider:
    """``provider`` wrapped for tracing, once."""
    return provider if isinstance(provider, TracedLLMProvider) else TracedLLMProvider(provider)
