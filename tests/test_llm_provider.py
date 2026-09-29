"""OpenAI-compatible provider: rate limits, retries and concurrency."""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from paperpilot.domain.enums import AnalysisStatus
from paperpilot.errors import ProviderError
from paperpilot.providers.llm import LLMProvider, LLMRequest, OpenAICompatibleProvider

OK_BODY = {
    "model": "test-model",
    "choices": [{"message": {"content": "## Answer\n42"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


def _request() -> LLMRequest:
    return LLMRequest(agent="summary", task="summarize", system="s", user="u")


def _provider(handler, *, sleeps=None, **kwargs) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="https://llm.test/v1",
        api_key="key",
        model="test-model",
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
        **kwargs,
    )


def test_a_rate_limit_is_retried_honouring_retry_after():
    calls, sleeps = [], []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=OK_BODY)

    response = _provider(handler, sleeps=sleeps).generate(_request())

    assert response.text == "## Answer\n42"
    assert len(calls) == 2
    assert sleeps == [7.0]


def test_backoff_grows_between_attempts_without_retry_after():
    sleeps = []
    provider = _provider(lambda r: httpx.Response(503), sleeps=sleeps, max_attempts=4, backoff_seconds=2.0)

    with pytest.raises(ProviderError):
        provider.generate(_request())

    assert len(sleeps) == 3
    assert sleeps[0] < sleeps[1] < sleeps[2]
    assert 1.5 <= sleeps[0] <= 2.5


def test_persistent_rate_limiting_explains_itself():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"message": "Too Many Requests"}})

    with pytest.raises(ProviderError) as info:
        _provider(handler, max_attempts=3).generate(_request())

    assert len(calls) == 3
    assert "rate-limiting" in info.value.message
    assert "PAPERPILOT_LLM_MAX_CONCURRENCY" in info.value.message
    assert info.value.details["status_code"] == 429


def test_a_rejected_key_fails_immediately():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    with pytest.raises(ProviderError) as info:
        _provider(handler).generate(_request())

    assert len(calls) == 1
    assert "401" in info.value.message and "invalid api key" in info.value.message


def _truncated_body(completion_tokens: int) -> dict:
    return {
        "model": "test-model",
        "choices": [
            {"message": {"content": "## Findings\n### Finding 1\n- Claim: the mod"}, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": completion_tokens},
    }


def test_a_truncated_response_is_re_requested_with_a_larger_budget():
    budgets = []

    def handler(request):
        budgets.append(json.loads(request.content)["max_tokens"])
        if len(budgets) == 1:
            return httpx.Response(200, json=_truncated_body(2048))
        return httpx.Response(200, json=OK_BODY)

    response = _provider(handler, max_tokens=2048, max_tokens_ceiling=16384).generate(_request())

    assert response.text == "## Answer\n42"
    assert budgets == [2048, 4096]


def test_truncation_at_the_ceiling_names_the_token_limit():
    budgets = []

    def handler(request):
        budgets.append(json.loads(request.content)["max_tokens"])
        return httpx.Response(200, json=_truncated_body(budgets[-1]))

    with pytest.raises(ProviderError) as info:
        _provider(handler, max_tokens=2048, max_tokens_ceiling=6000).generate(_request())

    assert budgets == [2048, 4096, 6000]
    assert "cut off at the 6000-token output limit" in info.value.message


def test_concurrent_requests_share_a_limited_number_of_slots():
    lock = threading.Lock()
    in_flight = peak = 0

    def handler(request):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return httpx.Response(200, json=OK_BODY)

    provider = _provider(handler, max_concurrency=2)
    threads = [threading.Thread(target=provider.generate, args=(_request(),)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak == 2


class _AlwaysRateLimited(LLMProvider):
    name = "rate_limited"

    def generate(self, request):
        raise ProviderError("the model provider kept rate-limiting requests (HTTP 429)")


def test_a_run_where_every_agent_fails_is_a_failure_not_an_empty_report(settings, embedder, paper_pdf):
    from paperpilot.service import PaperPilotService
    from paperpilot.storage.sqlite_repo import (
        InMemoryAnalysisRepository,
        InMemoryDocumentRepository,
    )

    service = PaperPilotService(
        settings,
        documents=InMemoryDocumentRepository(),
        analyses=InMemoryAnalysisRepository(),
        llm_provider=_AlwaysRateLimited(),
        embedding_provider=embedder,
    )
    record = service.analyze_file(paper_pdf)

    assert record.status is AnalysisStatus.FAILED
    assert record.report_markdown is None
    assert any("429" in error for error in record.errors)
    retry_events = [e for e in record.events if e.stage.startswith("agent:") and e.detail.get("attempt") == 2]
    assert retry_events and all(e.status == "failed" for e in retry_events)


def test_huggingface_provider_is_built_without_loading_weights():
    from paperpilot.config import settings_from_env
    from paperpilot.providers.llm import HuggingFaceLocalProvider, build_llm_provider

    settings = settings_from_env(llm_provider="huggingface", llm_model="ibm-granite/granite-4.1-3b")
    provider = build_llm_provider(settings)

    assert isinstance(provider, HuggingFaceLocalProvider)
    assert provider.model == "ibm-granite/granite-4.1-3b"
    assert provider._model is None


def test_huggingface_provider_parses_only_the_generated_tokens():
    torch = pytest.importorskip("torch")
    from paperpilot.providers.llm import HuggingFaceLocalProvider

    prompt_ids = torch.tensor([[1, 2, 3]])

    class Inputs(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, messages, **_kwargs):
            assert [m["role"] for m in messages] == ["system", "user"]
            return Inputs(input_ids=prompt_ids)

        def decode(self, ids, **_kwargs):
            assert ids.tolist() == [7, 8]
            return "## Answer\n42"

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3, 7, 8]])

    provider = HuggingFaceLocalProvider(model="ibm-granite/granite-4.1-3b")
    provider._model, provider._tokenizer = Model(), Tokenizer()

    response = provider.generate(_request())

    assert response.text == "## Answer\n42"
    assert response.provider == "huggingface"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (3, 2)


def test_models_are_asked_for_markdown_not_json():
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    _provider(handler).generate(_request())

    assert "response_format" not in payloads[0]


def test_reasoning_effort_is_sent_only_when_configured():
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    _provider(handler).generate(_request())
    _provider(handler, reasoning_effort="none").generate(_request())

    assert "reasoning_effort" not in payloads[0]
    assert payloads[1]["reasoning_effort"] == "none"
