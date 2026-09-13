"""Shared machinery for specialist agents.

An agent's job is narrow by design: build a small number of targeted retrieval
queries, render a versioned prompt over the retrieved evidence, call the model
for JSON, coerce it into typed models, and hand the result to deterministic
validation. No agent reads the whole PDF, and no agent's prose is ever passed to
another agent — only validated structures cross node boundaries.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..domain.enums import ElementType
from ..domain.models import (
    AgentResult,
    AgentUsage,
    Evidence,
    Finding,
    PaperMetadata,
)
from ..errors import AgentError, ProviderError
from ..observability import info, warning
from ..providers.llm import LLMProvider, LLMRequest, LLMResponse
from ..retrieval.index import PaperIndex
from ..retrieval.retriever import RetrievalQuery, RetrievalResult, Retriever
from ..validation.evidence import validate_findings
from .prompts.registry import PromptTemplate

#: Hard cap on context handed to one agent, in characters. Targeted retrieval is
#: the point: an agent that needs more than this is asking the wrong question.
MAX_CONTEXT_CHARS = 14_000


@dataclass(slots=True)
class AgentContext:
    """Everything an agent is allowed to see."""

    document_id: str
    metadata: PaperMetadata
    index: PaperIndex
    retriever: Retriever
    priority_sections: list[str] = field(default_factory=list)
    attempt: int = 1
    #: Feedback from a previous failed attempt, used on retry.
    retry_feedback: str | None = None
    prior_results: dict[str, Any] = field(default_factory=dict)


def format_metadata(metadata: PaperMetadata) -> str:
    """Compact, factual metadata block for prompts."""
    lines = [
        f"title: {metadata.title or 'unknown'}",
        f"pages: {metadata.page_count} (page numbers are 1-based PDF indices)",
        f"sections detected: {', '.join(metadata.sections[:24]) or 'none'}",
        f"tables: {metadata.table_count}, figures: {metadata.figure_count}, "
        f"equations: {metadata.equation_count}",
    ]
    if metadata.ocr_page_count:
        lines.append(
            f"NOTE: {metadata.ocr_page_count} page(s) were read by OCR and may contain "
            "transcription errors."
        )
    return "\n".join(lines)


def format_context(results: list[RetrievalResult], *, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Render retrieved chunks as a citable context block."""
    blocks: list[str] = []
    used = 0
    for result in results:
        chunk = result.chunk
        header = (
            f"[chunk_id: {chunk.chunk_id} | page: {chunk.page} | "
            f"section: {chunk.citation_section or 'unknown'} | "
            f"type: {chunk.element_type.value}"
            + (f" | element_id: {chunk.element_id}" if chunk.element_id else "")
            + (" | EXTRACTED BY OCR" if chunk.source.value == "ocr" else "")
            + "]"
        )
        body = f"{header}\n{chunk.text}"
        if used + len(body) > max_chars:
            break
        blocks.append(body)
        used += len(body)
    return "\n\n".join(blocks) if blocks else "(no relevant context was retrieved)"


def evidence_payload(results: list[RetrievalResult]) -> list[dict[str, Any]]:
    """Structured mirror of the context block, for providers that want it."""
    return [r.to_dict() for r in results]


def dedupe_results(groups: list[list[RetrievalResult]], *, limit: int) -> list[RetrievalResult]:
    """Interleave several result lists, dropping duplicates, up to ``limit``.

    Round-robin rather than concatenation so that every query an agent asked is
    represented even when one of them returns many strong hits.
    """
    seen: set[str] = set()
    merged: list[RetrievalResult] = []
    for rank in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if rank >= len(group):
                continue
            result = group[rank]
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)
            merged.append(result)
            if len(merged) >= limit:
                return merged
    return merged


class BaseAgent(ABC):
    """Template method shared by every specialist agent."""

    name: str = "base"
    prompt: PromptTemplate
    #: Number of chunks handed to the model.
    context_limit: int = 12

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    # --- to implement in subclasses --------------------------------------

    @abstractmethod
    def queries(self, context: AgentContext) -> list[RetrievalQuery]:
        """Targeted retrieval queries this agent needs."""

    @abstractmethod
    def build_result(
        self, payload: dict[str, Any], context: AgentContext, results: list[RetrievalResult]
    ) -> tuple[list[Finding], dict[str, Any], list[str]]:
        """Turn raw model JSON into (findings, payload, open_questions)."""

    def should_run(self, context: AgentContext) -> tuple[bool, str | None]:
        """Conditional activation. Returns ``(run, skip_reason)``."""
        return True, None

    # --- template method --------------------------------------------------

    def run(self, context: AgentContext) -> AgentResult:
        should_run, skip_reason = self.should_run(context)
        if not should_run:
            info("agent.skipped", agent=self.name, reason=skip_reason)
            return AgentResult(
                agent=self.name,
                status="skipped",
                skip_reason=skip_reason,
                attempt=context.attempt,
            )

        started = time.perf_counter()
        info("agent.started", agent=self.name, attempt=context.attempt)

        results = self.retrieve(context)
        if not results:
            warning("agent.no_context", agent=self.name)

        try:
            response = self.call_model(context, results)
        except ProviderError as exc:
            warning("agent.failed", agent=self.name, error=str(exc)[:300])
            return AgentResult(
                agent=self.name, status="failed", error=str(exc), attempt=context.attempt
            )

        try:
            findings, payload, open_questions = self.build_result(
                response.content, context, results
            )
        except Exception as exc:  # noqa: BLE001 - a malformed payload is an agent failure
            warning("agent.parse_failed", agent=self.name, error=str(exc)[:300])
            return AgentResult(
                agent=self.name,
                status="failed",
                error=f"could not parse agent output: {exc}",
                attempt=context.attempt,
            )

        for finding in findings:
            finding.agent = self.name
        validate_findings(findings, context.index)

        usage = AgentUsage(
            agent=self.name,
            prompt_version=self.prompt.version,
            model=response.model,
            provider=response.provider,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            attempt=context.attempt,
            estimated_cost_usd=response.estimated_cost_usd(),
        )
        info(
            "agent.completed",
            agent=self.name,
            attempt=context.attempt,
            findings=len(findings),
            evidence=sum(len(f.evidence) for f in findings),
            grounded=sum(1 for f in findings if f.has_evidence),
            tokens=usage.total_tokens,
            duration_ms=usage.latency_ms,
        )
        return AgentResult(
            agent=self.name,
            status="ok",
            findings=findings,
            payload=payload,
            open_questions=open_questions,
            usage=usage,
            attempt=context.attempt,
        )

    # --- steps ------------------------------------------------------------

    def retrieve(self, context: AgentContext) -> list[RetrievalResult]:
        groups = [context.retriever.search(q) for q in self.queries(context)]
        return dedupe_results(groups, limit=self.context_limit)

    def call_model(
        self, context: AgentContext, results: list[RetrievalResult]
    ) -> LLMResponse:
        user = self.prompt.render(
            metadata=format_metadata(context.metadata),
            context=format_context(results),
        )
        if context.retry_feedback:
            user += (
                "\n\nPREVIOUS ATTEMPT WAS REJECTED\n"
                f"{context.retry_feedback}\n"
                "Re-answer using only quotes you can find verbatim in the context above. "
                "Drop any claim you cannot ground, rather than restating it."
            )
        request = LLMRequest(
            agent=self.name,
            task=self.name,
            system=self.prompt.system,
            user=user,
            prompt_version=self.prompt.version,
            evidence=evidence_payload(results),
            context={
                "document_id": context.document_id,
                "attempt": context.attempt,
                "metadata": context.metadata.model_dump(mode="json"),
            },
        )
        return self.provider.generate_json(request)


def evidence_from_payload(items: Any, *, agent: str) -> list[Evidence]:
    """Parse a list of evidence objects, dropping the unparseable ones."""
    from ..validation.schema import parse_evidence

    if not isinstance(items, list):
        return []
    return [e for item in items if (e := parse_evidence(item, agent=agent)) is not None]


def section_queries(
    text: str,
    *,
    sections: list[str] | None,
    top_k: int,
    element_types: list[ElementType] | None = None,
) -> list[RetrievalQuery]:
    """Build a broad query plus a section-scoped one when sections are known."""
    queries = [RetrievalQuery(text=text, top_k=top_k, element_types=element_types)]
    if sections:
        queries.append(
            RetrievalQuery(
                text=text, top_k=max(3, top_k // 2), sections=sections, element_types=element_types
            )
        )
    return queries


def require_dict(payload: Any, agent: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AgentError(f"{agent} returned {type(payload).__name__}, expected a JSON object")
    return payload


def json_preview(payload: Any, limit: int = 400) -> str:
    try:
        return json.dumps(payload)[:limit]
    except (TypeError, ValueError):
        return str(payload)[:limit]
