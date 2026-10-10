"""Shared machinery for specialist agents.

An agent's job is narrow by design: build a small number of targeted retrieval
queries, render a versioned prompt over the retrieved evidence, call the model
for a Markdown answer, parse that deterministically into a response document,
read typed models out of it, and hand the result to deterministic validation. No agent reads the whole PDF, and no agent's prose is ever passed to
another agent — only validated structures cross node boundaries.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..discourse.attribution import attribution_map, describe_sources
from ..domain.enums import ElementType
from ..domain.models import (
    AgentResult,
    AgentUsage,
    Finding,
    PaperMetadata,
)
from ..errors import ProviderError
from ..observability import info, warning
from ..providers.llm import LLMProvider, LLMRequest, LLMResponse
from ..response import parse_markdown
from ..response.reader import DocumentReader
from ..retrieval.index import PaperIndex
from ..retrieval.retriever import RetrievalQuery, RetrievalResult, Retriever
from ..tracing import span
from ..validation.evidence import validate_findings
from .grounding import Grounder, SourceMap, source_label
from .prompts.registry import PromptTemplate
from .specs import agent_guidance

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
    #: What kind of paper this is (see :mod:`paperpilot.discourse.profile`) and
    #: the signals that decided it; selects the agent's reference guide.
    paper_kind: str = "empirical"
    paper_kind_signals: list[str] = field(default_factory=list)


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


def _attribution_tag(chunk, index: PaperIndex | None) -> str:
    """Header tag for a chunk in which the paper reports another work."""
    if index is None:
        return ""
    kind, sources = attribution_map(index).chunk_attribution(chunk.chunk_id)
    if kind == "cited":
        return f" | REPORTS CITED WORK {describe_sources(sources)}"
    if kind == "mixed":
        return f" | PARTLY REPORTS CITED WORK {describe_sources(sources)}"
    return ""


def _source_block(position: int, result: RetrievalResult, index: PaperIndex | None = None) -> str:
    chunk = result.chunk
    header = (
        f"[source: {source_label(position)} | page: {chunk.page} | "
        f"section: {chunk.citation_section or 'unknown'} | "
        f"type: {chunk.element_type.value}"
        + (f" | element_id: {chunk.element_id}" if chunk.element_id else "")
        + (" | EXTRACTED BY OCR" if chunk.source.value == "ocr" else "")
        + _attribution_tag(chunk, index)
        + "]"
    )
    return f"{header}\n{chunk.text}"


def fit_context(
    results: list[RetrievalResult],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
    index: PaperIndex | None = None,
) -> list[RetrievalResult]:
    """The leading results whose source blocks fit in ``max_chars``.

    Exactly the chunks :func:`format_context` renders, so what an agent is
    recorded as having read is what its prompt really contained.
    """
    fitted: list[RetrievalResult] = []
    used = 0
    for position, result in enumerate(results):
        size = len(_source_block(position, result, index))
        if used + size > max_chars:
            break
        fitted.append(result)
        used += size
    return fitted


def format_context(
    results: list[RetrievalResult],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
    index: PaperIndex | None = None,
) -> str:
    """Render retrieved chunks as a citable context block.

    Each chunk is labelled ``S1``, ``S2``, … in retrieval order — the labels the
    model cites and :class:`~paperpilot.agents.grounding.SourceMap` resolves.
    With ``index``, a chunk in which the paper reports another work says so in
    its header, so the model can tell the paper's findings from the ones it cites.
    """
    blocks = [
        _source_block(position, result, index)
        for position, result in enumerate(
            fit_context(results, max_chars=max_chars, index=index)
        )
    ]
    return "\n\n".join(blocks) if blocks else "(no relevant context was retrieved)"


def evidence_payload(results: list[RetrievalResult]) -> list[dict[str, Any]]:
    """Structured mirror of the context block, for providers that want it."""
    return [{**r.to_dict(), "source_id": source_label(i)} for i, r in enumerate(results)]


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
    #: Words that mark a section this agent must read, matched against the start
    #: of each word of the paper's own headings ("metric" -> "4.2 Evaluation
    #: Metrics"). Similarity search alone misses a section whose wording differs
    #: from the query; the paper's structure does not.
    anchor_section_hints: tuple[str, ...] = ()
    #: Element types this agent must see in full, e.g. every equation.
    anchor_element_types: tuple[ElementType, ...] = ()
    #: Chunks always left for similarity search, however many anchors match.
    min_search_slots: int = 4

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    # --- to implement in subclasses --------------------------------------

    @abstractmethod
    def queries(self, context: AgentContext) -> list[RetrievalQuery]:
        """Targeted retrieval queries this agent needs."""

    @abstractmethod
    def build_result(
        self, reader: DocumentReader, context: AgentContext, grounder: Grounder
    ) -> tuple[list[Finding], dict[str, Any], list[str]]:
        """Read (findings, payload, open_questions) out of the parsed response."""

    def should_run(self, context: AgentContext) -> tuple[bool, str | None]:
        """Conditional activation. Returns ``(run, skip_reason)``."""
        return True, None

    # --- template method --------------------------------------------------

    def run(self, context: AgentContext) -> AgentResult:
        with span(
            self.name,
            kind="agent",
            inputs={
                "attempt": context.attempt,
                "paper_kind": context.paper_kind,
                "priority_sections": context.priority_sections,
                "retry_feedback": context.retry_feedback,
            },
            metadata={"prompt_version": self.prompt.version, "context_limit": self.context_limit},
        ) as s:
            result = self._run(context)
            s.set_outputs(
                status=result.status,
                findings=[
                    {
                        "claim": f.claim,
                        "status": getattr(f.status, "value", f.status),
                        "evidence": len(f.evidence),
                        "grounded": f.has_evidence,
                    }
                    for f in result.findings
                ],
                open_questions=result.open_questions,
                skip_reason=result.skip_reason,
                context_chunk_ids=result.context_chunk_ids,
            )
            if result.status == "failed":
                s.fail(result.error or "agent failed", error_type="AgentFailed")
            return result

    def _run(self, context: AgentContext) -> AgentResult:
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

        with span("retrieve_context", kind="chain") as s:
            results = fit_context(self.retrieve(context), index=context.index)
            s.set_outputs(
                sources=[
                    {
                        "source": source_label(i),
                        "chunk_id": r.chunk_id,
                        "page": r.chunk.page,
                        "section": r.chunk.citation_section,
                        "matched_by": r.matched_by,
                    }
                    for i, r in enumerate(results)
                ]
            )
        shown = [r.chunk_id for r in results]
        if not results:
            warning("agent.no_context", agent=self.name)
        structural = [r for r in results if r.matched_by == ["structure"]]
        info(
            "agent.context",
            agent=self.name,
            sources=len(results),
            structural=len(structural),
            searched=len(results) - len(structural),
            anchor_sections=list(dict.fromkeys(
                r.chunk.citation_section for r in structural if r.chunk.citation_section
            )),
            chars=len(format_context(results, index=context.index)),
        )

        call_started = time.perf_counter()
        try:
            response = self.call_model(context, results)
        except ProviderError as exc:
            warning("agent.failed", agent=self.name, error=str(exc)[:300])
            return AgentResult(
                agent=self.name,
                status="failed",
                error=str(exc),
                attempt=context.attempt,
                context_chunk_ids=shown,
            )

        info(
            "agent.model_answered",
            agent=self.name,
            model=response.model,
            latency_ms=round(response.latency_ms or (time.perf_counter() - call_started) * 1000, 1),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            chars=len(response.text),
        )
        with span("parse_markdown", kind="parser", inputs={"chars": len(response.text)}) as s:
            document = parse_markdown(response.text)
            s.set_outputs(
                sections=document.stats.sections,
                tables=document.stats.tables,
                quotes=document.stats.blockquotes,
                citations=document.stats.citations,
                repairs=document.warnings,
            )
        info(
            "agent.response_parsed",
            agent=self.name,
            sections=document.stats.sections,
            tables=document.stats.tables,
            quotes=document.stats.blockquotes,
            citations=document.stats.citations,
            repairs=len(document.warnings),
        )
        record = {
            "response_markdown": response.text,
            "response_document": document.model_dump(mode="json", exclude_none=True),
            "context_chunk_ids": shown,
        }
        if not document.sections:
            warning("agent.response_unstructured", agent=self.name, chars=len(response.text))
        if document.warnings:
            warning("agent.response_repaired", agent=self.name, repairs=document.warnings[:5])
        grounder = Grounder(sources=SourceMap(results), index=context.index, agent=self.name)
        try:
            findings, payload, open_questions = self.build_result(
                DocumentReader(document), context, grounder
            )
        except Exception as exc:  # noqa: BLE001 - an unreadable answer is an agent failure
            warning("agent.parse_failed", agent=self.name, error=str(exc)[:300])
            return AgentResult(
                agent=self.name,
                status="failed",
                error=f"could not read agent output: {exc}",
                attempt=context.attempt,
                **record,
            )

        for finding in findings:
            finding.agent = self.name
        with span("validate_findings", kind="tool", inputs={"findings": len(findings)}) as s:
            validate_findings(findings, context.index)
            s.set_outputs(
                grounded=sum(1 for f in findings if f.has_evidence),
                evidence=sum(len(f.evidence) for f in findings),
            )

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
            **record,
        )

    # --- steps ------------------------------------------------------------

    def retrieve(self, context: AgentContext) -> list[RetrievalResult]:
        anchored = self.anchors(context)
        groups = [context.retriever.search(q) for q in self.queries(context)]
        searched = dedupe_results(groups, limit=self.context_limit)
        seen = {r.chunk_id for r in anchored}
        # The bibliography matches almost any query lexically and says nothing
        # about the paper's own content, so it never fills a context slot.
        rest = [
            r
            for r in searched
            if r.chunk_id not in seen
            and r.chunk.element_type is not ElementType.REFERENCE
            and not (r.chunk.section or "").lower().startswith(_NON_CONTENT_SECTIONS)
        ]
        return (anchored + rest)[: self.context_limit]

    def anchors(self, context: AgentContext) -> list[RetrievalResult]:
        """Chunks this agent reads because of where they are, not how they score.

        Deterministic: element-type anchors in document order, then the matching
        sections round-robin (the first chunk of every matching section before
        the second of any), each table or figure with its caption.
        """
        budget = max(0, self.context_limit - self.min_search_slots)
        if not budget:
            return []
        index = context.index
        groups: list[list[RetrievalResult]] = []
        for element_type in self.anchor_element_types:
            groups.extend([[_anchor(c)] for c in index.by_element_type(element_type)])
        labels: list[str] = []
        for label in matching_sections(index.sections(), self.anchor_section_hints):
            # A matching section brings its subsections: "3 System Architecture"
            # covers "3.2 Structure-Aware Chunking" whatever that is called.
            children = [c.citation_section for c in index.chunks if c.section == label]
            for name in [label, *children]:
                if name and name not in labels:
                    labels.append(name)
        for label in labels:
            chunks = [c for c in index.chunks if c.citation_section == label]
            if chunks:
                groups.append([_anchor(c) for c in chunks])

        picked: list[RetrievalResult] = []
        seen: set[str] = set()
        for result in dedupe_results(groups, limit=budget * 2):
            members = (
                index.by_element_id(result.chunk.element_id)
                if result.chunk.element_id
                else [result.chunk]
            )
            for chunk in members:
                if chunk.chunk_id not in seen and len(picked) < budget:
                    seen.add(chunk.chunk_id)
                    picked.append(_anchor(chunk))
        return picked

    def call_model(
        self, context: AgentContext, results: list[RetrievalResult]
    ) -> LLMResponse:
        user = self.prompt.render(
            guidance=agent_guidance(
                self.name, context.paper_kind, signals=context.paper_kind_signals
            ),
            metadata=format_metadata(context.metadata),
            context=format_context(results, index=context.index),
        )
        if context.retry_feedback:
            user += (
                "\n\nPREVIOUS ATTEMPT WAS REJECTED\n"
                f"{context.retry_feedback}\n"
                "Re-answer in the same Markdown format, using only quotes you can find "
                "verbatim in the context above. "
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
        return self.provider.generate(request)


_NON_CONTENT_SECTIONS = ("reference", "bibliograph", "acknowledg", "front matter")


def matching_sections(sections: list[str], hints: tuple[str, ...]) -> list[str]:
    """Headings, in document order, with a word starting with one of ``hints``."""
    if not hints:
        return []
    matched = []
    for label in sections:
        words = re.findall(r"[a-z]+", label.lower())
        if label.lower().startswith(_NON_CONTENT_SECTIONS):
            continue
        if any(word.startswith(hint) for word in words for hint in hints):
            matched.append(label)
    return matched


def _anchor(chunk) -> RetrievalResult:
    return RetrievalResult(chunk=chunk, score=0.0, matched_by=["structure"])


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
