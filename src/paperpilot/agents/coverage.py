"""Coverage sweep: nothing in the paper goes unread.

The specialists read what their retrieval selects, and selection by similarity
or heading misses whatever is phrased or titled unexpectedly: a hardware
section, a regularisation paragraph continued on the next page, a complexity
table. Which parts were missed is not a judgement call. Every agent records the
chunks it was shown, so the unread content of the paper is a set difference.

That set is split, in document order, into batches that fit one prompt, and
each batch is read by this agent. Selection, batching and ordering depend only
on the paper and on what the specialists were shown, so the same paper and the
same specialist contexts always produce the same sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..domain.enums import ElementType
from ..domain.models import (
    AgentResult,
    AgentUsage,
    CoverageDetail,
    CoverageResult,
    Finding,
    PaperChunk,
)
from ..response.reader import DocumentReader, clean_text, quotes_from_text
from ..retrieval.index import PaperIndex
from ..retrieval.retriever import RetrievalQuery, RetrievalResult
from ..validation.evidence import validate_evidence
from ..validation.values import value_in_text
from .base import _NON_CONTENT_SECTIONS, MAX_CONTEXT_CHARS, AgentContext, BaseAgent, fit_context
from .grounding import Grounder, parse_findings
from .prompts.registry import COVERAGE

#: A chunk shorter than this is a stray label ("Scaled Dot-Product Attention"),
#: not content worth a model call.
MIN_CONTENT_CHARS = 40
#: Chunks per sweep prompt; the character budget usually binds first.
BATCH_CHUNKS = 16

_DETAIL_COLUMNS: dict[str, tuple[str, ...]] = {
    "name": ("setting", "parameter", "what"),
    "value": ("amount", "number"),
    "source": ("sources", "citation", "ref"),
    "quote": ("evidence", "verbatim quote"),
}


def is_content(chunk: PaperChunk) -> bool:
    """Whether a chunk states something about the paper's own work."""
    section = (chunk.section or "").lower()
    return (
        chunk.element_type is not ElementType.REFERENCE
        and not section.startswith(_NON_CONTENT_SECTIONS)
        and len(chunk.text.strip()) >= MIN_CONTENT_CHARS
    )


@dataclass(slots=True)
class CoveragePlan:
    """What the specialists read, and the batches the sweep will read."""

    content: list[PaperChunk]
    read: set[str]
    batches: list[list[PaperChunk]]
    unread: list[PaperChunk]


def plan_coverage(
    index: PaperIndex, agent_results: dict[str, AgentResult], *, max_batches: int
) -> CoveragePlan:
    """Split the content no agent was shown into prompt-sized batches.

    A table or figure is never split across batches: its caption travels with
    its body. Batches past ``max_batches`` are left unread and reported.
    """
    content = [c for c in index.chunks if is_content(c)]
    read = {
        chunk_id
        for result in agent_results.values()
        if result.status == "ok"
        for chunk_id in result.context_chunk_ids
    }
    pending = [c for c in content if c.chunk_id not in read]

    units: list[list[PaperChunk]] = []
    placed: set[str] = set()
    for chunk in pending:
        if chunk.chunk_id in placed:
            continue
        members = (
            [m for m in index.by_element_id(chunk.element_id) if is_content(m)]
            if chunk.element_id
            else [chunk]
        ) or [chunk]
        placed.update(m.chunk_id for m in members)
        units.append(members)

    batches: list[list[PaperChunk]] = []
    current: list[PaperChunk] = []
    for unit in units:
        candidate = current + unit
        if current and (
            len(candidate) > BATCH_CHUNKS or len(fit_context(_as_results(candidate), index=index)) < len(candidate)
        ):
            batches.append(current)
            current = []
        current = current + unit
    if current:
        batches.append(current)

    swept = batches[:max_batches]
    unread = [c for batch in batches[max_batches:] for c in batch]
    return CoveragePlan(content=content, read=read, batches=swept, unread=unread)


def _as_results(chunks: list[PaperChunk]) -> list[RetrievalResult]:
    return [RetrievalResult(chunk=c, score=0.0, matched_by=["coverage"]) for c in chunks]


class CoverageAgent(BaseAgent):
    """Reads one batch of otherwise unread content."""

    name = "coverage"
    prompt = COVERAGE
    context_limit = BATCH_CHUNKS

    def __init__(self, provider, *, chunks: list[PaperChunk]) -> None:
        super().__init__(provider)
        self.chunks = chunks

    def queries(self, context: AgentContext) -> list[RetrievalQuery]:
        return []

    def retrieve(self, context: AgentContext) -> list[RetrievalResult]:
        return fit_context(
            _as_results(self.chunks), max_chars=MAX_CONTEXT_CHARS, index=context.index
        )

    def build_result(
        self, reader: DocumentReader, context: AgentContext, grounder: Grounder
    ) -> tuple[list[Finding], dict[str, Any], list[str]]:
        findings = parse_findings(reader, grounder, category="coverage")
        details: list[CoverageDetail] = []
        for table in reader.tables(reader.section("settings and values", "settings", "values")):
            for record in reader.records(table, _DETAIL_COLUMNS):
                name, value = (
                    clean_text(record[key].value) if record.get(key) is not None else ""
                    for key in ("name", "value")
                )
                if not (name and value):
                    continue
                evidence = grounder.evidence_list(
                    [
                        ref
                        for cell in (record.get("quote"),)
                        if cell is not None
                        for ref in _quote_refs(cell, record.get("source"))
                    ]
                )
                for piece in evidence:
                    validate_evidence(piece, context.index)
                details.append(
                    CoverageDetail(
                        name=name,
                        value=value,
                        evidence=evidence,
                        value_found=any(
                            e.is_verified
                            and (value_in_text(value, e.quote) or value.strip() in e.quote)
                            for e in evidence
                        ),
                    )
                )
        payload = CoverageResult(details=details, findings=findings)
        return findings, payload.model_dump(mode="json", exclude={"findings"}), []


def _quote_refs(quote_cell, source_cell):
    refs = quotes_from_text(quote_cell.text)
    labels = list(quote_cell.citations) + (list(source_cell.citations) if source_cell else [])
    for ref in refs:
        if not ref.citations:
            ref.citations = labels
    return refs


def _total_usage(usages: list[AgentUsage]) -> AgentUsage | None:
    if not usages:
        return None
    return AgentUsage(
        agent="coverage",
        prompt_version=usages[0].prompt_version,
        model=usages[0].model,
        provider=usages[0].provider,
        prompt_tokens=sum(u.prompt_tokens for u in usages),
        completion_tokens=sum(u.completion_tokens for u in usages),
        total_tokens=sum(u.total_tokens for u in usages),
        latency_ms=round(sum(u.latency_ms for u in usages), 2),
        estimated_cost_usd=(
            sum(u.estimated_cost_usd for u in usages if u.estimated_cost_usd is not None)
            if any(u.estimated_cost_usd is not None for u in usages)
            else None
        ),
    )


def merge_coverage_results(
    results: list[AgentResult], plan: CoveragePlan, index: PaperIndex
) -> AgentResult:
    """One envelope for every batch, with the coverage accounting filled in."""
    ok = [r for r in results if r.status == "ok"]
    swept = {chunk_id for r in ok for chunk_id in r.context_chunk_ids}
    # The same setting read twice (a caption repeated on several pages) is listed once.
    details: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for result in ok:
        for detail in result.payload.get("details", []):
            key = (detail["name"].strip().lower(), detail["value"].strip().lower())
            if key not in seen:
                seen.add(key)
                details.append(detail)
    sections_read = list(
        dict.fromkeys(
            c.citation_section for c in plan.content if c.chunk_id in swept and c.citation_section
        )
    )
    unread_sections = list(
        dict.fromkeys(
            c.citation_section
            for c in plan.content
            if c.chunk_id not in swept and c.chunk_id not in plan.read and c.citation_section
        )
    )
    payload = CoverageResult(
        details=[CoverageDetail(**d) for d in details],
        sections_read=sections_read,
        content_chunks=len(plan.content),
        read_by_specialists=sum(1 for c in plan.content if c.chunk_id in plan.read),
        # Only what the specialists left: the sweep may re-read a chunk one of
        # them already covered, and counting it on both sides reports more
        # chunks read than the paper has.
        read_by_sweep=sum(
            1 for c in plan.content if c.chunk_id in swept and c.chunk_id not in plan.read
        ),
        unread_sections=unread_sections,
    ).model_dump(mode="json", exclude={"findings"})

    failed = [r for r in results if r.status == "failed"]
    status = "skipped" if not results else "ok" if ok else "failed"
    order = {c.chunk_id: position for position, c in enumerate(index.chunks)}
    return AgentResult(
        agent="coverage",
        status=status,  # type: ignore[arg-type]
        findings=[f for r in ok for f in r.findings],
        payload=payload,
        error="; ".join(r.error or "" for r in failed) or None,
        skip_reason=None if results else "every content chunk was read by a specialist",
        usage=_total_usage([r.usage for r in ok if r.usage]),
        attempt=1,
        context_chunk_ids=sorted(swept, key=order.__getitem__),
    )
