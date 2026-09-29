"""Human-readable rendering of PaperPilot's structured log events.

A structured event such as::

    {"event": "agent.completed", "agent": "math", "findings": 5, "tokens": 4996}

reads, in a terminal, as::

    11:38:49  ✓ Math          5 findings, 3 with verified evidence

Rendering never changes what is logged: the JSON formatter still receives every
field. Each event has a stage label and a sentence built from its fields; an
event without a template falls back to ``name key=value``.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

#: Fields every event carries from the run context; shown as a short run tag.
_CONTEXT_FIELDS = {"analysis_id", "document_id"}
_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "taskName", "thread", "threadName", "color_message",
}

#: Events whose meaning a progress line already states. Shown only at DEBUG.
REDUNDANT_EVENTS = frozenset({
    "agent.started", "agent.skipped", "agent.completed", "agent.failed",
    "graph.parallel_started", "graph.parallel_completed", "graph.completed",
    "workflow.completed", "ingestion.started", "ingestion.completed",
    "planner.plan_created", "verification.completed", "index.built", "task.submitted",
    "retry.decided", "service.upload_registered",
})

_ANSI = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m", "red": "\033[31m",
    "green": "\033[32m", "yellow": "\033[33m", "blue": "\033[34m", "cyan": "\033[36m",
}

_STATUS_MARK = {
    "started": ("▶", "cyan"),
    "completed": ("✓", "green"),
    "skipped": ("○", "dim"),
    "failed": ("✗", "red"),
}

_STAGE_NAMES = {
    "ingestion": "Ingestion",
    "indexing": "Indexing",
    "planning": "Planner",
    "aggregation": "Aggregate",
    "verification": "Verifier",
    "retry": "Retry",
    "report": "Report",
}

_PREFIX_STAGES = {
    "api": "Server",
    "service": "Analysis",
    "task": "Queue",
    "ingestion": "Ingestion",
    "ocr": "OCR",
    "pdf": "Ingestion",
    "index": "Indexing",
    "planner": "Planner",
    "retrieval": "Retrieval",
    "llm": "Model",
    "verification": "Verifier",
    "retry": "Retry",
    "storage": "Storage",
    "validation": "Validation",
    "evaluation": "Evaluation",
    "workflow": "Analysis",
    "graph": "Pipeline",
}


def _n(value: Any) -> str:
    """``4996`` -> ``4,996``; non-numbers pass through."""
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else str(value)


def _count(value: Any, word: str) -> str:
    """``_count(1, "table")`` -> ``1 table``; ``_count(3, "table")`` -> ``3 tables``."""
    return f"{_n(value)} {word}{'' if value == 1 else 's'}"


def _seconds(ms: Any) -> str:
    try:
        return f"{float(ms) / 1000:.1f} s"
    except (TypeError, ValueError):
        return "?"


def _clip(text: Any, limit: int = 90) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _sections(values: Any, limit: int = 3) -> str:
    names = [str(v) for v in (values or [])]
    shown = ", ".join(f"§{name}" for name in names[:limit])
    more = f" +{len(names) - limit} more" if len(names) > limit else ""
    return f"{shown}{more}" if shown else "none"


def _progress(f: dict) -> str:
    message = str(f.get("summary") or "")
    attempt = f.get("attempt")
    if f.get("status") == "started" and isinstance(attempt, int) and attempt > 1:
        message += f" (retry, attempt {attempt})"
    return message


#: event name -> sentence built from the event's fields.
MESSAGES: dict[str, Callable[[dict], str]] = {
    "progress": _progress,
    "api.startup": lambda f: (
        f"PaperPilot {f.get('version')} ready — LLM: {f.get('llm_provider')}"
        f"{' (' + str(f['llm_model']) + ')' if f.get('llm_model') else ''}, "
        f"embeddings: {f.get('embedding_provider')}"
    ),
    "api.shutdown": lambda f: "Server stopped",
    "api.upload": lambda f: f"Upload received: stored as {f.get('document_id')} ({_n(f.get('bytes'))} bytes)",
    "service.upload_registered": lambda f: f"Paper stored as {f.get('document_id')} ({_n(f.get('bytes'))} bytes)",
    "service.analysis_started": lambda f: (
        f"Analysis started for {f.get('paper') or f.get('document_id')} "
        f"using {f.get('llm_provider')}{' / ' + str(f['llm_model']) if f.get('llm_model') else ''}"
    ),
    "service.analysis_finished": lambda f: (
        f"Analysis {f.get('status')} in {f.get('duration_s')} s — {_count(f.get('findings'), 'finding')}, "
        f"{_count(f.get('verified_claims', 0), 'claim')} verified, {_count(f.get('tokens', 0), 'token')}"
        f"{', ~$' + format(f['cost_usd'], '.4f') if f.get('cost_usd') else ''}"
        f"{', ' + str(f['retries']) + ' retry round(s)' if f.get('retries') else ''}"
    ),
    "report.saved": lambda f: f"Report saved to {f.get('path')} ({_n(f.get('chars'))} characters)",
    "task.already_running": lambda f: f"Analysis {f.get('task_id')} is already running; not started twice",
    "task.failed": lambda f: f"Background analysis crashed: {_clip(f.get('error'), 160)}",
    "ingestion.summary": lambda f: (
        f"Found {_count(f.get('sections'), 'section')}, {_count(f.get('equations'), 'equation')}, "
        f"{_count(f.get('tables'), 'table')}, {_count(f.get('figures'), 'figure')}"
        f"{' (' + str(f['ocr_pages']) + ' page(s) read by OCR)' if f.get('ocr_pages') else ''}"
    ),
    "ingestion.ocr_page": lambda f: f"Page {f.get('page')} has no usable text layer; reading it with OCR",
    "ingestion.figure_ocr": lambda f: (
        f"Read a figure on page {f.get('page')} with OCR ({_n(f.get('chart_rows'))} chart rows)"
    ),
    "ingestion.table_ocr": lambda f: (
        f"Re-read a {f.get('rows')}×{f.get('columns')} table on page {f.get('page')} with OCR — "
        f"{'kept' if f.get('accepted') else 'rejected'}"
        f"{': ' + _clip(f['reason'], 80) if f.get('reason') else ''}"
    ),
    "agent.context": lambda f: (
        f"Reading {_count(f.get('sources'), 'source')} ({_n(f.get('chars'))} chars): "
        f"{_n(f.get('structural'))} from the paper's structure "
        f"[{_sections(f.get('anchor_sections'))}], {_n(f.get('searched'))} by search"
    ),
    "agent.model_answered": lambda f: (
        f"Model answered in {_seconds(f.get('latency_ms'))} — {_n(f.get('prompt_tokens'))} prompt + "
        f"{_n(f.get('completion_tokens'))} completion tokens ({_n(f.get('chars'))} chars)"
    ),
    "agent.response_parsed": lambda f: (
        f"Parsed answer: {_count(f.get('sections'), 'section')}, {_count(f.get('tables'), 'table')}, "
        f"{_count(f.get('quotes'), 'quote')}, {_count(f.get('citations'), 'citation')}"
        f"{', ' + str(f['repairs']) + ' formatting repair(s)' if f.get('repairs') else ''}"
    ),
    "agent.response_repaired": lambda f: f"Repaired the answer's Markdown: {_clip(f.get('repairs'), 140)}",
    "agent.response_unstructured": lambda f: (
        f"The answer ({_n(f.get('chars'))} chars) did not follow the Markdown layout; nothing could be read from it"
    ),
    "agent.no_context": lambda f: "No passages were retrieved for this agent",
    "agent.parse_failed": lambda f: f"Could not read the answer: {_clip(f.get('error'), 160)}",
    "grounding.quote_dropped": lambda f: f"Dropped a quote not found in the paper: “{_clip(f.get('quote'), 70)}”",
    "validation.evidence_dropped": lambda f: f"Dropped a citation: {_clip(f.get('reason'), 120)}",
    "validation.finding_dropped": lambda f: f"Dropped a finding: {_clip(f.get('reason'), 120)}",
    "llm.attempt_failed": lambda f: (
        f"Model call failed (attempt {f.get('attempt')}): {_clip(f.get('error'), 120)}"
        f"{'; retrying in ' + str(f['retry_in_s']) + ' s' if f.get('retry_in_s') is not None else ''}"
    ),
    "llm.response_truncated": lambda f: (
        f"Answer was cut off at {_n(f.get('max_tokens'))} tokens; asking again with {_n(f.get('retry_with'))}"
    ),
    "planner.model_failed": lambda f: f"Planning model unavailable ({_clip(f.get('error'), 100)}); using the rule-based plan",
    "verification.claim": lambda f: (
        f"{str(f.get('status', '')).replace('_', ' ').capitalize()} "
        f"(confidence {f.get('confidence')}, numbers {f.get('numeric_accuracy')}): “{_clip(f.get('claim'), 80)}”"
    ),
    "verification.provider_failed": lambda f: f"Verifier model call failed: {_clip(f.get('error'), 120)}",
    "retry.decided": lambda f: (
        f"Re-running {', '.join(f.get('agents') or [])} "
        f"(retry {_n(int(f.get('retry_count') or 0) + 1)} of {f.get('max_retries')})"
    ),
    "workflow.failed": lambda f: f"Analysis failed: {_clip(f.get('error'), 160)}",
    "graph.node_failed": lambda f: f"Stage {f.get('node')} failed: {_clip(f.get('error'), 140)}",
    "graph.node_crashed": lambda f: f"Stage {f.get('node')} crashed: {_clip(f.get('error'), 140)}",
    "ocr.unavailable": lambda f: (
        f"OCR engine {f.get('engine')} unavailable{' at ' + str(f['url']) if f.get('url') else ''}: "
        f"{_clip(f.get('error'), 100)}"
    ),
    "ocr.page_failed": lambda f: f"OCR failed on page {f.get('page')}: {_clip(f.get('error'), 120)}",
}


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}


def _stage(event: str, fields: dict) -> str:
    stage = str(fields.get("stage") or "")
    if event == "progress" and stage:
        if stage.startswith("agent:"):
            return stage.split(":", 1)[1].replace("_", " ").title()
        return _STAGE_NAMES.get(stage, stage.title())
    if fields.get("agent") and event.startswith(("agent.", "grounding.", "llm.", "validation.")):
        return str(fields["agent"]).replace("_", " ").title()
    return _PREFIX_STAGES.get(event.split(".", 1)[0], event.split(".", 1)[0].title())


class PrettyFormatter(logging.Formatter):
    """``HH:MM:SS  mark Stage        sentence  [run]``."""

    def __init__(self, *, color: bool = False) -> None:
        super().__init__()
        self.color = color

    @staticmethod
    def is_redundant(record: logging.LogRecord) -> bool:
        return record.getMessage() in REDUNDANT_EVENTS

    def _paint(self, text: str, style: str) -> str:
        return f"{_ANSI[style]}{text}{_ANSI['reset']}" if self.color and style in _ANSI else text

    def format(self, record: logging.LogRecord) -> str:
        event = record.getMessage()
        fields = _fields(record)
        template = MESSAGES.get(event)
        try:
            sentence = template(fields) if template else None
        except Exception:  # noqa: BLE001 - a rendering bug must never lose the event
            sentence = None
        if not sentence:
            extras = " ".join(
                f"{k}={_clip(v, 60)}" for k, v in fields.items() if k not in _CONTEXT_FIELDS
            )
            sentence = f"{event} {extras}".strip()

        if record.levelno >= logging.ERROR:
            mark, style = "✗", "red"
        elif record.levelno >= logging.WARNING:
            mark, style = "⚠", "yellow"
        elif event == "progress":
            mark, style = _STATUS_MARK.get(str(fields.get("status")), ("·", "blue"))
        else:
            mark, style = "·", "dim"

        clock = time.strftime("%H:%M:%S", time.localtime(record.created))
        run = str(fields.get("analysis_id") or "")
        tag = f"  [{run.removeprefix('analysis-')[:6]}]" if run else ""
        line = (
            f"{self._paint(clock, 'dim')}  {self._paint(mark, style)} "
            f"{self._paint(f'{_stage(event, fields):<12}', 'bold')} {sentence}"
            f"{self._paint(tag, 'dim')}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# --- HTTP access lines -------------------------------------------------------

_READ_METHODS = {"GET", "HEAD", "OPTIONS"}


def _access_parts(record: logging.LogRecord) -> tuple[str, str, str, int] | None:
    """``(client, method, path, status)`` from a uvicorn access record."""
    args = record.args
    if isinstance(args, tuple) and len(args) >= 5:
        try:
            return str(args[0]), str(args[1]), str(args[2]), int(args[4])
        except (TypeError, ValueError):
            return None
    return None


class HttpAccessFilter(logging.Filter):
    """Keeps the request lines worth reading; see ``configure_http_logging``."""

    def __init__(self, mode: str = "changes") -> None:
        super().__init__()
        self.mode = mode

    def filter(self, record: logging.LogRecord) -> bool:
        if self.mode == "all":
            return True
        if self.mode == "none":
            return False
        parts = _access_parts(record)
        if parts is None:
            return True
        _, method, _, status = parts
        if status >= 400:
            return True
        return self.mode == "changes" and method.upper() not in _READ_METHODS


_HTTP_REASONS = {200: "OK", 201: "Created", 202: "Accepted", 204: "No Content"}
_ID_IN_PATH = re.compile(r"(analysis|paper)-[0-9a-f]{6,}")


class HttpFormatter(logging.Formatter):
    """``HH:MM:SS  ↔ HTTP         POST /papers → 201 Created``."""

    def __init__(self, *, color: bool = False) -> None:
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        parts = _access_parts(record)
        clock = time.strftime("%H:%M:%S", time.localtime(record.created))
        if parts is None:
            return f"{clock}  · HTTP         {record.getMessage()}"
        _, method, path, status = parts
        style = "red" if status >= 500 else "yellow" if status >= 400 else "green"
        outcome = f"{status} {_HTTP_REASONS.get(status, '')}".strip()
        if self.color:
            outcome = f"{_ANSI[style]}{outcome}{_ANSI['reset']}"
            clock = f"{_ANSI['dim']}{clock}{_ANSI['reset']}"
        return f"{clock}  ↔ HTTP         {method} {path} → {outcome}"
