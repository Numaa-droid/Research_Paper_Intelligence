"""Spans, traces and the tracer.

A *trace* is one analysis run. It is a tree of *spans*: the run itself, every
graph node, every agent, and inside an agent each retrieval query, model call
and parsing step. A span records what went in, what came out, how long it took
and anything that was logged while it was open.

The current span lives in a :class:`~contextvars.ContextVar`. The graph engine
already copies the context into its worker threads, so spans opened by agents
running in parallel nest under the group that started them without any
plumbing. Code that runs outside a trace (unit tests, a bare agent) gets a
no-op span, so instrumentation never has to check whether tracing is on.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from ..observability import add_event_listener, warning

#: Longest string kept in a span's inputs or outputs.
MAX_STRING_CHARS = 60_000
#: Most items kept from one list.
MAX_LIST_ITEMS = 300
#: Most log events kept per span.
MAX_EVENTS_PER_SPAN = 200

#: The span that new spans nest under.
_current: ContextVar[Span | None] = ContextVar("paperpilot_current_span", default=None)

#: Fields every log event inside a run already carries; repeating them on each
#: span event adds nothing.
_CONTEXT_FIELDS = frozenset({"analysis_id", "document_id"})


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def clip(value: Any, *, depth: int = 0) -> Any:
    """A JSON-safe copy of ``value`` with long strings and lists cut short."""
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            return value[:MAX_STRING_CHARS] + f"… [{len(value) - MAX_STRING_CHARS} more chars]"
        return value
    if isinstance(value, bool | int | float) or value is None:
        return value
    if depth > 8:
        return str(value)[:200]
    if isinstance(value, dict):
        return {str(k): clip(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        items = list(value)
        clipped = [clip(v, depth=depth + 1) for v in items[:MAX_LIST_ITEMS]]
        if len(items) > MAX_LIST_ITEMS:
            clipped.append(f"… [{len(items) - MAX_LIST_ITEMS} more items]")
        return clipped
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str | int):
        return enum_value
    return str(value)


class Span:
    """One timed step of a run."""

    __slots__ = (
        "span_id", "trace_id", "parent_id", "name", "kind", "start_time", "end_time",
        "status", "inputs", "outputs", "metadata", "error", "events", "_trace",
    )

    def __init__(
        self,
        trace: Trace,
        *,
        name: str,
        kind: str,
        parent_id: str | None,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._trace = trace
        self.span_id = _new_id("span")
        self.trace_id = trace.trace_id
        self.parent_id = parent_id
        self.name = name
        self.kind = kind
        self.start_time = time.time()
        self.end_time: float | None = None
        self.status = "running"
        self.inputs: dict[str, Any] = clip(inputs or {})
        self.outputs: dict[str, Any] = {}
        self.metadata: dict[str, Any] = clip(metadata or {})
        self.error: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []

    # --- recording --------------------------------------------------------

    @property
    def recording(self) -> bool:
        return True

    def content(self, text: str | None) -> str | None:
        """``text`` itself, or only its length when the trace keeps no content."""
        if text is None or self._trace.capture_content:
            return text
        return f"[{len(text)} chars not captured]"

    def set_inputs(self, **fields: Any) -> None:
        with self._trace.lock:
            self.inputs.update(clip(fields))

    def set_outputs(self, **fields: Any) -> None:
        with self._trace.lock:
            self.outputs.update(clip(fields))

    def set_metadata(self, **fields: Any) -> None:
        with self._trace.lock:
            self.metadata.update(clip(fields))

    def add_event(self, name: str, level: str = "info", **fields: Any) -> None:
        with self._trace.lock:
            if len(self.events) >= MAX_EVENTS_PER_SPAN:
                return
            self.events.append(
                {"ts": time.time(), "name": name, "level": level, "fields": clip(fields)}
            )

    def fail(self, message: str, *, error_type: str = "Error", stack: str | None = None) -> None:
        """Mark the span failed without raising; the caller handled the error."""
        with self._trace.lock:
            self.status = "error"
            self.error = {"type": error_type, "message": message[:4000]}
            if stack:
                self.error["stack"] = stack[-6000:]

    def finish(self) -> None:
        with self._trace.lock:
            if self.end_time is None:
                self.end_time = time.time()
            if self.status == "running":
                self.status = "ok"

    def to_dict(self) -> dict[str, Any]:
        end = self.end_time
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start_time": self.start_time,
            "end_time": end,
            "duration_ms": round((end - self.start_time) * 1000, 2) if end else None,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
            "error": self.error,
            "events": list(self.events),
        }


class NoopSpan:
    """Stands in for a span when no trace is active."""

    recording = False
    span_id = None

    def content(self, text: str | None) -> str | None:
        return text

    def set_inputs(self, **fields: Any) -> None: ...
    def set_outputs(self, **fields: Any) -> None: ...
    def set_metadata(self, **fields: Any) -> None: ...
    def add_event(self, name: str, level: str = "info", **fields: Any) -> None: ...
    def fail(self, message: str, *, error_type: str = "Error", stack: str | None = None) -> None: ...


_NOOP = NoopSpan()


class Trace:
    """Every span of one run, in the order they started."""

    def __init__(
        self,
        *,
        name: str,
        trace_id: str | None = None,
        analysis_id: str | None = None,
        document_id: str | None = None,
        capture_content: bool = True,
    ) -> None:
        self.trace_id = trace_id or _new_id("trace")
        self.name = name
        self.analysis_id = analysis_id
        self.document_id = document_id
        self.capture_content = capture_content
        self.lock = threading.RLock()
        self.spans: list[Span] = []
        self.started_at = time.time()
        self.ended_at: float | None = None
        self.status = "running"

    def open_span(self, **kwargs: Any) -> Span:
        span = Span(self, **kwargs)
        with self.lock:
            self.spans.append(span)
        return span

    def summary(self) -> dict[str, Any]:
        """Headline numbers for the run list."""
        with self.lock:
            spans = list(self.spans)
            prompt = completion = 0
            cost = 0.0
            priced = False
            counts: dict[str, int] = {}
            errors = 0
            for span in spans:
                counts[span.kind] = counts.get(span.kind, 0) + 1
                if span.status == "error":
                    errors += 1
                if span.kind == "llm":
                    prompt += int(span.metadata.get("prompt_tokens") or 0)
                    completion += int(span.metadata.get("completion_tokens") or 0)
                    if span.metadata.get("cost_usd") is not None:
                        priced = True
                        cost += float(span.metadata["cost_usd"])
            root = spans[0] if spans else None
            end = self.ended_at
            return {
                "trace_id": self.trace_id,
                "name": self.name,
                "analysis_id": self.analysis_id,
                "document_id": self.document_id,
                "status": self.status,
                "started_at": self.started_at,
                "ended_at": end,
                "duration_ms": round((end - self.started_at) * 1000, 2) if end else None,
                "title": (root.inputs.get("paper") if root else None) or self.name,
                "span_count": len(spans),
                "llm_calls": counts.get("llm", 0),
                "retrievals": counts.get("retriever", 0),
                "agents": counts.get("agent", 0),
                "error_count": errors,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "cost_usd": round(cost, 6) if priced else None,
                "model": root.metadata.get("llm_model") if root else None,
                #: The run's own input and result, for the runs table.
                "inputs": dict(root.inputs) if root else {},
                "outputs": dict(root.outputs) if root else {},
            }

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {**self.summary(), "spans": [s.to_dict() for s in self.spans]}


class Tracer:
    """Opens traces, keeps the running ones in memory, and stores finished ones."""

    def __init__(self, store=None, *, enabled: bool = True, capture_content: bool = True) -> None:
        self.store = store
        self.enabled = enabled
        self.capture_content = capture_content
        self._live: dict[str, Trace] = {}
        self._lock = threading.Lock()

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        analysis_id: str | None = None,
        document_id: str | None = None,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span | NoopSpan]:
        """Run the block as a new trace; yields its root span."""
        if not self.enabled:
            yield _NOOP
            return
        trace = Trace(
            name=name,
            analysis_id=analysis_id,
            document_id=document_id,
            capture_content=self.capture_content,
        )
        root = trace.open_span(
            name=name, kind="chain", parent_id=None, inputs=inputs, metadata=metadata
        )
        with self._lock:
            self._live[trace.trace_id] = trace
        token = _current.set(root)
        try:
            yield root
        except BaseException as exc:
            root.fail(str(exc), error_type=type(exc).__name__, stack=traceback.format_exc())
            raise
        finally:
            _current.reset(token)
            root.finish()
            with trace.lock:
                trace.ended_at = root.end_time
                for span in trace.spans:
                    if span.end_time is None:  # a worker never closed it
                        span.fail("the run ended before this step finished", error_type="Abandoned")
                        span.finish()
                trace.status = "error" if root.status == "error" else "ok"
            self._save(trace)

    def _save(self, trace: Trace) -> None:
        try:
            if self.store is not None:
                self.store.save(trace.to_dict())
        except Exception as exc:  # noqa: BLE001 - losing a trace must never fail the run
            warning("tracing.save_failed", trace_id=trace.trace_id, error=str(exc)[:300])
        finally:
            with self._lock:
                self._live.pop(trace.trace_id, None)

    # --- reading ----------------------------------------------------------

    def live(self) -> list[Trace]:
        with self._lock:
            return list(self._live.values())

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            live = self._live.get(trace_id)
        if live is not None:
            return live.to_dict()
        return self.store.get(trace_id) if self.store is not None else None

    def latest_for_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        for trace in self.live():
            if trace.analysis_id == analysis_id:
                return trace.to_dict()
        if self.store is None:
            return None
        return self.store.latest_for_analysis(analysis_id)

    def list(self, *, analysis_ids: list[str] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Running traces first, then stored ones, newest first."""
        wanted = set(analysis_ids) if analysis_ids is not None else None
        running = [
            t.summary()
            for t in self.live()
            if wanted is None or t.analysis_id in wanted
        ]
        stored = (
            self.store.list(analysis_ids=analysis_ids, limit=limit) if self.store is not None else []
        )
        seen = {s["trace_id"] for s in running}
        merged = running + [s for s in stored if s["trace_id"] not in seen]
        merged.sort(key=lambda s: s["started_at"], reverse=True)
        return merged[:limit]


# --- instrumentation API ----------------------------------------------------


def current_span() -> Span | NoopSpan:
    return _current.get() or _NOOP


@contextmanager
def span(
    name: str,
    *,
    kind: str = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Span | NoopSpan]:
    """Record the block as a child of the current span.

    An exception escaping the block marks the span failed and is re-raised.
    Outside a trace this is a no-op.
    """
    parent = _current.get()
    if parent is None:
        yield _NOOP
        return
    child = parent._trace.open_span(
        name=name, kind=kind, parent_id=parent.span_id, inputs=inputs, metadata=metadata
    )
    token = _current.set(child)
    try:
        yield child
    except BaseException as exc:
        child.fail(str(exc), error_type=type(exc).__name__, stack=traceback.format_exc())
        raise
    finally:
        _current.reset(token)
        child.finish()


def add_event(name: str, level: str = "info", **fields: Any) -> None:
    """Attach an event to the current span, if there is one."""
    current = _current.get()
    if current is not None:
        current.add_event(name, level, **fields)


def _on_log_event(level: int, event: str, fields: dict[str, Any]) -> None:
    """Every INFO-or-louder log line inside a span becomes one of its events."""
    if level < 20:
        return
    current = _current.get()
    if current is None:
        return
    name = "warning" if level == 30 else "error" if level >= 40 else "info"
    current.add_event(
        event, name, **{k: v for k, v in fields.items() if k not in _CONTEXT_FIELDS}
    )


add_event_listener(_on_log_event)
