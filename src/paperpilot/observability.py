"""Structured logging and lightweight tracing.

Every event is structured (a name plus fields). How it is rendered depends on
where it goes:

* ``json`` — one JSON object per line, for log aggregators;
* ``pretty`` — one readable sentence per line with a stage label, for a person
  watching a terminal;
* ``text`` — ``event key=value`` lines;
* ``auto`` (default) — ``pretty`` on an interactive terminal, ``json`` otherwise.

Document text is *not* logged by default: events reference document and chunk
identifiers instead.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_LOGGER_NAME = "paperpilot"
_configured = False

#: Correlation identifiers attached to every event emitted inside a run.
#: The default is ``None`` rather than ``{}`` so the empty mapping can never be
#: mutated and shared across contexts.
_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "paperpilot_log_context", default=None
)


def _current_context() -> dict[str, Any]:
    return _context.get() or {}


#: Called with ``(level, event, fields)`` for every structured event; the tracer
#: uses this to attach log lines to the span that was open when they happened.
EventListener = Callable[[int, str, dict[str, Any]], None]
_listeners: list[EventListener] = []


def add_event_listener(listener: EventListener) -> None:
    if listener not in _listeners:
        _listeners.append(listener)

_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "taskName", "thread", "threadName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _safe(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable single-line formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            k: _safe(v)
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        suffix = " ".join(f"{k}={v}" for k, v in extras.items())
        base = f"{record.levelname:<7} {record.name} {record.getMessage()}"
        return f"{base} {suffix}".rstrip()


def _safe(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return str(value)


def resolve_format(fmt: str, stream) -> str:
    """``auto`` becomes ``pretty`` for a person at a terminal and ``json`` otherwise."""
    if fmt != "auto":
        return fmt
    isatty = getattr(stream, "isatty", None)
    return "pretty" if callable(isatty) and isatty() else "json"


def _wants_color(stream) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty() and not os.environ.get("NO_COLOR"))


def configure_logging(
    level: str = "INFO", fmt: str = "auto", stream=None, *, http: str = "changes"
) -> None:
    """Install the PaperPilot log handler. Safe to call more than once."""
    global _configured
    stream = stream or sys.stderr
    resolved = resolve_format(fmt, stream)
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    if resolved == "json":
        handler.setFormatter(JsonFormatter())
    elif resolved == "pretty":
        from .log_render import PrettyFormatter

        handler.setFormatter(PrettyFormatter(color=_wants_color(stream)))
        handler.addFilter(
            lambda record: logging.getLevelName(level.upper()) <= logging.DEBUG
            or not PrettyFormatter.is_redundant(record)
        )
    else:
        handler.setFormatter(TextFormatter())
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    configure_http_logging(http, pretty=resolved == "pretty", color=_wants_color(stream))
    _configured = True


def configure_http_logging(mode: str, *, pretty: bool = False, color: bool = False) -> None:
    """Decide which HTTP request lines the server prints.

    ``changes`` (default) keeps requests that change something or went wrong —
    uploads, analysis starts, any 4xx/5xx — and drops the successful ``GET``
    polling a browser repeats every second. ``errors`` keeps only 4xx/5xx,
    ``all`` keeps everything and ``none`` nothing.
    """
    from .log_render import HttpAccessFilter, HttpFormatter

    access = logging.getLogger("uvicorn.access")
    for existing in [f for f in access.filters if isinstance(f, HttpAccessFilter)]:
        access.removeFilter(existing)
    access.addFilter(HttpAccessFilter(mode))
    if pretty:
        for handler in access.handlers:
            handler.setFormatter(HttpFormatter(color=color))


def get_logger(name: str | None = None) -> logging.Logger:
    if not _configured:
        configure_logging()
    return logging.getLogger(_LOGGER_NAME if not name else f"{_LOGGER_NAME}.{name}")


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach correlation fields to every event emitted inside the block."""
    token = _context.set({**_current_context(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def _sanitize(fields: dict[str, Any]) -> dict[str, Any]:
    """Rename fields that collide with reserved ``LogRecord`` attributes.

    ``logging`` raises if ``extra`` contains a key such as ``filename`` or
    ``module``, which are natural names for our own event fields. Prefixing the
    collisions keeps the field rather than losing the whole event.
    """
    clean: dict[str, Any] = {}
    for key, value in fields.items():
        clean[f"f_{key}" if key in _RESERVED else key] = value
    return clean


def log_event(level: int, event: str, **fields: Any) -> None:
    """Emit one structured event, merged with the ambient log context."""
    logger = get_logger()
    logger.log(level, event, extra=_sanitize({**_current_context(), **fields}))
    for listener in _listeners:
        listener(level, event, fields)


def info(event: str, **fields: Any) -> None:
    log_event(logging.INFO, event, **fields)


def debug(event: str, **fields: Any) -> None:
    log_event(logging.DEBUG, event, **fields)


def warning(event: str, **fields: Any) -> None:
    log_event(logging.WARNING, event, **fields)


def error(event: str, **fields: Any) -> None:
    log_event(logging.ERROR, event, **fields)


@contextmanager
def timed(event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block and emit ``<event>.started`` / ``<event>.completed``."""
    started = time.perf_counter()
    result: dict[str, Any] = {}
    info(f"{event}.started", **fields)
    try:
        yield result
    except Exception as exc:  # noqa: BLE001 - re-raised after logging
        elapsed = (time.perf_counter() - started) * 1000
        error(
            f"{event}.failed",
            duration_ms=round(elapsed, 2),
            error_type=type(exc).__name__,
            error=str(exc),
            **fields,
        )
        raise
    else:
        elapsed = (time.perf_counter() - started) * 1000
        result["duration_ms"] = round(elapsed, 2)
        info(f"{event}.completed", **{**fields, **result})


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]
