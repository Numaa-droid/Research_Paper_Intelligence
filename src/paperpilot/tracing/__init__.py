"""Self-hosted run tracing: what every agent saw, asked, answered and decided.

Each analysis is recorded as a tree of spans and viewable at ``/tracing``.
Nothing leaves the machine: traces are kept in ``<data_dir>/traces.sqlite``.
"""

from .tracer import (
    NoopSpan,
    Span,
    Trace,
    Tracer,
    add_event,
    clip,
    current_span,
    span,
)

__all__ = [
    "NoopSpan",
    "Span",
    "Trace",
    "Tracer",
    "add_event",
    "clip",
    "current_span",
    "span",
]
