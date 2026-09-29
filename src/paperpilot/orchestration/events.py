"""Progress events.

The user-facing answer to "what is the system doing, and why did it decide that?"
Every stage emits one, they are persisted with the analysis, and the API exposes
them so a polling client can render live progress without guessing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..domain.models import ProgressEvent
from ..observability import log_event

#: A sink receives every event as it happens (CLI printer, websocket, log).
EventSink = Callable[[ProgressEvent], None]


class EventRecorder:
    """Collects progress events and forwards them to an optional sink."""

    def __init__(self, sink: EventSink | None = None) -> None:
        self.events: list[ProgressEvent] = []
        self.sink = sink

    def emit(self, stage: str, status: str, message: str = "", **detail) -> ProgressEvent:
        event = ProgressEvent(stage=stage, status=status, message=message, detail=detail)
        self.events.append(event)
        # The same sentence the UI shows, so a terminal tells the same story.
        level = logging.WARNING if status == "failed" else logging.INFO
        log_event(level, "progress", stage=stage, status=status, summary=message, **detail)
        if self.sink is not None:
            self.sink(event)
        return event

    def started(self, stage: str, message: str = "", **detail) -> ProgressEvent:
        return self.emit(stage, "started", message, **detail)

    def completed(self, stage: str, message: str = "", **detail) -> ProgressEvent:
        return self.emit(stage, "completed", message, **detail)

    def skipped(self, stage: str, message: str = "", **detail) -> ProgressEvent:
        return self.emit(stage, "skipped", message, **detail)

    def failed(self, stage: str, message: str = "", **detail) -> ProgressEvent:
        return self.emit(stage, "failed", message, **detail)

    def drain(self) -> list[ProgressEvent]:
        """Return the events collected so far and clear the buffer."""
        events, self.events = self.events, []
        return events
