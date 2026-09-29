"""Readable terminal logs and HTTP request-line filtering."""

from __future__ import annotations

import io
import logging

import pytest

from paperpilot.log_render import HttpAccessFilter, HttpFormatter, PrettyFormatter
from paperpilot.observability import configure_logging, info, log_context, resolve_format


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def pretty_stream():
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="pretty", stream=stream)
    yield stream
    configure_logging(level="ERROR", fmt="text")


def _access(method: str, path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1,
        '%s - "%s %s HTTP/%s" %d', ("127.0.0.1:50953", method, path, "1.1", status), None,
    )


@pytest.mark.parametrize(
    ("mode", "method", "path", "status", "shown"),
    [
        ("changes", "GET", "/analyses/analysis-0593f8d3af99", 200, False),
        ("changes", "GET", "/", 304, False),
        ("changes", "POST", "/papers", 201, True),
        ("changes", "POST", "/papers/paper-1/analyze", 202, True),
        ("changes", "GET", "/analyses/missing", 404, True),
        ("changes", "GET", "/analyses/x/report", 500, True),
        ("errors", "POST", "/papers", 201, False),
        ("errors", "GET", "/analyses/missing", 404, True),
        ("all", "GET", "/analyses/x", 200, True),
        ("none", "GET", "/analyses/x", 500, False),
    ],
)
def test_http_request_lines_are_filtered_by_mode(mode, method, path, status, shown):
    assert HttpAccessFilter(mode).filter(_access(method, path, status)) is shown


def test_kept_http_lines_read_as_method_path_and_outcome():
    line = HttpFormatter().format(_access("POST", "/papers", 201))
    assert line.endswith("HTTP         POST /papers → 201 Created")


def test_auto_format_is_readable_on_a_terminal_and_json_otherwise():
    assert resolve_format("auto", _Tty()) == "pretty"
    assert resolve_format("auto", io.StringIO()) == "json"
    assert resolve_format("text", _Tty()) == "text"


def test_events_render_as_sentences_with_a_stage_and_run_tag(pretty_stream):
    with log_context(analysis_id="analysis-0593f8d3af99", document_id="paper-1"):
        info(
            "agent.model_answered",
            agent="methodology",
            latency_ms=71322.4,
            prompt_tokens=3120,
            completion_tokens=1876,
            chars=5012,
        )
    line = pretty_stream.getvalue().strip()

    assert "Methodology" in line
    assert "Model answered in 71.3 s — 3,120 prompt + 1,876 completion tokens" in line
    assert line.endswith("[0593f8]")
    assert "document_id" not in line


def test_progress_events_reach_the_terminal_and_duplicates_do_not(pretty_stream):
    from paperpilot.orchestration.events import EventRecorder

    EventRecorder().completed("agent:math", "5 findings, 3 with verified evidence", agent="math")
    info("agent.completed", agent="math", findings=5)
    lines = pretty_stream.getvalue().strip().splitlines()

    assert len(lines) == 1
    assert "✓ Math" in lines[0] and "5 findings, 3 with verified evidence" in lines[0]


def test_an_event_without_a_template_still_shows_its_fields(pretty_stream):
    info("custom.event", widgets=3)
    assert "custom.event widgets=3" in pretty_stream.getvalue()


def test_json_format_still_carries_every_field():
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)
    try:
        info("agent.context", agent="math", sources=14)
    finally:
        configure_logging(level="ERROR", fmt="text")
    assert '"event": "agent.context"' in stream.getvalue() and '"sources": 14' in stream.getvalue()


def test_pretty_formatter_hides_redundant_events_only_by_name():
    record = logging.LogRecord("paperpilot", logging.INFO, __file__, 1, "agent.started", None, None)
    assert PrettyFormatter.is_redundant(record)
    record.msg = "agent.context"
    assert not PrettyFormatter.is_redundant(record)
