"""Run tracing: span trees, live traces, storage, and the trace API."""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from paperpilot.api.app import create_app
from paperpilot.api.tasks import InlineTaskRunner
from paperpilot.observability import info
from paperpilot.tracing import Tracer, current_span, span
from paperpilot.tracing.store import SQLiteTraceStore
from paperpilot.tracing.tracer import MAX_STRING_CHARS, clip


def _by_name(trace: dict) -> dict[str, list[dict]]:
    named: dict[str, list[dict]] = {}
    for s in trace["spans"]:
        named.setdefault(s["name"], []).append(s)
    return named


# --- tracer ------------------------------------------------------------------


def test_a_span_outside_a_trace_records_nothing():
    with span("orphan", kind="tool") as s:
        s.set_outputs(value=1)
    assert not s.recording
    assert not current_span().recording


def test_spans_nest_under_the_span_that_was_open(tmp_path):
    tracer = Tracer(SQLiteTraceStore(tmp_path / "t.sqlite"))
    with tracer.trace("run", analysis_id="analysis-1", inputs={"paper": "p.pdf"}) as root:
        with span("outer", kind="node") as outer:
            with span("inner", kind="llm") as inner:
                inner.set_metadata(prompt_tokens=10, completion_tokens=5, cost_usd=0.001)
            outer.set_outputs(done=True)

    trace = tracer.latest_for_analysis("analysis-1")
    spans = _by_name(trace)
    assert spans["outer"][0]["parent_id"] == root.span_id
    assert spans["inner"][0]["parent_id"] == spans["outer"][0]["span_id"]
    assert spans["outer"][0]["outputs"] == {"done": True}
    assert all(s["status"] == "ok" and s["end_time"] for s in trace["spans"])
    assert trace["status"] == "ok"
    assert trace["title"] == "p.pdf"
    assert (trace["llm_calls"], trace["total_tokens"], trace["cost_usd"]) == (1, 15, 0.001)


def test_an_exception_marks_the_span_and_the_trace_failed_and_still_raises(tmp_path):
    tracer = Tracer(SQLiteTraceStore(tmp_path / "t.sqlite"))
    with pytest.raises(ValueError), tracer.trace("run", analysis_id="analysis-2"):
        with span("step"):
            raise ValueError("bad input")

    trace = tracer.latest_for_analysis("analysis-2")
    step = _by_name(trace)["step"][0]
    assert step["status"] == "error"
    assert step["error"]["type"] == "ValueError"
    assert "bad input" in step["error"]["message"]
    assert trace["status"] == "error"
    assert trace["error_count"] == 2


def test_spans_opened_in_worker_threads_nest_under_the_parent(tmp_path):
    tracer = Tracer(SQLiteTraceStore(tmp_path / "t.sqlite"))

    def work(name: str) -> None:
        with span(name, kind="agent"):
            with span(f"{name}.llm", kind="llm"):
                pass

    with tracer.trace("run", analysis_id="analysis-3"):
        with span("group", kind="node") as group, ThreadPoolExecutor(max_workers=3) as pool:
            for future in [
                pool.submit(contextvars.copy_context().run, work, n) for n in ("a", "b", "c")
            ]:
                future.result()

    spans = _by_name(tracer.latest_for_analysis("analysis-3"))
    for name in ("a", "b", "c"):
        assert spans[name][0]["parent_id"] == group.span_id
        assert spans[f"{name}.llm"][0]["parent_id"] == spans[name][0]["span_id"]


def test_log_events_attach_to_the_open_span(tmp_path):
    tracer = Tracer(SQLiteTraceStore(tmp_path / "t.sqlite"))
    with tracer.trace("run", analysis_id="analysis-4"), span("step"):
        info("something.happened", detail=42)

    step = _by_name(tracer.latest_for_analysis("analysis-4"))["step"][0]
    assert [(e["name"], e["fields"]) for e in step["events"]] == [
        ("something.happened", {"detail": 42})
    ]


def test_a_running_trace_is_readable_before_it_finishes(tmp_path):
    tracer = Tracer(SQLiteTraceStore(tmp_path / "t.sqlite"))
    with tracer.trace("run", analysis_id="analysis-5"):
        with span("in-flight"):
            live = tracer.latest_for_analysis("analysis-5")
            assert live["status"] == "running"
            assert [s["status"] for s in live["spans"]] == ["running", "running"]
            assert tracer.list()[0]["status"] == "running"
    assert tracer.latest_for_analysis("analysis-5")["status"] == "ok"


def test_a_disabled_tracer_records_nothing(tmp_path):
    tracer = Tracer(None, enabled=False)
    with tracer.trace("run", analysis_id="analysis-6") as root:
        with span("step") as s:
            assert not s.recording
    assert not root.recording
    assert tracer.list() == []


def test_long_values_are_clipped():
    clipped = clip({"text": "x" * (MAX_STRING_CHARS + 10), "items": list(range(1000))})
    assert clipped["text"].endswith("[10 more chars]")
    assert len(clipped["items"]) == 301


# --- store -------------------------------------------------------------------


def test_the_store_keeps_only_the_newest_traces(tmp_path):
    store = SQLiteTraceStore(tmp_path / "t.sqlite", retention=2)
    tracer = Tracer(store)
    for n in range(3):
        with tracer.trace("run", analysis_id=f"analysis-{n}"):
            pass
    listed = store.list()
    assert [t["analysis_id"] for t in listed] == ["analysis-2", "analysis-1"]
    assert store.list(analysis_ids=["analysis-1"])[0]["analysis_id"] == "analysis-1"
    assert store.list(analysis_ids=[]) == []
    assert store.get(listed[0]["trace_id"])["spans"]


# --- the analysis workflow ---------------------------------------------------


def test_an_analysis_is_traced_end_to_end(service, paper_pdf):
    record = service.analyze_file(paper_pdf)
    trace = service.trace_for_analysis(record.analysis_id)
    spans = _by_name(trace)

    assert trace["status"] == "ok"
    root = trace["spans"][0]
    assert root["name"] == "analysis" and root["parent_id"] is None
    assert root["outputs"]["status"] == "completed"

    # Every graph node, in order, directly under the run.
    nodes = [s["name"] for s in trace["spans"] if s["parent_id"] == root["span_id"]]
    assert nodes[:3] == ["ingest", "plan", "specialists"]
    assert nodes[-1] == "synthesize"
    assert any(e["name"] == "graph.route" for e in root["events"])

    # Specialists ran in parallel under their group node.
    group = spans["specialists"][0]
    members = [s for s in trace["spans"] if s["parent_id"] == group["span_id"]]
    assert {m["name"] for m in members} >= {"agent:summary", "agent:methodology"}

    # An agent's inner steps: retrieval, the model call, parsing, validation.
    summary = spans["summary"][0]
    steps = [s["name"] for s in trace["spans"] if s["parent_id"] == summary["span_id"]]
    assert steps[0] == "retrieve_context"
    assert {"parse_markdown", "validate_findings"} <= set(steps)
    assert summary["outputs"]["status"] == "ok"

    llm = next(s for s in trace["spans"] if s["kind"] == "llm" and s["metadata"]["agent"] == "summary")
    assert [m["role"] for m in llm["inputs"]["messages"]] == ["system", "user"]
    assert llm["outputs"]["message"]["content"]
    assert llm["metadata"]["total_tokens"] > 0

    retrieval = spans["retrieval"][0]
    assert retrieval["inputs"]["query"]
    assert retrieval["outputs"]["documents"][0]["text"]

    assert spans["verify_claim"][0]["outputs"]["status"]
    assert trace["llm_calls"] == sum(1 for s in trace["spans"] if s["kind"] == "llm")
    assert trace["total_tokens"] > 0


def test_content_capture_can_be_turned_off(service, paper_pdf):
    service.tracer.capture_content = False
    record = service.analyze_file(paper_pdf)
    trace = service.trace_for_analysis(record.analysis_id)
    llm = next(s for s in trace["spans"] if s["kind"] == "llm")
    assert llm["inputs"]["messages"][1]["content"].endswith("chars not captured]")
    assert llm["metadata"]["total_tokens"] > 0
    doc = _by_name(trace)["retrieval"][0]["outputs"]["documents"][0]
    assert doc["text"].endswith("chars not captured]")
    assert doc["chunk_id"]


# --- API ---------------------------------------------------------------------


ADA = {"name": "Ada", "email": "ada@example.com", "password": "correct horse"}
EVE = {"name": "Eve", "email": "eve@example.com", "password": "battery staple"}


def test_the_trace_api_serves_only_the_owners_traces(service, paper_pdf):
    app = create_app(service.settings, service=service, task_runner=InlineTaskRunner())
    with TestClient(app) as ada, TestClient(app) as eve:
        assert ada.post("/auth/signup", json=ADA).status_code == 201
        assert eve.post("/auth/signup", json=EVE).status_code == 201

        upload = ada.post(
            "/papers", files={"file": ("paper.pdf", paper_pdf.read_bytes(), "application/pdf")}
        )
        analysis_id = ada.post(f"/papers/{upload.json()['document_id']}/analyze").json()[
            "analysis_id"
        ]

        listed = ada.get("/traces").json()
        assert [t["analysis_id"] for t in listed] == [analysis_id]
        trace_id = listed[0]["trace_id"]
        # The runs table shows each run's input and result without loading its spans.
        assert listed[0]["inputs"]["paper"] == "paper.pdf"
        assert listed[0]["outputs"]["status"] == "completed"
        assert "spans" not in listed[0]

        full = ada.get(f"/traces/{trace_id}").json()
        assert full["spans"] and full["status"] == "ok"
        assert ada.get(f"/analyses/{analysis_id}/trace").json()["trace_id"] == trace_id

        assert eve.get("/traces").json() == []
        assert eve.get(f"/traces/{trace_id}").status_code == 404
        assert eve.get(f"/analyses/{analysis_id}/trace").status_code == 404


def test_the_trace_viewer_page_loads_without_a_session(service):
    app = create_app(service.settings, service=service, task_runner=InlineTaskRunner())
    with TestClient(app) as anonymous:
        page = anonymous.get("/tracing")
        assert page.status_code == 200
        assert "Traces" in page.text and "Runs overview" in page.text
        assert anonymous.get("/traces").status_code == 401
