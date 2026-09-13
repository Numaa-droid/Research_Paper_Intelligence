"""HTTP API contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from paperpilot.api.app import create_app
from paperpilot.api.tasks import InlineTaskRunner, ThreadPoolTaskRunner
from paperpilot.domain.enums import AnalysisStatus


@pytest.fixture
def client(service):
    app = create_app(service.settings, service=service, task_runner=InlineTaskRunner())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def pdf_bytes(paper_pdf):
    return paper_pdf.read_bytes()


def _upload(client, pdf_bytes, *, name="paper.pdf", content_type="application/pdf"):
    return client.post("/papers", files={"file": (name, pdf_bytes, content_type)})


# --- health ----------------------------------------------------------------


def test_health_reports_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["llm_provider"] == "mock"


def test_health_warns_that_the_mock_provider_is_active(client):
    warnings = " ".join(client.get("/health").json()["warnings"])
    assert "mock provider" in warnings


# --- upload ----------------------------------------------------------------


def test_upload_returns_a_document_id(client, pdf_bytes):
    response = _upload(client, pdf_bytes)
    assert response.status_code == 201
    body = response.json()
    assert body["document_id"].startswith("paper-")
    assert body["file_size_bytes"] > 0


def test_upload_rejects_a_non_pdf_content_type(client):
    response = client.post("/papers", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_file"


def test_upload_rejects_a_file_that_is_not_really_a_pdf(client):
    response = client.post(
        "/papers", files={"file": ("fake.pdf", b"GIF89a not a pdf", "application/pdf")}
    )
    assert response.status_code == 415


def test_upload_rejects_an_oversized_file(service, pdf_bytes):
    service.settings.max_upload_mb = 0.0001
    app = create_app(service.settings, service=service, task_runner=InlineTaskRunner())
    with TestClient(app) as client:
        response = _upload(client, pdf_bytes)
    assert response.status_code == 413


def test_upload_sanitises_a_hostile_filename(client, pdf_bytes, service):
    response = _upload(client, pdf_bytes, name="../../../etc/passwd.pdf")
    document_id = response.json()["document_id"]
    stored = service.resolve_path(document_id)
    assert ".." not in str(stored.name)
    assert stored.parent == service.settings.upload_dir


# --- documents -------------------------------------------------------------


def test_get_document(client, pdf_bytes):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    body = client.get(f"/papers/{document_id}").json()
    assert body["document_id"] == document_id
    assert body["page_numbering"] == "pdf_index"


def test_unknown_document_is_404(client):
    response = client.get("/papers/paper-does-not-exist")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_list_documents(client, pdf_bytes):
    _upload(client, pdf_bytes)
    body = client.get("/papers").json()
    assert isinstance(body, list) and body


# --- analysis --------------------------------------------------------------


def test_analyze_returns_an_analysis_id_immediately(client, pdf_bytes):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    response = client.post(f"/papers/{document_id}/analyze", json={"wait": False})
    assert response.status_code == 202
    assert response.json()["analysis_id"].startswith("analysis-")


def test_analysis_is_pollable_and_reports_progress(client, pdf_bytes):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    analysis_id = client.post(f"/papers/{document_id}/analyze").json()["analysis_id"]

    body = client.get(f"/analyses/{analysis_id}").json()
    assert body["status"] == AnalysisStatus.COMPLETED.value
    assert body["progress"] == 1.0
    assert body["agents"]
    assert body["events"]
    assert body["report_url"] == f"/analyses/{analysis_id}/report"


def test_synchronous_analysis_returns_the_completed_run(client, pdf_bytes):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    body = client.post(f"/papers/{document_id}/analyze", json={"wait": True}).json()
    assert body["status"] == AnalysisStatus.COMPLETED.value
    assert body["finding_counts"]


def test_analysis_exposes_the_plan_and_agent_reasons(client, pdf_bytes):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    analysis_id = client.post(f"/papers/{document_id}/analyze").json()["analysis_id"]
    body = client.get(f"/analyses/{analysis_id}").json()
    assert body["plan"]["rationale"]
    assert set(body["agents"]) >= {"summary", "methodology", "experiments"}


def test_analyzing_an_unknown_document_is_404(client):
    assert client.post("/papers/paper-missing/analyze").status_code == 404


def test_unknown_analysis_is_404(client):
    assert client.get("/analyses/analysis-missing").status_code == 404


# --- report ----------------------------------------------------------------


def test_report_is_markdown_with_citations(client, pdf_bytes):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    analysis_id = client.post(f"/papers/{document_id}/analyze").json()["analysis_id"]

    response = client.get(f"/analyses/{analysis_id}/report")
    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert "[p. " in response.text
    assert "# PaperPilot Report" in response.text


def test_report_as_json_returns_the_full_record(client, pdf_bytes):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    analysis_id = client.post(f"/papers/{document_id}/analyze").json()["analysis_id"]
    body = client.get(f"/analyses/{analysis_id}/report?format=json").json()
    assert body["analysis_id"] == analysis_id
    assert body["findings"]


def test_findings_endpoint_exposes_structured_results(client, pdf_bytes):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    analysis_id = client.post(f"/papers/{document_id}/analyze").json()["analysis_id"]
    body = client.get(f"/analyses/{analysis_id}/findings").json()
    assert body["findings"] and body["verification_results"]
    assert body["stats"]["agents_run"]


def test_report_for_an_incomplete_analysis_is_409(client, pdf_bytes):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    record = client.post(f"/papers/{document_id}/analyze", json={"wait": False}).json()
    # Create a second, unstarted analysis to observe the not-ready state.
    from paperpilot.service import PaperPilotService  # noqa: F401

    service = client.app.state.service
    pending = service.create_analysis(document_id)
    response = client.get(f"/analyses/{pending.analysis_id}/report")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "report_not_ready"
    assert record["status"] == AnalysisStatus.COMPLETED.value


# --- UI and docs -----------------------------------------------------------


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "PaperPilot" in response.text


def test_openapi_schema_is_available(client):
    schema = client.get("/openapi.json").json()
    assert "/papers" in schema["paths"]
    assert "/analyses/{analysis_id}/report" in schema["paths"]


# --- background execution --------------------------------------------------


def test_thread_pool_runner_reports_running_state():
    import threading

    runner = ThreadPoolTaskRunner(max_workers=1)
    release = threading.Event()
    try:
        runner.submit("job", release.wait)
        assert runner.is_running("job") is True
        release.set()
    finally:
        runner.shutdown()


def test_thread_pool_runner_does_not_double_submit():
    import threading

    runner = ThreadPoolTaskRunner(max_workers=1)
    release = threading.Event()
    calls: list[int] = []

    def task() -> None:
        calls.append(1)
        release.wait(timeout=2)

    try:
        runner.submit("job", task)
        runner.submit("job", task)
        release.set()
    finally:
        runner.shutdown()
    assert len(calls) <= 1


def test_a_failing_task_does_not_escape(caplog):
    runner = ThreadPoolTaskRunner(max_workers=1)

    def boom() -> None:
        raise RuntimeError("task exploded")

    try:
        runner.submit("job", boom)
    finally:
        runner.shutdown()
