"""HTTP API contract."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from paperpilot.api.app import create_app
from paperpilot.api.tasks import InlineTaskRunner, ThreadPoolTaskRunner
from paperpilot.domain.enums import AnalysisStatus

ACCOUNT = {"name": "Ada", "email": "ada@example.com", "password": "correct horse"}


@contextmanager
def _signed_in(service, account=ACCOUNT, **settings):
    """A client for ``service`` with ``settings`` applied, signed in as ``account``."""
    for name, value in settings.items():
        setattr(service.settings, name, value)
    app = create_app(service.settings, service=service, task_runner=InlineTaskRunner())
    with TestClient(app) as test_client:
        if account:
            response = test_client.post("/auth/signup", json=account)
            assert response.status_code == 201, response.text
        yield test_client


@pytest.fixture
def client(service):
    with _signed_in(service) as test_client:
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
    with _signed_in(service, max_upload_mb=0.0001) as client:
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
    assert "# PaperMind Report" in response.text


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
    assert "PaperMind" in response.text


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


def test_model_responses_are_exposed_as_markdown_and_parsed_documents(client, pdf_bytes):
    from paperpilot.response import ResponseDocument

    document_id = _upload(client, pdf_bytes).json()["document_id"]
    analysis_id = client.post(f"/papers/{document_id}/analyze").json()["analysis_id"]
    body = client.get(f"/analyses/{analysis_id}/responses").json()

    summary = body["agents"]["summary"]
    assert summary["response_markdown"].startswith("## ")
    assert ResponseDocument.model_validate(summary["response_document"]).sections
    assert body["verification"] and body["verification"][0]["response_document"]


def test_findings_endpoint_omits_raw_model_responses(client, pdf_bytes):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    analysis_id = client.post(f"/papers/{document_id}/analyze").json()["analysis_id"]
    verdict = client.get(f"/analyses/{analysis_id}/findings").json()["verification_results"][0]
    assert "response_markdown" not in verdict


def test_visuals_endpoint_returns_chartable_data(client, pdf_bytes):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    analysis_id = client.post(f"/papers/{document_id}/analyze").json()["analysis_id"]
    body = client.get(f"/analyses/{analysis_id}/visuals").json()

    assert body["paper"]["page_count"] > 0
    assert set(body["story"]) == {"problem", "approach", "result", "limitations"}
    assert body["verification"] and body["verification"][0]["status"]
    for row in body["results"]:
        # Only numbers whose evidence was re-found in the paper are charted.
        assert row["page"] >= 1
        assert row["number"] is None or isinstance(row["number"], float)


def test_visuals_are_refused_before_the_analysis_completes(client, pdf_bytes, service):
    document_id = _upload(client, pdf_bytes).json()["document_id"]
    record = service.create_analysis(document_id, owner_id=client.get("/auth/me").json()["user_id"])
    response = client.get(f"/analyses/{record.analysis_id}/visuals")
    assert response.status_code == 409


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("81.4", (81.4, "")),
        ("81.4 ± 0.3", (81.4, "")),
        ("45.2%", (45.2, "%")),
        ("3.1x fewer FLOPs", (3.1, "×")),
        ("1,234", (1234.0, "")),
        ("n/a", (None, "")),
        ("12 examples", (12.0, "")),
    ],
)
def test_reported_values_are_parsed_for_charts(value, expected):
    from paperpilot.reporting.visuals import parse_number

    assert parse_number(value) == expected


def test_response_document_schema_is_published(client):
    schema = client.get("/schemas/response-document").json()
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert {"TableCell", "Section"} <= set(schema["$defs"])


# --- access control --------------------------------------------------------


def test_without_a_session_every_api_call_is_refused(service, pdf_bytes):
    with _signed_in(service, account=None) as client:
        assert _upload(client, pdf_bytes).status_code == 401
        assert client.get("/papers").status_code == 401
        response = client.get("/analyses", headers={"Authorization": "Bearer forged"})
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"


def test_the_ui_shell_health_docs_and_sign_in_stay_public(service):
    with _signed_in(service, account=None) as client:
        for path in ("/", "/health", "/openapi.json", "/auth/config"):
            assert client.get(path).status_code == 200, path


def test_sign_up_sets_a_session_cookie_and_returns_a_bearer_token(service):
    with _signed_in(service, account=None) as client:
        response = client.post("/auth/signup", json=ACCOUNT)
        assert response.status_code == 201
        assert "httponly" in response.headers["set-cookie"].lower()
        token = response.json()["token"]
        assert client.get("/auth/me").json()["email"] == "ada@example.com"
        client.cookies.clear()
        assert client.get("/auth/me").status_code == 401
        assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_log_in_log_out_and_wrong_passwords(service):
    with _signed_in(service) as client:
        assert client.post("/auth/logout").status_code == 204
        assert client.get("/auth/me").status_code == 401
        wrong = client.post("/auth/login", json={"email": "ada@example.com", "password": "nope nope"})
        unknown = client.post("/auth/login", json={"email": "who@example.com", "password": "nope nope"})
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json()["message"] == unknown.json()["message"]
        ok = client.post("/auth/login", json={"email": "ADA@example.com ", "password": "correct horse"})
        assert ok.status_code == 200
        assert client.get("/auth/me").status_code == 200


def test_sign_up_validates_and_refuses_a_second_account_for_the_same_email(service):
    with _signed_in(service) as client:
        assert client.post("/auth/signup", json=ACCOUNT).status_code == 409
        short = client.post("/auth/signup", json={**ACCOUNT, "email": "b@example.com", "password": "short"})
        assert short.status_code == 400
        bad_email = client.post("/auth/signup", json={**ACCOUNT, "email": "not-an-email"})
        assert bad_email.status_code == 400


def test_an_invite_code_is_required_when_configured(service):
    with _signed_in(service, account=None, invite_code="let-me-in") as client:
        assert client.get("/auth/config").json()["invite_required"] is True
        assert client.post("/auth/signup", json=ACCOUNT).status_code == 403
        wrong = client.post("/auth/signup", json={**ACCOUNT, "invite_code": "guess"})
        assert wrong.status_code == 403
        ok = client.post("/auth/signup", json={**ACCOUNT, "invite_code": "let-me-in"})
        assert ok.status_code == 201
        assert not any("invite code" in w for w in client.get("/health").json()["warnings"])


def test_health_warns_when_the_ocr_server_is_unreachable(service):
    # Port 9 (discard) refuses connections, so the availability probe fails fast.
    with _signed_in(
        service, ocr_enabled=True, ocr_base_url="http://127.0.0.1:9/v1"
    ) as client:
        warnings = " ".join(client.get("/health").json()["warnings"])
    assert "PaddleOCR-VL server at http://127.0.0.1:9/v1 is not reachable" in warnings


def test_health_warns_when_sign_up_is_open(client):
    assert any("invite code" in w for w in client.get("/health").json()["warnings"])


def test_each_account_sees_only_its_own_papers_and_analyses(service, pdf_bytes):
    with _signed_in(service) as ada:
        document_id = _upload(ada, pdf_bytes).json()["document_id"]
        analysis_id = ada.post(f"/papers/{document_id}/analyze", json={"wait": True}).json()["analysis_id"]
        assert [a["analysis_id"] for a in ada.get("/analyses").json()] == [analysis_id]
        assert ada.get("/analyses").json()[0]["status"] == "completed"

        ada.cookies.clear()
        bob = {"name": "Bob", "email": "bob@example.com", "password": "another secret"}
        assert ada.post("/auth/signup", json=bob).status_code == 201
        assert ada.get("/papers").json() == []
        assert ada.get("/analyses").json() == []
        assert ada.get(f"/papers/{document_id}").status_code == 404
        assert ada.get(f"/analyses/{analysis_id}").status_code == 404
        assert ada.get(f"/analyses/{analysis_id}/report").status_code == 404
        assert ada.post(f"/papers/{document_id}/analyze").status_code == 404


def test_the_first_account_inherits_papers_uploaded_before_accounts_existed(service, paper_pdf):
    import io

    earlier = service.register_upload(io.BytesIO(paper_pdf.read_bytes()), filename="old.pdf")
    with _signed_in(service) as client:
        assert [p["document_id"] for p in client.get("/papers").json()] == [earlier.document_id]


def test_changing_the_password_signs_out_other_sessions(service):
    with _signed_in(service, account=None) as client:
        here = {"Authorization": f"Bearer {client.post('/auth/signup', json=ACCOUNT).json()['token']}"}
        login = {"email": ACCOUNT["email"], "password": ACCOUNT["password"]}
        elsewhere = {"Authorization": f"Bearer {client.post('/auth/login', json=login).json()['token']}"}
        wrong = client.post(
            "/auth/password", json={"current_password": "x" * 9, "new_password": "y" * 9}, headers=here
        )
        assert wrong.status_code == 401
        changed = client.post(
            "/auth/password",
            json={"current_password": ACCOUNT["password"], "new_password": "brand new pass"},
            headers=here,
        )
        assert changed.status_code == 204
        assert client.get("/auth/me", headers=here).status_code == 200
        assert client.get("/auth/me", headers=elsewhere).status_code == 401
        assert client.post("/auth/login", json={**login, "password": "brand new pass"}).status_code == 200


def test_sign_in_attempts_are_rate_limited(service):
    with _signed_in(service, account=None) as client:
        codes = [
            client.post("/auth/login", json={"email": "a@b.co", "password": "wrong pass"}).status_code
            for _ in range(21)
        ]
    assert codes[:20] == [401] * 20 and codes[20] == 429


def test_uploads_and_analyses_are_rate_limited_per_client(service, pdf_bytes):
    with _signed_in(service, rate_limit_requests=2, rate_limit_window_seconds=60) as client:
        document_id = _upload(client, pdf_bytes).json()["document_id"]
        assert client.post(f"/papers/{document_id}/analyze", json={"wait": True}).status_code == 202
        limited = _upload(client, pdf_bytes)
        assert limited.status_code == 429
        assert 1 <= int(limited.headers["Retry-After"]) <= 60
        # Reads are not costly and are never limited: the UI polls them.
        assert client.get(f"/papers/{document_id}").status_code == 200


def test_an_oversized_upload_is_refused_from_its_declared_length(service):
    with _signed_in(service, max_upload_mb=0.01) as client:
        response = _upload(client, b"%PDF-" + b"0" * 200_000)
    assert response.status_code == 413
    assert not list(service.files.upload_dir.iterdir())


def test_an_oversized_upload_without_a_length_is_cut_off_while_streaming(service):
    boundary = "pp-boundary"

    def body():
        yield (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"big.pdf\"\r\n"
            "Content-Type: application/pdf\r\n\r\n%PDF-"
        ).encode()
        for _ in range(64):
            yield b"0" * 16_384
        yield f"\r\n--{boundary}--\r\n".encode()

    with _signed_in(service, max_upload_mb=0.01) as client:
        response = client.post(
            "/papers", content=body(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"
    assert not list(service.files.upload_dir.iterdir())


def test_the_rate_limiter_window_slides():
    from paperpilot.api.guard import RateLimiter

    now = [0.0]
    limiter = RateLimiter(2, 10, clock=lambda: now[0])
    assert limiter.hit("a") is None and limiter.hit("a") is None
    assert limiter.hit("a") == 10
    assert limiter.hit("b") is None
    now[0] = 10.5
    assert limiter.hit("a") is None


def test_storage_stops_writing_an_upload_past_the_cap(tmp_path):
    import io

    from paperpilot.errors import FileTooLargeError
    from paperpilot.storage.files import LocalFileStorage

    storage = LocalFileStorage(upload_dir=tmp_path / "u", report_dir=tmp_path / "r")
    with pytest.raises(FileTooLargeError):
        storage.save_upload(io.BytesIO(b"x" * 3_000_000), filename="a.pdf", document_id="d", max_bytes=1_000_000)
    assert not list((tmp_path / "u").iterdir())
