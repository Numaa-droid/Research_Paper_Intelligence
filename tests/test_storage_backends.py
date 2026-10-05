"""The storage contract, run against every backend.

SQLite always runs. PostgreSQL (the Supabase backend) runs when
``PAPERPILOT_TEST_DATABASE_URL`` points at a database the tests may write to;
each test gets its own throwaway schema, dropped afterwards.
"""

from __future__ import annotations

import io
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from paperpilot.domain.enums import AnalysisStatus
from paperpilot.domain.models import AnalysisRecord, PaperMetadata
from paperpilot.errors import ConflictError
from paperpilot.storage.factory import build_repositories
from paperpilot.storage.repository import User

PG_URL = os.environ.get("PAPERPILOT_TEST_DATABASE_URL")
BACKENDS = ["sqlite", pytest.param("postgres", marks=pytest.mark.skipif(
    not PG_URL, reason="set PAPERPILOT_TEST_DATABASE_URL to test the PostgreSQL backend"))]


@pytest.fixture(params=BACKENDS)
def backend_settings(request, settings):
    if request.param == "sqlite":
        yield settings
        return
    schema = f"pptest_{uuid.uuid4().hex[:10]}"
    yield settings.model_copy(update={"database_url": PG_URL, "database_schema": schema})
    import psycopg

    with psycopg.connect(PG_URL, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


@pytest.fixture
def repos(backend_settings):
    return build_repositories(backend_settings)


def _paper(document_id: str, *, when: datetime | None = None) -> PaperMetadata:
    return PaperMetadata(
        document_id=document_id, filename=f"{document_id}.pdf", title=f"Title {document_id}",
        sha256=f"hash-{document_id}", ingested_at=when or datetime.now(UTC),
    )


def _user(email: str) -> User:
    return User(
        user_id=f"user-{uuid.uuid4().hex[:8]}", email=email, name="Ada",
        password_hash="scrypt$x$y", created_at=datetime.now(UTC),
    )


def test_documents_round_trip_and_keep_their_owner_on_resave(repos):
    documents, _, _ = repos
    documents.save(_paper("p1"), owner_id="user-a")
    resaved = _paper("p1")
    resaved.page_count = 9
    documents.save(resaved)                      # an analysis re-saving parsed metadata
    assert documents.get("p1").page_count == 9
    assert documents.owner_of("p1") == "user-a"
    assert documents.find_by_hash("hash-p1").document_id == "p1"
    assert documents.get("missing") is None


def test_documents_list_by_owner_newest_first_and_unowned_papers_can_be_claimed(repos):
    documents, _, _ = repos
    now = datetime.now(UTC)
    documents.save(_paper("old", when=now - timedelta(days=2)), owner_id="user-a")
    documents.save(_paper("new", when=now), owner_id="user-a")
    documents.save(_paper("theirs"), owner_id="user-b")
    documents.save(_paper("orphan"))
    assert [p.document_id for p in documents.list(owner_id="user-a")] == ["new", "old"]
    assert documents.claim_unowned("user-b") == 1
    assert {p.document_id for p in documents.list(owner_id="user-b")} == {"theirs", "orphan"}
    assert len(documents.list()) == 4


def test_analyses_round_trip_and_list_across_documents(repos):
    documents, analyses, _ = repos
    for d in ("p1", "p2", "p3"):
        documents.save(_paper(d))
    base = datetime.now(UTC)
    for i, d in enumerate(("p1", "p2", "p3")):
        analyses.save(AnalysisRecord(
            analysis_id=f"a{i}", document_id=d, status=AnalysisStatus.PENDING,
            created_at=base + timedelta(seconds=i), updated_at=base,
        ))
    record = analyses.get("a1")
    record.status = AnalysisStatus.COMPLETED
    record.report_markdown = "# Report\n\nQuoted \u0000 byte survives as JSON."
    analyses.save(record)
    assert analyses.get("a1").status is AnalysisStatus.COMPLETED
    assert analyses.get("a1").report_markdown == record.report_markdown
    assert [r.analysis_id for r in analyses.list_for_documents(["p1", "p2"])] == ["a1", "a0"]
    assert analyses.list_for_documents([]) == []
    assert [r.analysis_id for r in analyses.list_for_document("p3")] == ["a2"]


def test_users_and_sessions(repos):
    _, _, users = repos
    ada = _user("ada@example.com")
    users.create(ada)
    with pytest.raises(ConflictError):
        users.create(_user("ada@example.com"))
    assert users.count() == 1
    assert users.get_by_email("ada@example.com").user_id == ada.user_id

    now = datetime.now(UTC)
    users.create_session("live", ada.user_id, now + timedelta(days=1))
    users.create_session("other", ada.user_id, now + timedelta(days=1))
    users.create_session("stale", ada.user_id, now - timedelta(seconds=1))
    assert users.session_user("live", now).email == "ada@example.com"
    assert users.session_user("stale", now) is None
    users.delete_sessions(ada.user_id, keep="live")
    assert users.session_user("other", now) is None
    assert users.session_user("live", now) is not None
    users.delete_session("live")
    assert users.session_user("live", now) is None

    users.update_password(ada.user_id, "scrypt$new$hash")
    assert users.get(ada.user_id).password_hash == "scrypt$new$hash"


def test_the_whole_app_runs_on_the_backend(backend_settings, embedder, llm, paper_pdf):
    """Sign-up, upload, analysis and per-account isolation, end to end."""
    from paperpilot.api.app import create_app
    from paperpilot.api.tasks import InlineTaskRunner
    from paperpilot.service import PaperPilotService

    service = PaperPilotService(backend_settings, llm_provider=llm, embedding_provider=embedder)
    earlier = service.register_upload(io.BytesIO(paper_pdf.read_bytes()), filename="before.pdf")
    app = create_app(backend_settings, service=service, task_runner=InlineTaskRunner())
    with TestClient(app) as client:
        ada = {"name": "Ada", "email": "ada@example.com", "password": "correct horse"}
        assert client.post("/auth/signup", json=ada).status_code == 201
        # The first account inherits the paper uploaded before accounts existed.
        assert [p["document_id"] for p in client.get("/papers").json()] == [earlier.document_id]
        run = client.post(f"/papers/{earlier.document_id}/analyze", json={"wait": True}).json()
        assert run["status"] == "completed"
        listed = client.get("/analyses").json()
        assert listed[0]["analysis_id"] == run["analysis_id"] and listed[0]["findings"] > 0
        assert "Sparse Routing" in client.get(f"/analyses/{run['analysis_id']}/report").text

        client.cookies.clear()
        bob = {"name": "Bob", "email": "bob@example.com", "password": "another secret"}
        assert client.post("/auth/signup", json=bob).status_code == 201
        assert client.get("/papers").json() == []
        assert client.get(f"/analyses/{run['analysis_id']}").status_code == 404


def test_sqlite_data_can_be_copied_into_another_backend(backend_settings, settings, tmp_path):
    from paperpilot.storage.migrate import copy_from_sqlite

    source = tmp_path / "old.sqlite"
    old = build_repositories(settings.model_copy(update={"database_url": f"sqlite:///{source}"}))
    old_docs, old_analyses, old_users = old
    ada = _user("ada@example.com")
    old_users.create(ada)
    old_docs.save(_paper("p1"), owner_id=ada.user_id)
    old_analyses.save(AnalysisRecord(analysis_id="a1", document_id="p1", status=AnalysisStatus.COMPLETED))

    target = build_repositories(backend_settings)
    counts = copy_from_sqlite(source, documents=target[0], analyses=target[1], users=target[2])
    assert counts == {"users": 1, "users_skipped": 0, "documents": 1, "analyses": 1}
    assert target[0].owner_of("p1") == ada.user_id
    assert target[1].get("a1").status is AnalysisStatus.COMPLETED
    assert target[2].get_by_email("ada@example.com").password_hash == ada.password_hash
    again = copy_from_sqlite(source, documents=target[0], analyses=target[1], users=target[2])
    assert again["users_skipped"] == 1 and target[2].count() == 1
