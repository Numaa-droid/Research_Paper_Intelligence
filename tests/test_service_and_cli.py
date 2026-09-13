"""Application service, storage repositories, configuration and CLI."""

from __future__ import annotations

import json

import pytest

from paperpilot.cli import EXIT_BAD_INPUT, EXIT_OK, build_parser, main
from paperpilot.config import Settings
from paperpilot.domain.enums import AnalysisStatus
from paperpilot.domain.models import AnalysisRecord, PaperMetadata
from paperpilot.errors import NotFoundError
from paperpilot.storage.files import LocalFileStorage, safe_filename
from paperpilot.storage.sqlite_repo import (
    InMemoryAnalysisRepository,
    SQLiteAnalysisRepository,
    SQLiteConnection,
    SQLiteDocumentRepository,
)

# --- configuration ---------------------------------------------------------


def test_defaults_run_offline():
    settings = Settings()
    assert settings.llm_provider == "mock"
    assert settings.embedding_provider == "hashing"


def test_secrets_are_redacted():
    settings = Settings(llm_api_key="sk-super-secret", embedding_api_key="sk-also-secret")
    redacted = settings.redacted()
    assert redacted["llm_api_key"] == "***redacted***"
    assert "sk-super-secret" not in json.dumps(redacted)


def test_overlap_must_be_smaller_than_the_chunk():
    with pytest.raises(ValueError, match="chunk_overlap_chars"):
        Settings(chunk_max_chars=500, chunk_overlap_chars=500)


def test_derived_paths_sit_under_the_data_dir(tmp_path):
    settings = Settings(data_dir=tmp_path / "d")
    assert settings.upload_dir.parent == settings.data_dir
    assert settings.report_dir.parent == settings.data_dir


def test_non_sqlite_urls_are_rejected():
    with pytest.raises(ValueError, match="sqlite"):
        _ = Settings(database_url="postgresql://localhost/db").sqlite_path


def test_ensure_directories_is_idempotent(tmp_path):
    settings = Settings(data_dir=tmp_path / "d", database_url=f"sqlite:///{tmp_path}/d/x.sqlite")
    settings.ensure_directories()
    settings.ensure_directories()
    assert settings.upload_dir.exists()


# --- storage ---------------------------------------------------------------


def test_sqlite_round_trips_a_document(tmp_path):
    repo = SQLiteDocumentRepository(SQLiteConnection(tmp_path / "db.sqlite"))
    metadata = PaperMetadata(document_id="paper-1", title="A paper", sha256="abc", page_count=4)
    repo.save(metadata)
    loaded = repo.get("paper-1")
    assert loaded.title == "A paper" and loaded.page_count == 4
    assert repo.find_by_hash("abc").document_id == "paper-1"
    assert repo.get("missing") is None


def test_saving_a_document_twice_updates_it(tmp_path):
    repo = SQLiteDocumentRepository(SQLiteConnection(tmp_path / "db.sqlite"))
    repo.save(PaperMetadata(document_id="p", title="first"))
    repo.save(PaperMetadata(document_id="p", title="second"))
    assert repo.get("p").title == "second"
    assert len(repo.list()) == 1


def test_sqlite_round_trips_an_analysis(tmp_path):
    connection = SQLiteConnection(tmp_path / "db.sqlite")
    SQLiteDocumentRepository(connection).save(PaperMetadata(document_id="paper-1"))
    repo = SQLiteAnalysisRepository(connection)
    repo.save(
        AnalysisRecord(
            analysis_id="analysis-1", document_id="paper-1", status=AnalysisStatus.COMPLETED
        )
    )
    assert repo.get("analysis-1").status is AnalysisStatus.COMPLETED
    assert len(repo.list_for_document("paper-1")) == 1


def test_in_memory_repository_matches_the_interface():
    repo = InMemoryAnalysisRepository()
    repo.save(AnalysisRecord(analysis_id="a", document_id="d"))
    assert repo.get("a") is not None
    assert repo.get("missing") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Traversal is defeated by taking the basename, not by escaping it.
        ("../../etc/passwd", "passwd"),
        ("/etc/shadow", "shadow"),
        ("normal-file.pdf", "normal-file.pdf"),
        ("", "upload.pdf"),
        ("with spaces.pdf", "with_spaces.pdf"),
        ("....", "upload.pdf"),
    ],
)
def test_filenames_are_sanitised(raw, expected):
    assert safe_filename(raw) == expected


@pytest.mark.parametrize("raw", ["../../etc/passwd", "a/b/c.pdf", "..%2F..%2Fx.pdf", "\\x.pdf"])
def test_sanitised_filenames_can_never_traverse(raw):
    cleaned = safe_filename(raw)
    assert "/" not in cleaned and "\\" not in cleaned
    assert not cleaned.startswith(".")


def test_file_storage_round_trip(tmp_path):
    storage = LocalFileStorage(upload_dir=tmp_path / "up", report_dir=tmp_path / "rep")
    storage.save_bytes(b"%PDF-1.4", filename="x.pdf", document_id="paper-9")
    assert storage.path_for("paper-9").exists()
    storage.save_report("analysis-9", "# report")
    assert storage.read_report("analysis-9") == "# report"
    assert storage.read_report("missing") is None


def test_require_path_raises_for_an_unknown_document(tmp_path):
    storage = LocalFileStorage(upload_dir=tmp_path / "up", report_dir=tmp_path / "rep")
    with pytest.raises(NotFoundError):
        storage.require_path("paper-missing")


# --- service ---------------------------------------------------------------


def test_service_analyses_a_local_file(service, paper_pdf):
    record = service.analyze_file(paper_pdf)
    assert record.status is AnalysisStatus.COMPLETED
    assert record.report_path and record.report_markdown


def test_service_persists_the_report_to_disk(service, paper_pdf):
    from pathlib import Path

    record = service.analyze_file(paper_pdf)
    assert Path(record.report_path).read_text(encoding="utf-8") == record.report_markdown


def test_service_emits_progress_events_to_a_sink(service, paper_pdf):
    seen = []
    service.analyze_file(paper_pdf, event_sink=seen.append)
    assert seen
    assert {"ingestion", "report"} <= {e.stage for e in seen}


def test_service_updates_document_metadata_after_ingestion(service, paper_pdf):
    record = service.analyze_file(paper_pdf)
    metadata = service.get_document(record.document_id)
    assert metadata.page_count == 6
    assert metadata.sections


def test_unknown_analysis_raises(service):
    with pytest.raises(NotFoundError):
        service.get_analysis("analysis-missing")


def test_report_before_completion_raises(service, paper_pdf):
    metadata = service.register_local_file(paper_pdf)
    record = service.create_analysis(metadata.document_id)
    with pytest.raises(NotFoundError):
        service.get_report(record.analysis_id)


# --- CLI -------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "x.pdf"],
        ["ingest", "x.pdf"],
        ["serve"],
        ["report", "analysis-1"],
    ],
)
def test_parser_exposes_every_command(argv):
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]
    assert callable(args.func)


def test_cli_analyze_writes_a_report(tmp_path, paper_pdf, capsys):
    output = tmp_path / "report.md"
    code = main(
        [
            "analyze",
            str(paper_pdf),
            "--output",
            str(output),
            "--data-dir",
            str(tmp_path / "data"),
            "--quiet",
        ]
    )
    assert code == EXIT_OK
    report = output.read_text(encoding="utf-8")
    assert "# PaperPilot Report" in report
    assert "[p. " in report


def test_cli_analyze_can_emit_structured_json(tmp_path, paper_pdf):
    target = tmp_path / "result.json"
    code = main(
        [
            "analyze",
            str(paper_pdf),
            "--output",
            str(tmp_path / "r.md"),
            "--json-out",
            str(target),
            "--data-dir",
            str(tmp_path / "data"),
            "--quiet",
        ]
    )
    assert code == EXIT_OK
    payload = json.loads(target.read_text())
    assert payload["status"] == "completed"
    assert payload["findings"]


def test_cli_returns_a_nonzero_code_for_a_missing_file(tmp_path, capsys):
    code = main(["analyze", str(tmp_path / "nope.pdf"), "--quiet"])
    assert code == EXIT_BAD_INPUT
    assert "not found" in capsys.readouterr().err


def test_cli_returns_a_nonzero_code_for_a_non_pdf(tmp_path, capsys):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    code = main(["analyze", str(bad), "--data-dir", str(tmp_path / "d"), "--quiet"])
    assert code == EXIT_BAD_INPUT
    assert "unsupported_file" in capsys.readouterr().err


def test_cli_ingest_prints_structure(tmp_path, paper_pdf, capsys):
    code = main(["ingest", str(paper_pdf), "--data-dir", str(tmp_path / "d"), "--quiet"])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "pages       : 6" in out
    assert "4.2 Main Results" in out


def test_cli_ingest_json_format(tmp_path, paper_pdf, capsys):
    code = main(
        ["ingest", str(paper_pdf), "--format", "json", "--data-dir", str(tmp_path / "d"), "-q"]
    )
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["page_count"] == 6
    assert payload["chunks"]


def test_cli_progress_printer_renders_stages(capsys):
    import sys

    from paperpilot.cli import ProgressPrinter
    from paperpilot.domain.models import ProgressEvent

    printer = ProgressPrinter(stream=sys.stdout)
    printer(ProgressEvent(stage="ingestion", status="completed", message="done"))
    assert "ingestion" in capsys.readouterr().out


def test_cli_warns_when_the_mock_provider_is_active(tmp_path, paper_pdf, capsys):
    main(
        [
            "analyze",
            str(paper_pdf),
            "--output",
            str(tmp_path / "r.md"),
            "--data-dir",
            str(tmp_path / "d"),
        ]
    )
    assert "mock provider" in capsys.readouterr().err
