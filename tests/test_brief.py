"""The two-page brief read before the full report."""

from __future__ import annotations

import re

import pymupdf
import pytest
from fastapi.testclient import TestClient

from paperpilot.api.app import create_app
from paperpilot.api.tasks import InlineTaskRunner
from paperpilot.reporting.brief import TARGET_WORDS, clip
from paperpilot.reporting.pdf import render_report_pdf

ACCOUNT = {"name": "Ada", "email": "ada@example.com", "password": "correct horse"}
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


@pytest.fixture
def record(service, paper_pdf):
    return service.analyze_file(paper_pdf)


def test_every_analysis_gets_a_brief_far_shorter_than_the_report(record):
    brief, report = record.brief_markdown, record.report_markdown
    assert brief
    assert len(brief.split()) <= TARGET_WORDS
    assert len(brief.split()) * 3 < len(report.split())
    for heading in ("## In short", "## The problem", "## What it found", "## How far to trust it"):
        assert heading in brief


def test_the_brief_fits_on_two_pages(record):
    pdf = pymupdf.open("pdf", render_report_pdf(record.brief_markdown, title="brief"))
    assert pdf.page_count <= 2


def test_the_brief_cites_only_what_the_full_report_cites(record):
    # ``[p. N]`` is the legend explaining the notation, not a citation.
    citations = set(re.findall(r"\[p\. [^\]]+\]", record.brief_markdown)) - {"[p. N]"}
    assert citations
    assert citations <= set(re.findall(r"\[p\. [^\]]+\]", record.report_markdown))


def test_the_brief_never_says_the_same_sentence_twice(record):
    prose = [
        line
        for line in record.brief_markdown.splitlines()
        if line and not line.startswith(("#", "|", ">", "---", "_"))
    ]
    sentences = [
        re.sub(r"\[p\.[^\]]*\]|\*", "", s).strip().lower()
        for line in prose
        for s in _SENTENCES.split(re.sub(r"^\s*(?:-|\d+\.)\s*", "", line))
    ]
    sentences = [s for s in sentences if len(s) > 40]
    assert len(sentences) == len(set(sentences))


def test_clip_keeps_whole_sentences_within_the_budget():
    text = "First point here. Second point here. Third point here."
    assert clip(text, sentences=2) == "First point here. Second point here."
    long = clip("word " * 400, sentences=1, max_chars=50)
    assert len(long) <= 50 and long.endswith("…")
    assert clip(None, sentences=2) == ""


def test_the_api_serves_the_brief_and_both_documents_as_pdf(service, paper_pdf):
    app = create_app(service.settings, service=service, task_runner=InlineTaskRunner())
    with TestClient(app) as client:
        assert client.post("/auth/signup", json=ACCOUNT).status_code == 201
        upload = client.post(
            "/papers", files={"file": ("paper.pdf", paper_pdf.read_bytes(), "application/pdf")}
        )
        started = client.post(f"/papers/{upload.json()['document_id']}/analyze").json()
        analysis_id = started["analysis_id"]
        assert started["brief_url"] == f"/analyses/{analysis_id}/brief"

        brief = client.get(f"/analyses/{analysis_id}/brief")
        assert brief.status_code == 200
        assert brief.headers["content-type"].startswith("text/markdown")
        assert "## In short" in brief.text

        for url in (f"/analyses/{analysis_id}/brief?format=pdf", f"/analyses/{analysis_id}/report?format=pdf"):
            pdf = client.get(url)
            assert pdf.status_code == 200
            assert pdf.headers["content-type"] == "application/pdf"
            assert pdf.content.startswith(b"%PDF")
            assert "attachment" in pdf.headers["content-disposition"]


def test_an_analysis_from_before_briefs_still_serves_its_report(service, paper_pdf):
    record = service.analyze_file(paper_pdf)
    record.brief_markdown = None
    service.analyses.save(record)

    app = create_app(service.settings, service=service, task_runner=InlineTaskRunner())
    with TestClient(app) as client:
        assert client.post("/auth/signup", json=ACCOUNT).status_code == 201  # inherits the paper
        assert client.get(f"/analyses/{record.analysis_id}").json()["brief_url"] is None
        assert client.get(f"/analyses/{record.analysis_id}/brief").status_code == 404
        assert client.get(f"/analyses/{record.analysis_id}/report").status_code == 200
