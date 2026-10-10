"""Render the Markdown report as a PDF.

No new dependency: markdown-it (already used to parse model answers) turns the
report into HTML, and PyMuPDF's ``Story`` lays that HTML out onto A4 pages.
MuPDF's built-in fallback fonts cover the symbols papers use (≥, ∥, Σ, Greek,
CJK), so equations and quotes print as written.

Report text includes quotes from the paper and sentences from a model, neither
of which is HTML. Everything is therefore escaped, and only the two pieces of
markup the synthesizer itself emits (``<sub>`` and ``&nbsp;``) are restored.
"""

from __future__ import annotations

import io
import re

import pymupdf
from markdown_it import MarkdownIt

_MD = MarkdownIt("commonmark", {"html": False}).enable("table")

_PAGE = pymupdf.paper_rect("a4")
_MARGIN_X = 54
_MARGIN_TOP = 54
_MARGIN_BOTTOM = 64
_BODY = _PAGE + (_MARGIN_X, _MARGIN_TOP, -_MARGIN_X, -_MARGIN_BOTTOM)

_CSS = """
body { font-family: sans-serif; font-size: 9.5pt; line-height: 1.4; color: #1f2937; }
h1 { font-size: 20pt; color: #0f172a; margin: 0 0 8pt 0; }
h2 { font-size: 13.5pt; color: #1e3a8a; margin: 16pt 0 5pt 0; padding-top: 4pt;
     border-top: 0.6pt solid #cbd5e1; }
h3 { font-size: 11pt; color: #0f172a; margin: 11pt 0 4pt 0; }
h4 { font-size: 10pt; color: #475569; margin: 9pt 0 3pt 0; }
p { margin: 0 0 6pt 0; }
ul, ol { margin: 0 0 6pt 0; padding-left: 14pt; }
li { margin: 0 0 2.5pt 0; }
blockquote { margin: 0 0 7pt 0; padding: 3pt 0 3pt 8pt; color: #334155;
             border-left: 2pt solid #60a5fa; background-color: #f1f5f9; }
code { font-family: monospace; font-size: 8.5pt; color: #0f172a; }
pre { font-family: monospace; font-size: 8.5pt; background-color: #f3f4f6;
      padding: 5pt; margin: 0 0 7pt 0; white-space: pre-wrap; }
table { border-collapse: collapse; margin: 2pt 0 8pt 0; }
th { background-color: #e2e8f0; color: #0f172a; font-weight: bold; }
th, td { border: 0.5pt solid #cbd5e1; padding: 2.5pt 4pt; text-align: left; vertical-align: top;
         font-size: 8.5pt; }
sub { font-size: 7.5pt; color: #64748b; }
hr { border-top: 0.6pt solid #cbd5e1; margin: 10pt 0; }
em { color: #475569; }
"""

#: Markup the synthesizer writes on purpose; escaped by markdown-it, restored here.
_ALLOWED_MARKUP = (
    (re.compile(r"&lt;(/?)sub&gt;"), r"<\1sub>"),
    (re.compile(r"&amp;nbsp;"), "&nbsp;"),
)


def report_html(markdown: str) -> str:
    """The report's Markdown as safe HTML for layout."""
    html = _MD.render(markdown or "")
    for pattern, replacement in _ALLOWED_MARKUP:
        html = pattern.sub(replacement, html)
    return html


def render_report_pdf(markdown: str, *, title: str | None = None, kind: str = "report") -> bytes:
    """Lay the report (or the brief, ``kind="brief"``) out on A4 pages, with a footer."""
    story = pymupdf.Story(html=report_html(markdown), user_css=_CSS)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    more = True
    while more:
        device = writer.begin_page(_PAGE)
        more, _ = story.place(_BODY)
        story.draw(device)
        writer.end_page()
    writer.close()

    document = pymupdf.open("pdf", buffer.getvalue())
    label = f"PaperMind {kind} · {title}" if title else f"PaperMind {kind}"
    if len(label) > 90:
        label = label[:89] + "…"
    total = document.page_count
    for number, page in enumerate(document, start=1):
        y = _PAGE.height - 34
        page.draw_line((_MARGIN_X, y - 10), (_PAGE.width - _MARGIN_X, y - 10), color=(0.8, 0.84, 0.88), width=0.5)
        page.insert_text((_MARGIN_X, y), label, fontsize=7.5, color=(0.39, 0.45, 0.55))
        counter = f"Page {number} of {total}"
        width = pymupdf.get_text_length(counter, fontsize=7.5)
        page.insert_text((_PAGE.width - _MARGIN_X - width, y), counter, fontsize=7.5, color=(0.39, 0.45, 0.55))
    document.set_metadata(
        {
            "title": title or f"PaperMind {kind}",
            "author": "PaperMind",
            "subject": "Citation-grounded research paper analysis",
            "creator": "PaperMind",
            "producer": "PaperMind (PyMuPDF)",
        }
    )
    data = document.tobytes(garbage=3, deflate=True)
    document.close()
    return data


def report_filename(analysis_id: str, title: str | None) -> str:
    """``MedCoPilot-an-end-to-end-RAG-framework-0593f8.pdf`` — readable and unique."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title or "").strip("-")[:60].strip("-")
    short = analysis_id.removeprefix("analysis-")[:6]
    return f"{slug or 'paperpilot-report'}-{short}.pdf"
