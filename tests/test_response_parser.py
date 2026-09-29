"""Markdown model answers -> the deterministic response document."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperpilot.response import (
    ResponseDocument,
    parse_markdown,
    response_document_json_schema,
)
from paperpilot.response.reader import (
    DocumentReader,
    parse_bool,
    parse_score,
    quotes_from_text,
)

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "docs" / "schemas" / "response-document.schema.json"


def _only_table(markdown: str):
    document = parse_markdown(markdown)
    blocks = document.sections[0].blocks if document.sections else document.preamble
    return document, next(b for b in blocks if b.type == "table")


# --- tables ----------------------------------------------------------------


def test_table_cells_carry_coordinates_headers_and_row_headers():
    _, table = _only_table(
        "## Results\n\n"
        "| System | Dataset | F1 |\n"
        "| :--- | --- | ---: |\n"
        "| SRT-Base | NarrativeQA | 71.4 |\n"
        "| SRT-Large | NarrativeQA | 74.9 |\n"
    )

    assert (table.row_count, table.column_count) == (2, 3)
    assert [c.header for c in table.columns] == ["System", "Dataset", "F1"]
    assert [c.alignment for c in table.columns] == ["left", None, "right"]

    cell = table.rows[1].cells[2]
    assert (cell.row_index, cell.column_index) == (1, 2)
    assert cell.column_header == "F1"
    assert cell.row_header == "SRT-Large"
    assert table.rows[1].row_header == "SRT-Large"
    assert cell.value == "74.9" and cell.number == 74.9


def test_numeric_cells_keep_their_text_verbatim():
    _, table = _only_table(
        "| Metric | Value |\n| --- | --- |\n"
        "| acc | 85.2% |\n| delta | −0.5 ± 0.1 |\n| params | 1,234 |\n| name | ResNet-50 |\n"
        "| best | **91.3** |\n"
    )
    cells = {row.row_header: row.cells[1] for row in table.rows}

    assert (cells["acc"].value, cells["acc"].number, cells["acc"].unit) == ("85.2%", 85.2, "%")
    assert (cells["delta"].value, cells["delta"].number) == ("−0.5 ± 0.1", -0.5)
    assert cells["params"].number == 1234
    assert cells["name"].number is None
    assert cells["best"].strong and cells["best"].value == "91.3"


def test_an_escaped_pipe_stays_inside_its_cell():
    _, table = _only_table("| Quote |\n| --- |\n| a \\| b |\n")
    assert table.rows[0].cells[0].value == "a | b"


def test_extra_cells_from_an_unescaped_pipe_are_kept_not_dropped():
    document, table = _only_table(
        "| Metric | Source | Quote |\n| --- | --- | --- |\n"
        "| F1 | [S2] | Model | 74.9 | NarrativeQA |\n"
    )
    quote = table.rows[0].cells[2]

    assert table.column_count == 3 and len(table.rows[0].cells) == 3
    assert quote.value == "Model | 74.9 | NarrativeQA"
    assert any("merged" in w for w in document.warnings)


def test_short_rows_are_padded_with_empty_cells():
    _, table = _only_table("| A | B | C |\n| --- | --- | --- |\n| 1 |\n")
    assert [c.empty for c in table.rows[0].cells] == [False, True, True]


def test_citations_are_extracted_from_cells_and_removed_from_values():
    _, table = _only_table("| Metric | Source |\n| --- | --- |\n| F1 | [S2, S5] |\n")
    cell = table.rows[0].cells[1]
    assert [c.label for c in cell.citations] == ["S2", "S5"]
    assert cell.value == ""


# --- structure -------------------------------------------------------------


def test_headings_nest_into_sections_with_normalised_keys():
    document = parse_markdown(
        "Some preamble.\n\n"
        "## 1. Main Results:\ntext\n\n"
        "### Finding 1\n- Claim: x\n\n"
        "### Finding 2\n- Claim: y\n\n"
        "## Limitations\nnone given\n"
    )

    assert document.preamble[0].text == "Some preamble."
    assert [s.key for s in document.sections] == ["main results", "limitations"]
    findings = document.sections[0].subsections
    assert [s.heading for s in findings] == ["Finding 1", "Finding 2"]
    assert findings[1].path == ["1. Main Results:", "Finding 2"]
    assert findings[1].id == "s-1-2"


def test_a_bold_only_paragraph_is_read_as_a_heading():
    document = parse_markdown("**Research Problem**\n\nLong documents are hard.\n")
    section = DocumentReader(document).section("research problem")

    assert section is not None and section.implicit
    assert DocumentReader(document).text(section) == "Long documents are hard."


def _outline(markdown: str) -> dict[str, list[str]]:
    from paperpilot.response.reader import walk_sections

    document = parse_markdown(markdown)
    return {s.heading: [c.heading for c in s.subsections] for s in walk_sections(document.sections)}


def test_explicit_headings_nest_under_a_bold_pseudo_heading():
    outline = _outline("**Findings**\n\n#### Finding 1\n- Claim: a\n\n**Limitations**\n\nnone\n")
    assert outline["Findings"] == ["Finding 1"]
    assert outline["Limitations"] == []
    assert outline["Finding 1"] == []


def test_bold_pseudo_headings_under_a_real_heading_are_siblings():
    outline = _outline("## Findings\n\n**Finding 1**\n\ntext\n\n**Finding 2**\n\ntext\n\n## Methods\n\n**Step 1**\n")
    assert outline["Findings"] == ["Finding 1", "Finding 2"]
    assert outline["Methods"] == ["Step 1"]


def test_a_bold_label_ending_in_a_colon_stays_inside_its_section():
    document = parse_markdown(
        "## Findings\n### Finding 1\n- Claim: a\n\n**Evidence:**\n\n> [S1] quote\n\n### Finding 2\n"
    )
    reader = DocumentReader(document)
    finding = reader.section("finding 1")

    assert [s.heading for s in reader.subsections(reader.section("findings"))] == [
        "Finding 1", "Finding 2"
    ]
    assert [q.text for q in reader.quotes(finding)] == ["quote"]


def test_bold_key_value_items_are_split():
    document = parse_markdown("## Verdict\n- **Status:** supported\n- Confidence: 0.8\n")
    items = document.sections[0].blocks[0].items

    assert (items[0].key, items[0].value) == ("Status", "supported")
    assert (items[1].key, items[1].value) == ("Confidence", "0.8")


def test_blockquotes_keep_markdown_characters_verbatim():
    document = parse_markdown("## Evidence\n> [S1] the loss is a_i * b_j over __all__ pairs\n")
    quote = document.sections[0].blocks[0]
    assert quote.text == "[S1] the loss is a_i * b_j over __all__ pairs"


def test_code_and_display_math_are_verbatim():
    document = parse_markdown(
        "## Equation\n```latex\nL = \\sum_i \\log p(y_i | x_i)\n```\n\n$$\nE = m c^2\n$$\n"
    )
    code, math = document.sections[0].blocks
    assert (code.language, code.text) == ("latex", "L = \\sum_i \\log p(y_i | x_i)")
    assert (math.type, math.text) == ("math", "E = m c^2")


def test_a_code_fence_around_the_whole_answer_is_removed():
    document = parse_markdown("```markdown\n## Verdict\n- Status: supported\n```")
    assert [s.heading for s in document.sections] == ["Verdict"]
    assert document.warnings


def test_source_spans_point_at_the_markdown_lines():
    document = parse_markdown("## A\n\npara one\n\n## B\n- item\n")
    assert document.sections[0].blocks[0].span.line_start == 3
    assert document.sections[1].blocks[0].items[0].span.line_start == 6


# --- guarantees ------------------------------------------------------------


GARBAGE = [
    "",
    "   \n\n",
    '{"findings": [{"claim": "a"}, {"claim": "b',
    "| broken | table\n|---",
    "```python\nunclosed fence",
    "> ",
    "##",
    "\x00\x01 binary �",
    "- " * 500,
    "# " + "x" * 10_000,
]


@pytest.mark.parametrize("text", GARBAGE)
def test_any_text_parses_without_raising(text):
    document = parse_markdown(text)
    assert isinstance(document, ResponseDocument)


def test_parsing_is_deterministic():
    text = (
        "## Findings\n### Finding 1\n- Claim: x [S1]\n\n> [S1] quote\n\n"
        "| A | B |\n| - | - |\n| 1 | 2 | 3 |\n"
    )
    first = parse_markdown(text).model_dump_json()
    assert all(parse_markdown(text).model_dump_json() == first for _ in range(5))


def test_documents_round_trip_through_their_schema():
    document = parse_markdown(
        "intro\n\n## T\n| A |\n| - |\n| 1 |\n\n> [S1] q\n\n1. one\n   - nested\n\n---\n\n$$x$$\n"
    )
    dumped = json.loads(document.model_dump_json())
    assert ResponseDocument.model_validate(dumped) == document


def test_the_published_schema_file_is_in_sync():
    assert json.loads(SCHEMA_FILE.read_text(encoding="utf-8")) == response_document_json_schema(), (
        "regenerate with: paperpilot schema -o docs/schemas/response-document.schema.json"
    )


# --- reading fields ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[S1] first passage", [("first passage", ["S1"])]),
        ('"first passage" [S2]', [("first passage", ["S2"])]),
        ("[S1] one [S2] two", [("one", ["S1"]), ("two", ["S2"])]),
        ('"one" [S1] "two" [S2]', [("one", ["S1"]), ("two", ["S2"])]),
        ("[S1] [S3] shared passage", [("shared passage", ["S1", "S3"])]),
        ("[source: S4] labelled passage", [("labelled passage", ["S4"])]),
        ("uncited passage", [("uncited passage", [])]),
        ("[S1] para one\n\n[S2] para two", [("para one", ["S1"]), ("para two", ["S2"])]),
    ],
)
def test_quotes_are_attributed_to_their_sources(text, expected):
    assert [(q.text, q.labels) for q in quotes_from_text(text)] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0.8", 0.8), ("80%", 0.8), ("8/10", 0.8), ("high", 0.8), ("85", 0.85), ("n/a", None), ("", None)],
)
def test_scores_parse_from_common_spellings(text, expected):
    assert parse_score(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("yes", True), ("Yes, run it", True), ("no", False), ("skip", False), ("maybe", None)],
)
def test_booleans_parse_from_common_spellings(text, expected):
    assert parse_bool(text) is expected


def test_reader_matches_headings_by_alias_and_prefix():
    reader = DocumentReader(parse_markdown("## Main Results (with numbers)\n74.9 F1\n"))
    assert reader.text(reader.section("main results")) == "74.9 F1"


def test_placeholder_answers_read_as_empty():
    reader = DocumentReader(parse_markdown("## Limitations\nNot stated.\n\n## Ablations\n- None\n"))
    assert reader.text(reader.section("limitations")) == ""
    assert reader.items(reader.section("ablations")) == []


def test_table_records_match_columns_by_alias_without_stealing_exact_matches():
    reader = DocumentReader(
        parse_markdown("## R\n| Model | Dataset | Score |\n| - | - | - |\n| A | SQuAD | 90.1 |\n")
    )
    table = reader.tables(reader.section("r"))[0]
    record = reader.records(
        table, {"system": ("model",), "dataset": ("data",), "value": ("score",)}
    )[0]
    assert {k: v.value for k, v in record.items()} == {
        "system": "A", "dataset": "SQuAD", "value": "90.1"
    }


def test_unescaped_norm_bars_in_a_cell_are_rejoined():
    _, table = _only_table(
        "| Symbol | Meaning | Shape |\n| --- | --- | --- |\n| |d| | length of document d | scalar |\n"
    )
    assert [c.value for c in table.rows[0].cells] == ["|d|", "length of document d", "scalar"]
