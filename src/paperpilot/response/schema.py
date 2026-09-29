"""The response document schema.

Models no longer write JSON. They answer in Markdown, and
:func:`paperpilot.response.parser.parse_markdown` turns that Markdown into a
:class:`ResponseDocument` — the same input always yields the same document.
Nothing in the document is produced by a model except the text inside it.

The schema is deliberately paper-agnostic: it describes Markdown structure
(sections, paragraphs, lists, tables, quotes, code, math), not any one agent's
fields. Agents read their fields out of it by section heading and table column.

Conventions worth knowing:

* ``row_index`` and ``column_index`` are 0-based. ``row_index`` counts body rows
  only; the header row is described by ``TableBlock.columns``.
* A GFM table has no row-header syntax, so the first column is the row header:
  ``row_header`` on every cell of a row is that row's first cell text.
* ``text`` is plain text with Markdown syntax removed; ``markdown`` is the exact
  source. Quotes and code keep their source verbatim, so evidence is never
  altered by Markdown rendering (``a_i * b_j`` stays ``a_i * b_j``).
* Citations are source labels such as ``[S3]`` that point at the retrieved paper
  chunks listed in the prompt.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Bump the minor version for additive changes and the major for breaking ones.
SCHEMA_VERSION = "1.0"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceSpan(_Model):
    """Lines of the Markdown source a node came from (1-based, inclusive)."""

    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)


class Citation(_Model):
    """A reference to one retrieved source, e.g. ``[S3]``."""

    label: str = Field(description="Normalised source label, e.g. 'S3'.")
    raw: str = Field(description="The marker exactly as written, e.g. '[S3, S4]'.")


class _TextNode(_Model):
    text: str = Field(description="Plain text, Markdown syntax removed, citation markers kept.")
    markdown: str = Field(description="Source Markdown, verbatim.")
    citations: list[Citation] = Field(default_factory=list)


class HeadingBlock(_TextNode):
    type: Literal["heading"] = "heading"
    id: str
    level: int = Field(ge=1, le=6)
    implicit: bool = Field(
        default=False,
        description="True when the heading was written as a bold-only paragraph (**Title**).",
    )
    span: SourceSpan | None = None


class ParagraphBlock(_TextNode):
    type: Literal["paragraph"] = "paragraph"
    id: str
    span: SourceSpan | None = None


class ListItem(_TextNode):
    index: int = Field(ge=0, description="0-based position in its list.")
    number: int | None = Field(default=None, description="Printed number in an ordered list.")
    checked: bool | None = Field(default=None, description="Task-list state, if any.")
    key: str | None = Field(
        default=None, description="For 'Key: value' items, the key as written (bold removed)."
    )
    value: str | None = Field(default=None, description="For 'Key: value' items, the value.")
    children: list[Block] = Field(default_factory=list)
    span: SourceSpan | None = None


class ListBlock(_Model):
    type: Literal["list"] = "list"
    id: str
    ordered: bool
    start: int | None = None
    items: list[ListItem] = Field(default_factory=list)
    span: SourceSpan | None = None


class TableColumn(_Model):
    column_index: int = Field(ge=0)
    header: str = Field(description="Header text, plain.")
    markdown: str
    key: str = Field(description="Normalised header used for lookups, e.g. 'metric name'.")
    alignment: Literal["left", "center", "right"] | None = None


class TableCell(_TextNode):
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    column_header: str | None = None
    row_header: str | None = None
    value: str = Field(description="Cell text with citation markers removed.")
    number: float | None = Field(
        default=None, description="Numeric value when the cell is a single number."
    )
    unit: str | None = Field(default=None, description="Trailing unit of a numeric cell, e.g. '%'.")
    strong: bool = Field(default=False, description="Whole cell was bold (often 'best result').")
    empty: bool = False


class TableRow(_Model):
    row_index: int = Field(ge=0)
    row_header: str | None = None
    cells: list[TableCell] = Field(default_factory=list)
    span: SourceSpan | None = None


class TableBlock(_Model):
    type: Literal["table"] = "table"
    id: str
    column_count: int = Field(ge=0)
    row_count: int = Field(ge=0)
    columns: list[TableColumn] = Field(default_factory=list)
    rows: list[TableRow] = Field(default_factory=list)
    span: SourceSpan | None = None


class BlockquoteBlock(_TextNode):
    type: Literal["blockquote"] = "blockquote"
    id: str
    children: list[Block] = Field(default_factory=list)
    span: SourceSpan | None = None


class CodeBlock(_Model):
    type: Literal["code"] = "code"
    id: str
    language: str | None = None
    text: str = Field(description="Code exactly as written.")
    span: SourceSpan | None = None


class MathBlock(_Model):
    type: Literal["math"] = "math"
    id: str
    text: str = Field(description="Math source exactly as written, delimiters removed.")
    delimiter: Literal["$$", "\\["]
    span: SourceSpan | None = None


class ThematicBreakBlock(_Model):
    type: Literal["thematic_break"] = "thematic_break"
    id: str
    span: SourceSpan | None = None


class HtmlBlock(_Model):
    type: Literal["html"] = "html"
    id: str
    text: str
    span: SourceSpan | None = None


Block = Annotated[
    HeadingBlock
    | ParagraphBlock
    | ListBlock
    | TableBlock
    | BlockquoteBlock
    | CodeBlock
    | MathBlock
    | ThematicBreakBlock
    | HtmlBlock,
    Field(discriminator="type"),
]


class Section(_Model):
    """A heading and everything under it up to the next heading of its level."""

    id: str
    heading: str
    key: str = Field(description="Normalised heading used for lookups; numbering removed.")
    level: int = Field(ge=1, le=6)
    path: list[str] = Field(description="Headings from the outermost section down to this one.")
    implicit: bool = False
    blocks: list[Block] = Field(
        default_factory=list, description="Content before the first subsection."
    )
    subsections: list[Section] = Field(default_factory=list)
    span: SourceSpan | None = None


class DocumentStats(_Model):
    sections: int = 0
    paragraphs: int = 0
    lists: int = 0
    list_items: int = 0
    tables: int = 0
    table_cells: int = 0
    blockquotes: int = 0
    code_blocks: int = 0
    math_blocks: int = 0
    citations: int = 0


class ResponseDocument(_Model):
    """A model's Markdown answer as structured, deterministic JSON."""

    schema_version: str = SCHEMA_VERSION
    preamble: list[Block] = Field(
        default_factory=list, description="Content before the first heading."
    )
    sections: list[Section] = Field(default_factory=list)
    stats: DocumentStats = Field(default_factory=DocumentStats)
    warnings: list[str] = Field(
        default_factory=list, description="Repairs applied while parsing, in source order."
    )


for _model in (ListItem, BlockquoteBlock, Section, ResponseDocument):
    _model.model_rebuild()


def response_document_json_schema() -> dict:
    """JSON Schema (draft 2020-12) for :class:`ResponseDocument`."""
    schema = ResponseDocument.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://paperpilot.dev/schemas/response-document/{SCHEMA_VERSION}.json"
    return schema
