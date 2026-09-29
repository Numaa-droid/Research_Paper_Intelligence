"""Markdown -> :class:`ResponseDocument`, deterministically.

Parsing uses markdown-it (CommonMark + GFM tables) for block structure, and the
original source lines wherever rendering would change text a reader relies on:

* table rows are re-split from source, so a row with more cells than the header
  (usually an unescaped ``|`` inside a quote) keeps its text instead of losing it;
* blockquotes keep their source verbatim, so a quoted ``a_i * b_j`` is not
  turned into emphasis.

Every repair is recorded in ``ResponseDocument.warnings``. The parser never
raises on malformed Markdown: any text is a valid document, possibly an empty one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token

from .schema import (
    BlockquoteBlock,
    Citation,
    CodeBlock,
    DocumentStats,
    HeadingBlock,
    HtmlBlock,
    ListBlock,
    ListItem,
    MathBlock,
    ParagraphBlock,
    ResponseDocument,
    Section,
    SourceSpan,
    TableBlock,
    TableCell,
    TableColumn,
    TableRow,
    ThematicBreakBlock,
)

_MD = MarkdownIt("commonmark", {"html": False}).enable("table")

#: ``[S3]``, ``[S3, S4]``, ``[S3; s4]``, ``[source: S3]``, ``[Sources: S1, S2]``.
CITATION_RE = re.compile(
    r"\[\s*(?:sources?\s*:\s*)?(S\d{1,4}(?:\s*[,;]\s*S\d{1,4})*)\s*\]", re.IGNORECASE
)
_LABEL_RE = re.compile(r"S\d{1,4}", re.IGNORECASE)

_WRAPPING_FENCE_RE = re.compile(
    r"\A\s*(`{3,}|~{3,})[ \t]*(?:markdown|md)?[ \t]*\n(.*?)\n\1[ \t]*\Z", re.DOTALL | re.IGNORECASE
)
_KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _/().'&-]{0,59}?)\s*[:：]\s*(.*)\Z", re.DOTALL)
_TASK_RE = re.compile(r"^\[([ xX])\]\s+")
_NUMBER_CELL_RE = re.compile(
    r"^\s*([+\-−–]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?)\s*"
    r"(%|pp|x|×|ms|s|k|m|b|gb|mb)?\s*(?:(?:±|\+/-|\+-)\s*[\d.]+\s*%?)?\s*\Z",
    re.IGNORECASE,
)
_HEADING_NUMBER_RE = re.compile(r"^\s*(?:\(?[0-9ivxIVX]{1,5}[.)]|\d+(?:\.\d+)+\.?)\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_ESCAPED_PIPE = "\x00PIPE\x00"


# --- public helpers ----------------------------------------------------------


def normalize_key(text: str) -> str:
    """Lookup form of a heading, key or column header.

    ``"## 2. Main Results:"`` and ``"main results"`` normalise identically.
    """
    text = CITATION_RE.sub(" ", text or "")
    text = text.replace("**", " ").replace("__", " ").replace("`", " ")
    text = _HEADING_NUMBER_RE.sub("", text)
    return _NON_ALNUM_RE.sub(" ", text.lower()).strip()


def extract_citations(text: str) -> list[Citation]:
    citations: list[Citation] = []
    for match in CITATION_RE.finditer(text or ""):
        for label in _LABEL_RE.findall(match.group(1)):
            citations.append(Citation(label=label.upper(), raw=match.group(0)))
    return citations


#: Adjacent markers, "[S6], [S20]" or "[S1] and [S2]", removed as one unit so no
#: orphaned separator is left behind ("heads , .").
_CITATION_GROUP_RE = re.compile(
    rf"{CITATION_RE.pattern}(?:\s*(?:,|;|&|\band\b)?\s*{CITATION_RE.pattern})*", re.IGNORECASE
)


def strip_citations(text: str) -> str:
    """Remove citation markers and tidy the whitespace they leave behind."""
    stripped = _CITATION_GROUP_RE.sub("", text or "")
    stripped = re.sub(r"[ \t]+([.,;:!?)])", r"\1", stripped)
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


def parse_markdown(markdown: str) -> ResponseDocument:
    """Parse a model's Markdown answer into a :class:`ResponseDocument`."""
    return _Parser(markdown or "").parse()


# --- implementation ----------------------------------------------------------


@dataclass
class _Node:
    token: Token
    children: list[_Node] = field(default_factory=list)


def _tree(tokens: list[Token]) -> list[_Node]:
    root = _Node(Token("root", "", 1))
    stack = [root]
    for token in tokens:
        if token.nesting == 1:
            node = _Node(token)
            stack[-1].children.append(node)
            stack.append(node)
        elif token.nesting == -1:
            if len(stack) > 1:
                stack.pop()
        else:
            stack[-1].children.append(_Node(token))
    return root.children


def _inline_plain(tokens: list[Token] | None) -> str:
    parts: list[str] = []
    for child in tokens or []:
        kind = child.type
        if kind in ("text", "code_inline", "html_inline"):
            parts.append(child.content)
        elif kind == "softbreak":
            parts.append("\n")
        elif kind == "hardbreak":
            parts.append("\n")
        elif kind == "image":
            parts.append(child.content or "")
    return "".join(parts)


def _strong_only(tokens: list[Token] | None) -> str | None:
    """Text of an inline run that is exactly one bold span, else ``None``."""
    meaningful = [t for t in tokens or [] if not (t.type == "text" and not t.content.strip())]
    if meaningful and meaningful[-1].type == "text" and meaningful[-1].content.strip() in (":", "："):
        meaningful = meaningful[:-1]
    if len(meaningful) < 3:
        return None
    if meaningful[0].type != "strong_open" or meaningful[-1].type != "strong_close":
        return None
    inner = meaningful[1:-1]
    if any(t.type in ("strong_open", "strong_close", "softbreak", "hardbreak") for t in inner):
        return None
    return _inline_plain(inner).strip() or None


def _span(token: Token) -> SourceSpan | None:
    if not token.map:
        return None
    start, end = token.map
    return SourceSpan(line_start=start + 1, line_end=max(start + 1, end))


def _split_row(line: str) -> list[str]:
    """Split one GFM table row on unescaped pipes."""
    body = line.strip().replace("\\|", _ESCAPED_PIPE)
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    return [cell.replace(_ESCAPED_PIPE, "\\|").strip() for cell in body.split("|")]


def _rejoin_pipe_notation(cells: list[str], width: int) -> list[str]:
    """Undo splits caused by unescaped ``|x|`` (absolute value, norm, cardinality).

    ``| |d| | length |`` splits into ``["", "d", "", "length"]``; an empty cell,
    a short cell and an empty cell in a row that has too many cells can only be
    that notation, so they are joined back into ``|d|``.
    """
    cells = list(cells)
    i = 0
    while len(cells) > width and i + 2 < len(cells):
        if not cells[i] and cells[i + 1] and len(cells[i + 1]) <= 20 and not cells[i + 2]:
            cells[i : i + 3] = [f"|{cells[i + 1]}|"]
        i += 1
    return cells


def _parse_number(value: str) -> tuple[float | None, str | None]:
    match = _NUMBER_CELL_RE.match(value)
    if not match or not re.search(r"\d", match.group(1) or ""):
        return None, None
    raw = match.group(1).replace(" ", "").replace(",", "").replace("−", "-").replace("–", "-")
    try:
        number = float(raw)
    except ValueError:
        return None, None
    return number, (match.group(2) or None)


class _Parser:
    def __init__(self, markdown: str) -> None:
        self.warnings: list[str] = []
        source = markdown.replace("\r\n", "\n").replace("\r", "\n")
        wrapped = _WRAPPING_FENCE_RE.match(source)
        if wrapped:
            source = wrapped.group(2)
            self.warnings.append("unwrapped a code fence around the whole response")
        self.source = source
        self.lines = source.split("\n")
        self.stats = DocumentStats()
        self._block_ids = 0

    # --- driver --------------------------------------------------------------

    def parse(self) -> ResponseDocument:
        tokens = _MD.parse(self.source)
        blocks = self._blocks(_tree(tokens))
        preamble, sections = self._sectionize(blocks)
        return ResponseDocument(
            preamble=preamble, sections=sections, stats=self.stats, warnings=self.warnings
        )

    def _next_id(self, prefix: str) -> str:
        self._block_ids += 1
        return f"{prefix}-{self._block_ids:04d}"

    def _source_lines(self, token: Token) -> list[str]:
        if not token.map:
            return []
        start, end = token.map
        return self.lines[start:end]

    def _cites(self, text: str) -> list[Citation]:
        citations = extract_citations(text)
        self.stats.citations += len(citations)
        return citations

    # --- blocks --------------------------------------------------------------

    def _blocks(self, nodes: list[_Node]) -> list[Any]:
        out: list[Any] = []
        for node in nodes:
            block = self._block(node)
            if block is not None:
                out.append(block)
        return out

    def _block(self, node: _Node) -> Any:
        token = node.token
        kind = token.type

        if kind == "heading_open":
            inline = node.children[0].token if node.children else None
            markdown = inline.content if inline else ""
            text = _inline_plain(inline.children if inline else None).strip()
            return HeadingBlock(
                id=self._next_id("h"),
                level=int(token.tag[1:]),
                text=text,
                markdown=markdown,
                citations=self._cites(text),
                span=_span(token),
            )

        if kind == "paragraph_open":
            return self._paragraph(node)

        if kind in ("bullet_list_open", "ordered_list_open"):
            return self._list(node)

        if kind == "table_open":
            return self._table(node)

        if kind == "blockquote_open":
            self.stats.blockquotes += 1
            raw_lines = [re.sub(r"^\s{0,3}>\s?", "", line) for line in self._source_lines(token)]
            raw = "\n".join(raw_lines).strip()
            return BlockquoteBlock(
                id=self._next_id("q"),
                text=raw,
                markdown=raw,
                citations=self._cites(raw),
                children=self._blocks(node.children),
                span=_span(token),
            )

        if kind in ("fence", "code_block"):
            info = (token.info or "").strip().split()
            language = info[0].lower() if info else None
            text = token.content[:-1] if token.content.endswith("\n") else token.content
            self.stats.code_blocks += 1
            return CodeBlock(id=self._next_id("c"), language=language, text=text, span=_span(token))

        if kind == "hr":
            return ThematicBreakBlock(id=self._next_id("hr"), span=_span(token))

        if kind == "html_block":
            return HtmlBlock(id=self._next_id("html"), text=token.content, span=_span(token))

        return None

    def _paragraph(self, node: _Node) -> Any:
        token = node.token
        inline = node.children[0].token if node.children else None
        markdown = inline.content if inline else ""

        stripped = markdown.strip()
        for opener, closer, delimiter in (("$$", "$$", "$$"), ("\\[", "\\]", "\\[")):
            if len(stripped) > len(opener) + len(closer) and stripped.startswith(opener) and (
                stripped.endswith(closer)
            ):
                self.stats.math_blocks += 1
                return MathBlock(
                    id=self._next_id("m"),
                    text=stripped[len(opener) : -len(closer)].strip(),
                    delimiter=delimiter,  # type: ignore[arg-type]
                    span=_span(token),
                )

        bold = _strong_only(inline.children if inline else None)
        if bold is not None and len(bold) <= 120:
            return HeadingBlock(
                id=self._next_id("h"),
                level=6,
                text=bold.rstrip(":： "),
                markdown=markdown,
                citations=self._cites(bold),
                implicit=True,
                span=_span(token),
            )

        text = _inline_plain(inline.children if inline else None).strip()
        self.stats.paragraphs += 1
        return ParagraphBlock(
            id=self._next_id("p"),
            text=text,
            markdown=markdown,
            citations=self._cites(text),
            span=_span(token),
        )

    def _list(self, node: _Node) -> ListBlock:
        token = node.token
        ordered = token.type == "ordered_list_open"
        start = token.attrs.get("start") if ordered else None
        block = ListBlock(
            id=self._next_id("l"),
            ordered=ordered,
            start=int(start) if start is not None else (1 if ordered else None),
            span=_span(token),
        )
        self.stats.lists += 1
        for index, item_node in enumerate(n for n in node.children if n.token.type == "list_item_open"):
            block.items.append(self._list_item(item_node, index, ordered))
        return block

    def _list_item(self, node: _Node, index: int, ordered: bool) -> ListItem:
        self.stats.list_items += 1
        children = list(node.children)
        markdown = text = ""
        if children and children[0].token.type == "paragraph_open":
            first = children.pop(0)
            inline = first.children[0].token if first.children else None
            markdown = inline.content if inline else ""
            text = _inline_plain(inline.children if inline else None).strip()

        checked: bool | None = None
        task = _TASK_RE.match(text)
        if task:
            checked = task.group(1).lower() == "x"
            text = text[task.end() :]

        key = value = None
        pair = _KEY_VALUE_RE.match(text)
        if pair and not re.match(r"^\s*https?$", pair.group(1), re.IGNORECASE):
            key, value = pair.group(1).strip(), pair.group(2).strip()

        number = None
        if ordered and node.token.info and node.token.info.isdigit():
            number = int(node.token.info)

        return ListItem(
            index=index,
            number=number,
            checked=checked,
            key=key,
            value=value,
            text=text,
            markdown=markdown,
            citations=self._cites(text),
            children=self._blocks(children),
            span=_span(node.token),
        )

    def _table(self, node: _Node) -> TableBlock:
        token = node.token
        self.stats.tables += 1
        lines = self._source_lines(token)
        header_cells = _split_row(lines[0]) if lines else []
        alignments = [self._alignment(c) for c in _split_row(lines[1])] if len(lines) > 1 else []
        table_id = self._next_id("t")

        columns: list[TableColumn] = []
        for column_index, cell in enumerate(header_cells):
            text = _inline_plain(_MD.parseInline(cell)[0].children).strip()
            columns.append(
                TableColumn(
                    column_index=column_index,
                    header=text,
                    markdown=cell,
                    key=normalize_key(text),
                    alignment=alignments[column_index] if column_index < len(alignments) else None,
                )
            )

        width = len(columns)
        table = TableBlock(id=table_id, column_count=width, row_count=0, columns=columns, span=_span(token))
        start_line = (token.map[0] if token.map else 0) + 2
        for offset, line in enumerate(lines[2:]):
            if not line.strip():
                continue
            raw_cells = _split_row(line)
            line_number = start_line + offset + 1
            if width and len(raw_cells) > width:
                raw_cells = _rejoin_pipe_notation(raw_cells, width)
            if width and len(raw_cells) > width:
                self.warnings.append(
                    f"table {table_id} line {line_number}: {len(raw_cells)} cells for "
                    f"{width} columns; extra cells merged into '{columns[-1].header}'"
                )
                raw_cells = raw_cells[: width - 1] + [" | ".join(raw_cells[width - 1 :])]
            elif len(raw_cells) < width:
                raw_cells = raw_cells + [""] * (width - len(raw_cells))
            table.rows.append(self._row(raw_cells, len(table.rows), columns, line_number))

        table.row_count = len(table.rows)
        return table

    def _row(
        self, raw_cells: list[str], row_index: int, columns: list[TableColumn], line_number: int
    ) -> TableRow:
        cells: list[TableCell] = []
        row_header: str | None = None
        for column_index, cell_markdown in enumerate(raw_cells):
            children = _MD.parseInline(cell_markdown)[0].children
            text = _inline_plain(children).strip()
            value = strip_citations(text)
            if column_index == 0:
                row_header = value or None
            number, unit = _parse_number(value)
            strong = _strong_only(children) is not None
            self.stats.table_cells += 1
            cells.append(
                TableCell(
                    row_index=row_index,
                    column_index=column_index,
                    column_header=columns[column_index].header if column_index < len(columns) else None,
                    row_header=row_header,
                    text=text,
                    markdown=cell_markdown,
                    value=value,
                    number=number,
                    unit=unit,
                    strong=strong,
                    empty=not value,
                    citations=self._cites(text),
                )
            )
        for cell in cells:
            cell.row_header = row_header
        return TableRow(
            row_index=row_index,
            row_header=row_header,
            cells=cells,
            span=SourceSpan(line_start=line_number, line_end=line_number),
        )

    @staticmethod
    def _alignment(delimiter: str) -> str | None:
        cell = delimiter.strip()
        if cell.startswith(":") and cell.endswith(":"):
            return "center"
        if cell.endswith(":"):
            return "right"
        if cell.startswith(":"):
            return "left"
        return None

    # --- sections ------------------------------------------------------------

    @staticmethod
    def _implicit_level(block: HeadingBlock, stack: list[Section], last_implicit: Section | None) -> int:
        """Level of a bold-only pseudo-heading, which has none of its own.

        * ``**Evidence:**`` (ends in a colon) labels content of the current section,
          so it nests one level below it.
        * ``**Findings**`` continues the answer's pseudo-heading style: it is a
          sibling of the previous pseudo-heading while that one is still open, and
          otherwise sits one level below the nearest real heading (level 2 at top).
        """
        if block.markdown.strip().rstrip("*_ \t").endswith((":", "：")):
            return min(6, stack[-1].level + 1) if stack else 2
        if last_implicit is not None and any(s is last_implicit for s in stack):
            return last_implicit.level
        explicit = next((s for s in reversed(stack) if not s.implicit), None)
        return min(6, explicit.level + 1) if explicit else 2

    def _sectionize(self, blocks: list[Any]) -> tuple[list[Any], list[Section]]:
        preamble: list[Any] = []
        roots: list[Section] = []
        stack: list[Section] = []
        counters: dict[tuple[str, ...], int] = {}
        last_implicit: Section | None = None

        for block in blocks:
            if not isinstance(block, HeadingBlock):
                (stack[-1].blocks if stack else preamble).append(block)
                continue
            is_label = False
            if block.implicit:
                block.level = self._implicit_level(block, stack, last_implicit)
                is_label = block.markdown.strip().rstrip("*_ \t").endswith((":", "："))
            while stack and stack[-1].level >= block.level:
                stack.pop()
            parent_id = tuple(stack[-1].id.split("-")[1:]) if stack else ()
            counters[parent_id] = counters.get(parent_id, 0) + 1
            section = Section(
                id="-".join(("s", *parent_id, str(counters[parent_id]))),
                heading=block.text,
                key=normalize_key(block.text),
                level=block.level,
                path=[*(s.heading for s in stack), block.text],
                implicit=block.implicit,
                span=block.span,
            )
            self.stats.sections += 1
            (stack[-1].subsections if stack else roots).append(section)
            stack.append(section)
            if block.implicit and not is_label:
                last_implicit = section
        return preamble, roots
