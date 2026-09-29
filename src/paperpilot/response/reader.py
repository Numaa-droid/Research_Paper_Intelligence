"""Read agent fields out of a :class:`ResponseDocument`.

Every lookup is by normalised heading, key or column header with explicit
aliases, so ``## Main results``, ``### 4. Main Results:`` and ``**Main results**``
all resolve to the same field. Nothing here guesses: a field that is not present
is returned empty, and the agent decides what an empty field means.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .parser import CITATION_RE, extract_citations, normalize_key, strip_citations
from .schema import (
    BlockquoteBlock,
    Citation,
    CodeBlock,
    ListBlock,
    ListItem,
    MathBlock,
    ParagraphBlock,
    ResponseDocument,
    Section,
    TableBlock,
    TableCell,
)

_TRUE = {"yes", "y", "true", "t", "1", "run", "enabled", "enable", "on", "inferred", "differs"}
_FALSE = {"no", "n", "false", "f", "0", "skip", "skipped", "disabled", "disable", "off", "none"}
_CONFIDENCE_WORDS = {
    "very high": 0.9, "high": 0.8, "medium": 0.5, "moderate": 0.5, "low": 0.3, "very low": 0.15,
}
_EMPTY_VALUES = {"", "n a", "na", "none", "null", "not stated", "not reported", "unknown", "nil"}


@dataclass(slots=True)
class QuoteRef:
    """A quoted passage and the sources it was attributed to."""

    text: str
    citations: list[Citation] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return list(dict.fromkeys(c.label for c in self.citations))


# --- scalar coercion ---------------------------------------------------------


def clean_text(text: str | None) -> str:
    """Plain field text: citation markers removed, placeholder values emptied."""
    value = strip_citations(text or "").strip().strip('"“”').strip()
    return "" if normalize_key(value) in _EMPTY_VALUES else value


def parse_bool(text: Any) -> bool | None:
    key = normalize_key(str(text or ""))
    if not key:
        return None
    first = key.split()[0]
    if key in _TRUE or first in _TRUE:
        return True
    if key in _FALSE or first in _FALSE:
        return False
    return None


def parse_score(text: Any) -> float | None:
    """``0.8``, ``80%``, ``8/10``, ``high`` -> a value in [0, 1]."""
    raw = strip_citations(str(text or "")).strip().lower()
    if not raw:
        return None
    if word := _CONFIDENCE_WORDS.get(normalize_key(raw)):
        return word
    ratio = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", raw)
    if ratio and float(ratio.group(2)) > 0:
        return max(0.0, min(1.0, float(ratio.group(1)) / float(ratio.group(2))))
    number = re.match(r"^(\d+(?:\.\d+)?)\s*(%)?", raw)
    if not number:
        return None
    value = float(number.group(1))
    if number.group(2) or value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def parse_int(text: Any) -> int | None:
    match = re.search(r"-?\d+", str(text or ""))
    return int(match.group(0)) if match else None


def unquote(text: str) -> str:
    """Strip the quotation marks a model puts around a verbatim quote."""
    value = text.strip()
    for opener, closer in (('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’"), ("«", "»")):
        if len(value) >= 2 and value.startswith(opener) and value.endswith(closer):
            return value[len(opener) : -len(closer)].strip()
    return value


# --- traversal ---------------------------------------------------------------


def walk_sections(sections: Iterable[Section]) -> Iterator[Section]:
    for section in sections:
        yield section
        yield from walk_sections(section.subsections)


def section_blocks(section: Section, *, include_implicit: bool = True) -> list[Any]:
    """A section's own blocks, plus blocks under bold-only pseudo-headings.

    ``**Evidence:**`` inside a finding parses as an implicit subsection; its
    content still belongs to the finding.
    """
    blocks = list(section.blocks)
    if include_implicit:
        for sub in section.subsections:
            if sub.implicit:
                blocks.extend(section_blocks(sub))
    return blocks


def iter_blocks(blocks: Iterable[Any]) -> Iterator[Any]:
    """Depth-first over blocks, including blocks nested in lists and quotes."""
    for block in blocks:
        yield block
        if isinstance(block, ListBlock):
            for item in block.items:
                yield from iter_blocks(item.children)
        elif isinstance(block, BlockquoteBlock):
            continue  # a blockquote's children are its own text, already captured


class DocumentReader:
    def __init__(self, document: ResponseDocument) -> None:
        self.document = document

    # --- sections ------------------------------------------------------------

    def section(self, *aliases: str, within: Section | None = None) -> Section | None:
        """First section whose heading matches an alias (exact, then prefix)."""
        wanted = [normalize_key(a) for a in aliases]
        pool = list(walk_sections(within.subsections if within else self.document.sections))
        for section in pool:
            if section.key in wanted:
                return section
        for section in pool:
            if any(w and section.key.startswith(w + " ") for w in wanted):
                return section
        return None

    @staticmethod
    def subsections(section: Section | None) -> list[Section]:
        if section is None:
            return []
        return [s for s in section.subsections if not s.implicit] or list(section.subsections)

    # --- content -------------------------------------------------------------

    @staticmethod
    def paragraphs(section: Section | None) -> str:
        """The section's prose: paragraphs, and plain (non key-value) list items."""
        if section is None:
            return ""
        parts: list[str] = []
        for block in section_blocks(section):
            if isinstance(block, ParagraphBlock):
                parts.append(block.text)
            elif isinstance(block, MathBlock):
                parts.append(block.text)
        return clean_text("\n\n".join(p for p in parts if p.strip()))

    def text(self, section: Section | None) -> str:
        """Prose if there is any, otherwise the list items joined as lines."""
        prose = self.paragraphs(section)
        if prose or section is None:
            return prose
        return clean_text("\n".join(self.items(section)))

    @staticmethod
    def list_items(section: Section | None) -> list[ListItem]:
        if section is None:
            return []
        items: list[ListItem] = []
        for block in iter_blocks(section_blocks(section)):
            if isinstance(block, ListBlock):
                items.extend(block.items)
        return items

    def items(self, section: Section | None, *, limit: int = 60) -> list[str]:
        """List items as clean strings; a paragraph-only section yields its lines."""
        items = [clean_text(i.text) for i in self.list_items(section)]
        if not items and section is not None:
            items = [
                clean_text(line)
                for block in section_blocks(section)
                if isinstance(block, ParagraphBlock)
                for line in block.text.split("\n")
            ]
        return [i for i in items if i][:limit]

    def fields(self, section: Section | None) -> dict[str, ListItem]:
        """``- Key: value`` items keyed by normalised key (first wins)."""
        out: dict[str, ListItem] = {}
        for item in self.list_items(section):
            if item.key:
                out.setdefault(normalize_key(item.key), item)
        if section is not None:
            for block in section_blocks(section):
                if isinstance(block, ParagraphBlock):
                    for line in block.text.split("\n"):
                        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9 _/()-]{0,40}?)\s*:\s*(.+)$", line)
                        if match:
                            key = normalize_key(match.group(1))
                            out.setdefault(
                                key,
                                ListItem(
                                    index=0,
                                    text=line,
                                    markdown=line,
                                    key=match.group(1),
                                    value=match.group(2),
                                    citations=extract_citations(line),
                                ),
                            )
        return out

    @staticmethod
    def field(fields: dict[str, ListItem], *aliases: str) -> ListItem | None:
        for alias in aliases:
            item = fields.get(normalize_key(alias))
            if item is not None:
                return item
        return None

    def field_text(self, fields: dict[str, ListItem], *aliases: str) -> str:
        item = self.field(fields, *aliases)
        return clean_text(item.value) if item and item.value is not None else ""

    def key_value_map(self, section: Section | None, *, limit: int = 60) -> dict[str, str]:
        """A mapping written either as a two-column table or as ``- key: value``."""
        out: dict[str, str] = {}
        for table in self.tables(section):
            if table.column_count >= 2:
                for row in table.rows:
                    key, value = clean_text(row.cells[0].value), clean_text(row.cells[1].value)
                    if key:
                        out.setdefault(key, value)
        for item in self.list_items(section):
            if item.key and item.value is not None:
                out.setdefault(clean_text(item.key), clean_text(item.value))
        return dict(list(out.items())[:limit])

    @staticmethod
    def tables(section: Section | None) -> list[TableBlock]:
        if section is None:
            return []
        return [b for b in iter_blocks(section_blocks(section)) if isinstance(b, TableBlock)]

    @staticmethod
    def records(
        table: TableBlock, columns: dict[str, tuple[str, ...]]
    ) -> list[dict[str, TableCell | None]]:
        """Rows as ``{field: cell}`` using header aliases per field."""
        wanted = {
            name: [normalize_key(a) for a in (name, *aliases)] for name, aliases in columns.items()
        }
        positions: dict[str, int] = {}

        def assign(matches) -> None:
            for name in columns:
                if name in positions:
                    continue
                for column in table.columns:
                    if column.column_index not in positions.values() and matches(column.key, wanted[name]):
                        positions[name] = column.column_index
                        break

        # Exact headers for every field first, so a loose match cannot steal a
        # column that another field names exactly.
        assign(lambda key, names: key in names)
        assign(lambda key, names: any(n and (key.startswith(n + " ") or n in key.split()) for n in names))
        rows: list[dict[str, TableCell | None]] = []
        for row in table.rows:
            if all(cell.empty for cell in row.cells):
                continue
            rows.append(
                {
                    name: (row.cells[positions[name]] if name in positions else None)
                    for name in columns
                }
            )
        return rows

    @staticmethod
    def code(section: Section | None) -> CodeBlock | MathBlock | None:
        if section is None:
            return None
        for block in iter_blocks(section_blocks(section)):
            if isinstance(block, CodeBlock | MathBlock):
                return block
        return None

    def quotes(self, section: Section | None) -> list[QuoteRef]:
        """Verbatim quotes: blockquote paragraphs, and ``Quote:``/``Evidence:`` items."""
        if section is None:
            return []
        refs: list[QuoteRef] = []
        for block in iter_blocks(section_blocks(section)):
            if isinstance(block, BlockquoteBlock):
                refs.extend(quotes_from_text(block.text))
        fields = self.fields(section)
        for key in ("quote", "quotes", "evidence", "source quote", "supporting quote"):
            item = fields.get(key)
            if item is None:
                continue
            if item.value:
                refs.extend(quotes_from_text(item.value))
            for child in iter_blocks(item.children):
                if isinstance(child, ListBlock):
                    for sub in child.items:
                        refs.extend(quotes_from_text(sub.text))
        return [r for r in refs if clean_text(r.text)]  # drops "None." and "Not stated."


def quotes_from_text(text: str) -> list[QuoteRef]:
    """Split quoted text into passages, one per paragraph or per citation marker.

    Two attribution styles are understood:

    * leading — ``[S1] first passage [S2] second passage`` (the prompted style);
    * trailing — ``"first passage" [S1] "second passage" [S2]``.

    Adjacent markers (``[S1] [S2] passage``) attribute the passage to both.
    """
    refs: list[QuoteRef] = []
    for paragraph in re.split(r"\n\s*\n", text or ""):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        markers = list(CITATION_RE.finditer(paragraph))
        if not markers:
            refs.append(QuoteRef(text=unquote(paragraph)))
            continue

        segments = []  # (text, marker) pairs in the chosen attribution style
        trailing_style = markers[0].start() > 0 and not paragraph[markers[-1].end() :].strip(" .;,")
        if trailing_style:
            previous = 0
            for marker in markers:
                segments.append((paragraph[previous : marker.start()], marker))
                previous = marker.end()
        else:
            leading = paragraph[: markers[0].start()]
            if leading.strip():
                refs.append(QuoteRef(text=unquote(leading)))
            for position, marker in enumerate(markers):
                end = markers[position + 1].start() if position + 1 < len(markers) else len(paragraph)
                segments.append((paragraph[marker.end() : end], marker))

        pending: list[Citation] = []
        for body, marker in segments:
            citations = pending + extract_citations(marker.group(0))
            body = unquote(body.strip().strip(":-–—;, ").strip())
            if body:
                refs.append(QuoteRef(text=body, citations=citations))
                pending = []
            else:
                pending = citations
        if pending and refs:
            refs[-1].citations.extend(c for c in pending if c not in refs[-1].citations)
    return refs
