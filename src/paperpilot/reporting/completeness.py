"""What the paper states that the report leaves out, found without a model.

Agents choose what to report, so a value the paper states plainly can be missed
even by an agent that read its passage: the training cost column of a results
table, the configuration row of the big model, a ``name = value`` setting in
the training section. Two structures carry most stated values, and both are
checkable against the finished report:

* **Table columns.** A results table whose rows are cited is "read" only in the
  columns its reported values came from. The other columns are shown as they
  are; when most of the table went unread, the whole table is.
* **Stated settings.** Every ``name = value`` in the paper's prose (``ϵ_ls =
  0.1``, ``warmup_steps = 4000``) that no line of the report states, with its
  name or a word from beside it, is listed in the paper's own words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..domain.enums import ElementType
from ..retrieval.index import PaperIndex
from ..validation.cooccurrence import _values
from ..validation.numeric import ParsedNumber, numbers_equal
from ..validation.symbols import symbol_key

_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")
_DIGIT = re.compile(r"\d")
_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_CHAIN = re.compile(
    r"(?P<lhs>(?:(?<![\w/^])[^\s=,;()]+\s+=\s+)+)"
    r"(?P<val>-?\d+(?:\.\d+)?(?:\^-?\d+)?)(?!\w|\.\d)"
)
_IDENT = re.compile(r"^[^\W\d][\w]{0,12}$")
_WORD = re.compile(r"[a-z]{3,}")
_GENERIC = frozenset(
    "use used using we our the and for with value values set rate this that these those were "
    "was are is has have had all each".split()
)
#: The report's own warnings quote the paper to correct something else; what
#: they quote is not something the report states.
_WARNING = re.compile(r"⚠️ _[^\n]*?\._")
_SKIP_SECTIONS = re.compile(r"^\W*(?:references|bibliography|acknowledg)", re.IGNORECASE)
#: Show the whole table when at most this share of its value columns was read.
_MOSTLY_UNREAD = 1 / 3


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _has(numbers: list[ParsedNumber], value: ParsedNumber) -> bool:
    return any(numbers_equal(value, n) for n in numbers)


def _stem(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 4 else word


@dataclass(slots=True)
class PartialTable:
    caption: str
    page: int
    header: list[str]
    rows: list[list[str]]
    read_columns: list[str]
    whole: bool

    def markdown(self, esc) -> list[str]:
        lines = ["| " + " | ".join(esc(h) or " " for h in self.header) + " |"]
        lines.append("| " + " | ".join("---" for _ in self.header) + " |")
        lines.extend("| " + " | ".join(esc(c) for c in row) + " |" for row in self.rows)
        return lines


def partly_read_tables(index: PaperIndex, reported: list[dict]) -> list[PartialTable]:
    """Cited tables with value columns that no reported result came from."""
    values_by_table: dict[str, list[ParsedNumber]] = {}
    for item in reported:
        numbers = _values(str(item.get("value") or ""))
        for evidence in item.get("evidence", []):
            if evidence.get("element_id"):
                values_by_table.setdefault(evidence["element_id"], []).extend(numbers)

    chunks_by_table: dict[str, list] = {}
    for chunk in index.chunks:
        if chunk.element_type is ElementType.TABLE and chunk.element_id in values_by_table:
            chunks_by_table.setdefault(chunk.element_id, []).append(chunk)

    out: list[PartialTable] = []
    for element_id, chunks in chunks_by_table.items():
        grid = [ln for c in chunks for ln in c.text.split("\n") if ln.lstrip().startswith("|")]
        grid = [ln for ln in grid if not _SEPARATOR.match(ln.strip())]
        if len(grid) < 2:
            continue
        header, rows = _cells(grid[0]), [_cells(ln) for ln in grid[1:]]
        width = len(header)
        rows = [(r + [""] * width)[:width] for r in rows]
        value_columns = [i for i in range(1, width) if any(_DIGIT.search(r[i]) for r in rows)]
        if not value_columns:
            continue
        reported_values = values_by_table[element_id]
        read = [
            i for i in value_columns
            if any(_has(reported_values, n) for r in rows for n in _values(r[i]))
        ]
        unread = [i for i in value_columns if i not in read]
        if not unread:
            continue
        whole = len(read) <= len(value_columns) * _MOSTLY_UNREAD
        keep = list(range(width)) if whole else [0, *unread]
        shown = [[r[i] for i in keep] for r in rows if any(r[i] for i in keep[1:])]
        caption_chunk = next((c for c in chunks if not c.text.lstrip().startswith("|")), chunks[0])
        out.append(
            PartialTable(
                caption=caption_chunk.text.split("\n")[0].strip(),
                page=chunks[0].page,
                header=[header[i] for i in keep],
                rows=shown,
                read_columns=[header[i] for i in read],
                whole=whole,
            )
        )
    return out


@dataclass(slots=True)
class StatedSetting:
    names: list[str]
    value: ParsedNumber
    chain: str
    sentence: str
    page: int
    context: set[str]


class StatedSettings:
    """Every ``name = value`` stated in the paper's prose, in document order."""

    def __init__(self, index: PaperIndex) -> None:
        self.settings: list[StatedSetting] = []
        for chunk in index.chunks:
            if chunk.element_type is ElementType.TABLE or _SKIP_SECTIONS.match(chunk.section or ""):
                continue
            for sentence in _SENTENCES.split(" ".join(chunk.text.split())):
                for match in _CHAIN.finditer(sentence):
                    parts = [p.strip() for p in match.group("lhs").split("=") if p.strip()]
                    names = [p for p in parts if _IDENT.match(p)]
                    value = _values(match.group("val"))
                    if not names or not value:
                        continue
                    before = [
                        w for w in _WORD.findall(sentence[: match.start()].lower())
                        if w not in _GENERIC
                    ]
                    self.settings.append(
                        StatedSetting(
                            names=names,
                            value=value[-1],
                            chain=" = ".join([*parts, match.group("val")]),
                            sentence=sentence.strip(),
                            page=chunk.page,
                            context={_stem(w) for w in before[-3:]},
                        )
                    )

    def unmentioned(self, report: str) -> list[StatedSetting]:
        """Settings no line of ``report`` states with its name or a word beside it."""
        report = _WARNING.sub(" ", report)
        lines = [
            (line, _values(line), re.sub(r"[\s_^{}\\$]", "", line.lower()).replace("ε", "ϵ"),
             {_stem(w) for w in _WORD.findall(line.lower())})
            for line in report.splitlines()
            if _DIGIT.search(line)
        ]
        missing: list[StatedSetting] = []
        seen: set[str] = set()
        for setting in self.settings:
            if setting.chain in seen:
                continue
            seen.add(setting.chain)
            if not any(self._states(setting, *line) for line in lines):
                missing.append(setting)
        return missing

    @staticmethod
    def _states(setting: StatedSetting, raw: str, numbers, squashed: str, words: set[str]) -> bool:
        if not _has(numbers, setting.value):
            return False
        for name in setting.names:
            key = symbol_key(name).replace("ε", "ϵ")
            if len(key) <= 2:
                if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", raw):
                    return True
            elif key in squashed:
                return True
        return bool(setting.context & words)
