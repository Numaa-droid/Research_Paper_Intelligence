"""Deterministic structure detection: sections, tables, figures, equations, refs.

All of this is rule-based on purpose. Classifying a block as a table caption or
working out which section a paragraph sits in is a parsing problem with a right
answer, so it must not be delegated to a model that can hallucinate one.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from ..domain.enums import ElementType
from .layout import LAYOUT_KINDS
from .pdf_parser import TextBlock

# --- Section headings ------------------------------------------------------

#: "3 Method", "4.2 Main Results", "A.1 Proofs"
_NUMBERED_HEADING = re.compile(
    r"^\s*((?:\d+|[A-Z])(?:\.\d+){0,3})\.?\s+([A-Z][^\n]{2,80})\s*$"
)

#: A heading number set on its own line, e.g. the "3.2" of "3.2\nAttention".
_BARE_SECTION_NUMBER = re.compile(r"^(?:\d+|[A-Z])(?:\.\d+){0,3}\.?$")

_CANONICAL_SECTIONS = (
    "abstract", "introduction", "background", "related work", "preliminaries",
    "method", "methods", "methodology", "approach", "model", "architecture",
    "experiments", "experimental setup", "experimental results", "results",
    "evaluation", "ablation", "ablations", "ablation study", "analysis",
    "discussion", "limitations", "conclusion", "conclusions", "future work",
    "acknowledgments", "acknowledgements", "references", "bibliography",
    "appendix", "supplementary material", "broader impact", "ethics statement",
)

#: Canonical names that never double as a table header or a label in a figure,
#: so a short line is enough. "Model", "Results" or "Method" alone on a line is
#: just as often a column header, and needs typographic emphasis to count.
_UNAMBIGUOUS_SECTIONS = frozenset({
    "abstract", "references", "bibliography", "acknowledgments", "acknowledgements",
    "appendix", "supplementary material", "broader impact", "ethics statement",
})

_UNNUMBERED_HEADING = re.compile(
    r"^\s*(" + "|".join(re.escape(s) for s in _CANONICAL_SECTIONS) + r")\s*:?\s*$",
    re.IGNORECASE,
)

# --- Captions --------------------------------------------------------------

# A caption must carry an explicit separator ("Table 1:" / "Table 1."), otherwise
# an ordinary sentence such as "Table 1 reports the main comparison" would be
# mistaken for the table itself.
_CAPTION_SEP = r"\s*[.:\u2013\u2014\-]\s+"
_TABLE_CAPTION = re.compile(
    r"^\s*(table)\s+([IVXLC]+|\d+[a-z]?)" + _CAPTION_SEP, re.IGNORECASE
)
_FIGURE_CAPTION = re.compile(
    r"^\s*(figure|fig\.)\s+([IVXLC]+|\d+[a-z]?)" + _CAPTION_SEP, re.IGNORECASE
)
_ALGORITHM_CAPTION = re.compile(
    r"^\s*(algorithm)\s+(\d+[a-z]?)" + _CAPTION_SEP, re.IGNORECASE
)

# --- Equations -------------------------------------------------------------

_EQUATION_NUMBER = re.compile(r"\(\s*\d+[a-z]?\s*\)\s*$")
#: A line that is nothing but an equation number, e.g. "(2)" at the right margin.
EQUATION_NUMBER_LINE = re.compile(r"^\s*\(\s*(\d+[a-z]?)\s*\)\s*$")
#: "·", "×" and "±" are deliberately absent: result tables are full of them
#: ("2.3 · 10^19", "71.2 ± 0.4"), so on their own they do not make text math.
_MATH_SYMBOLS = set("=+−-*/^_∑∫∏√≤≥≈≠∈∀∃∇∂λμσθαβγδεπφψωΣΔΩ|⟨⟩‖∥⊤≡∝")
_NUMERIC_LINE = re.compile(r"^[\d\s.,·×^%±+\-−]+$")
_LATEX_HINT = re.compile(r"\\(frac|sum|int|prod|sqrt|mathbb|mathcal|alpha|beta|theta|sigma|log|exp)")
_ARGMIN = re.compile(r"\b(arg\s?max|arg\s?min|softmax|log\s?p|E_?\[|\bs\.t\.)", re.IGNORECASE)

# --- References ------------------------------------------------------------

_REFERENCE_ENTRY = re.compile(
    r"^\s*(\[\d{1,3}\]|\d{1,3}\.)\s+\S|^\s*[A-Z][A-Za-z'\-]+,\s+[A-Z]\.", re.MULTILINE
)


@dataclass(slots=True)
class DetectedHeading:
    page: int
    number: str | None
    title: str
    raw: str
    level: int
    block_no: int
    #: True when the heading is numbered or matches a canonical section name.
    #: Front matter (title block, author list) is heading-*shaped* but not a
    #: section, and is only distinguishable this way.
    is_canonical: bool = False
    #: True for appendix-style letter numbering ("A.1 Proofs"). An author line
    #: ("A. Researcher, B. Scientist") has the same shape, so a letter-numbered
    #: heading is only trusted once a real section has already been seen.
    is_alpha_numbered: bool = False

    @property
    def label(self) -> str:
        """Canonical section label used in citations, e.g. ``4.2 Main Results``."""
        return f"{self.number} {self.title}".strip() if self.number else self.title


def looks_like_heading(block: TextBlock, *, body_font_size: float) -> DetectedHeading | None:
    """Classify a block as a section heading, or return ``None``.

    Combines a textual pattern with a typographic signal: real headings are short
    and are usually set in a larger or bolder face than body text.
    """
    text = block.text.strip()
    if not text or len(text) > 120 or text.count("\n") > 1 or block.rotated:
        return None
    # "Fig. 1. Authors on Scopus" is short, set apart and mostly capitalised, so
    # it satisfies every typographic test for a heading. It names a figure.
    # Captions are recognised first, and never become sections.
    if extract_element_id(text)[0] is not ElementType.UNKNOWN:
        return None

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    first_line = lines[0]
    # PDFs often lay the number and the title out as separate spans, so the
    # block reads "3.2\nAttention"; the heading is the two lines together.
    if len(lines) == 2 and _BARE_SECTION_NUMBER.match(lines[0]):
        first_line = f"{lines[0]} {lines[1]}"
    elif block.is_heading:
        # Layout analysis already established that every line of this block
        # belongs to one heading, so a heading that wrapped is read whole
        # rather than truncated at the first line break.
        first_line = " ".join(lines)
    bigger = body_font_size > 0 and block.font_size >= body_font_size * 1.08
    # ``is_heading`` is set by layout analysis when the block was cut out of a
    # paragraph that had swallowed its heading; the heading then inherits the
    # paragraph's typography and has no emphasis of its own to test.
    emphasised = bigger or block.is_bold or block.is_heading

    match = _NUMBERED_HEADING.match(first_line)
    if match:
        number, title = match.group(1), match.group(2).strip()
        # Guard against citation lines like "1. Smith, J. Deep learning."
        if title.endswith("."):
            title = title[:-1]
        if len(title.split()) <= 12:
            level = number.count(".") + 1
            return DetectedHeading(
                block.page,
                number,
                title,
                first_line,
                level,
                block.block_no,
                is_canonical=True,
                is_alpha_numbered=number[:1].isalpha(),
            )

    plain = _UNNUMBERED_HEADING.match(first_line)
    if plain and (emphasised or plain.group(1).lower() in _UNAMBIGUOUS_SECTIONS):
        return DetectedHeading(
            block.page,
            None,
            plain.group(1).title(),
            first_line,
            1,
            block.block_no,
            is_canonical=True,
        )

    # Bold, short, title-case, starts with a capital, no terminal punctuation ->
    # probably a heading. The capital rules out identifiers such as "arXiv:1706…".
    if (
        emphasised
        and first_line[:1].isupper()
        and len(first_line.split()) <= 8
        and not first_line.endswith((".", ",", ";"))
    ):
        words = [w for w in first_line.split() if w[:1].isalpha()]
        if words and sum(1 for w in words if w[0].isupper()) >= max(1, len(words) - 1):
            return DetectedHeading(block.page, None, first_line, first_line, 1, block.block_no)
    return None


def body_font_size(blocks: list[TextBlock]) -> float:
    """Median font size of substantial blocks: the body-text baseline."""
    sizes = [b.font_size for b in blocks if b.font_size > 0 and len(b.text) > 120]
    if not sizes:
        sizes = [b.font_size for b in blocks if b.font_size > 0]
    return round(statistics.median(sizes), 2) if sizes else 0.0


def looks_like_equation(text: str) -> bool:
    """Heuristically decide whether a block is display mathematics."""
    stripped = text.strip()
    if not stripped or len(stripped) > 600:
        return False
    if _LATEX_HINT.search(stripped):
        return True

    letters = sum(1 for c in stripped if c.isalpha())
    # "^" is also the marker extraction puts before raised text ("10^20"), which
    # appears in prose and tables alike, so it does not count as mathematics.
    symbols = sum(1 for c in stripped if c in _MATH_SYMBOLS and c != "^")
    digits = sum(1 for c in stripped if c.isdigit())
    total = max(len(stripped), 1)

    lines = stripped.split("\n")
    if "=" not in stripped and sum(1 for ln in lines if _NUMERIC_LINE.match(ln.strip())) >= 2:
        return False  # a run of bare numbers is a table body, not an equation

    numbered = bool(_EQUATION_NUMBER.search(lines[-1]))
    symbol_ratio = symbols / total
    word_count = len(stripped.split())

    if "=" not in stripped and symbols < 2:
        return False
    # Display equations are symbol-dense and word-sparse.
    if symbol_ratio >= 0.06 and word_count <= 40 and letters / total < 0.7:
        return True
    if numbered and symbols >= 2 and word_count <= 40:
        return True
    if _ARGMIN.search(stripped) and "=" in stripped and word_count <= 30:
        return True
    return digits > 0 and symbol_ratio > 0.12 and word_count <= 20


#: Symbols that mark text as mathematics on their own. Hyphens, slashes and plus
#: signs are left out: prose is full of them ("term-frequency-based", "and/or").
_STRONG_MATH = set("=∑∫∏√≤≥≈≠∈∉∀∃∇∂‖∥·×÷±∞→⇒⊂⊆∪∩∼≡∝λμσθαβγδεπφψωΣΔΩ")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
#: Fragments of one display equation share a visual row; this much vertical slack
#: (points) admits subscripts, superscripts and the halves of a fraction.
_EQUATION_ROW_SLACK = 8.0


def _starts_equation(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped.split()) > 25 or stripped.count("\n") > 6:
        return False
    if any(EQUATION_NUMBER_LINE.match(line) for line in stripped.split("\n")):
        return True
    return any(c in _STRONG_MATH for c in stripped)


def _continues_equation(text: str) -> bool:
    """A piece that only makes sense beside math: "X" (a mis-mapped sum), "+ k1"."""
    stripped = text.strip()
    return _starts_equation(stripped) or (
        len(stripped.split()) <= 6 and not stripped.endswith((".", ":", ";", ","))
    )


def _on_row(run: list[TextBlock], block: TextBlock) -> bool:
    top = min(b.bbox[1] for b in run) - _EQUATION_ROW_SLACK
    bottom = max(b.bbox[3] for b in run) + _EQUATION_ROW_SLACK
    return block.page == run[0].page and block.bbox[1] <= bottom and block.bbox[3] >= top


def _merge_equation(run: list[TextBlock], *, displayed: bool = False) -> TextBlock | None:
    """One block for a run of fragments, or ``None`` if the run is not an equation.

    ``displayed`` marks a run set apart from the running text: named functions
    ("MultiHead(Q, K, V) = Concat(...)") are too wordy for the symbol-density
    test, but a centred line with an ``=`` is still display mathematics.
    """
    lines: list[str] = []
    number: str | None = None
    for block in run:
        for line in _CONTROL_CHARS.sub("", block.text).split("\n"):
            match = EQUATION_NUMBER_LINE.match(line)
            if match:
                number = match.group(1)
            elif line.strip():
                lines.append(line.strip())
    text = "\n".join(lines)
    if "=" not in text or not (number or displayed or looks_like_equation(text)):
        return None
    if number:
        text = f"{text}\n({number})"
    return TextBlock(
        page=run[0].page,
        text=text,
        bbox=(
            min(b.bbox[0] for b in run),
            min(b.bbox[1] for b in run),
            max(b.bbox[2] for b in run),
            max(b.bbox[3] for b in run),
        ),
        font_size=max(b.font_size for b in run),
        is_bold=False,
        block_no=run[0].block_no,
        source=run[0].source,
        element_hint=ElementType.EQUATION,
        confidence=run[0].confidence,
    )


def assemble_display_equations(blocks: list[TextBlock]) -> list[TextBlock]:
    """Rejoin display equations that PDF extraction split into side-by-side blocks.

    A typeset equation such as ``BM25(q, d) = Σ IDF(t) · … (2)`` often comes out
    of PyMuPDF as a row of small blocks — the left-hand side, the summation sign,
    its subscript, numerator, denominator and the equation number — each of which
    is meaningless alone. Consecutive blocks that sit on the same visual row, the
    first of which is mathematical, are merged back into one equation block, as
    long as the merged text reads as an equation. Anything else passes through
    unchanged, so prose and tables are unaffected.
    """
    out: list[TextBlock] = []
    run: list[TextBlock] = []

    def close() -> None:
        merged = None
        if len(run) >= 2 or (run and _is_displayed(run, blocks)):
            merged = _merge_equation(run, displayed=_is_displayed(run, blocks))
        out.extend([merged] if merged else run)
        run.clear()

    for block in blocks:
        if block.element_hint is not None or block.rotated:
            close()
            out.append(block)
            continue
        if run and _on_row(run, block) and _continues_equation(block.text):
            run.append(block)
            continue
        close()
        if _starts_equation(block.text):
            run.append(block)
        else:
            out.append(block)
    close()
    return out


#: A line indented this far from both edges of its block (as a fraction of the
#: block width) is set as display material, not running text.
_DISPLAY_INDENT = 0.1
_DISPLAY_MAX_WORDS = 40
#: Vertical overlap (points) below which two lines are on different rows.
_ROW_OVERLAP = 2.0


def _is_displayed(run: list[TextBlock], blocks: list[TextBlock]) -> bool:
    """True when a run sits indented from both edges of its text column.

    The column is taken from the running-text blocks that share the run's
    horizontal extent, so a two-column layout is measured per column.
    """
    x0 = min(b.bbox[0] for b in run)
    x1 = max(b.bbox[2] for b in run)
    column = [
        b
        for b in blocks
        if b.page == run[0].page
        and len(b.text) > 120
        and b.bbox[0] < x1
        and b.bbox[2] > x0
    ]
    if not column:
        return False
    left = statistics.median(b.bbox[0] for b in column)
    right = statistics.median(b.bbox[2] for b in column)
    margin = (right - left) * _DISPLAY_INDENT
    return x0 - left >= margin and right - x1 >= margin


def _sub_block(
    block: TextBlock, lines: list, *, hint: ElementType | None, heading: bool = False
) -> TextBlock:
    return TextBlock(
        page=block.page,
        text="\n".join(text for text, _ in lines),
        bbox=(
            min(b[0] for _, b in lines),
            min(b[1] for _, b in lines),
            max(b[2] for _, b in lines),
            max(b[3] for _, b in lines),
        ),
        font_size=block.font_size,
        is_bold=block.is_bold,
        block_no=block.block_no,
        source=block.source,
        element_hint=hint,
        confidence=block.confidence,
        lines=list(lines),
        is_heading=heading,
    )


#: A line reaching this close to the widest line in its block is a full line of
#: justified text, so whatever follows it is a continuation, not a new element.
_FULL_LINE_SLACK = 6.0
#: How many wrapped lines a heading may run to beyond its first.
_HEADING_WRAP_LINES = 2
_HEADING_MAX_WORDS = 12


def _heading_text(lines: list) -> tuple[str, bool] | None:
    """Return ``(heading text, is numbered)`` for these lines read as one."""
    joined = " ".join(text.strip() for text, _ in lines).strip()
    if not joined or len(joined) > 120:
        return None
    match = _NUMBERED_HEADING.match(joined)
    # Digits only: "S. O'Connor, Open artificial intelligence platforms" is a
    # reference entry with exactly the shape of a letter-numbered heading, and
    # inside a paragraph block there is no typography left to tell them apart.
    if match and not match.group(1)[:1].isalpha():
        if len(match.group(2).split()) <= _HEADING_MAX_WORDS:
            return joined, True
    if _UNNUMBERED_HEADING.match(joined):
        return joined, False
    return None


def _heading_run(
    lines: list, index: int, full: list[bool], at_left: list[bool]
) -> list | None:
    """The lines from ``index`` that together form one heading, or ``None``."""
    if not (index == 0 or at_left[index]):
        return None
    first = _heading_text(lines[index : index + 1])
    if first is None:
        return None
    _, numbered = first
    if not numbered and not (index == 0 or not full[index - 1]):
        return None

    run = lines[index : index + 1]
    # A heading that reaches the column edge has wrapped; the line it wrapped
    # onto is short. Body text under a heading fills the column instead, which
    # is what keeps the paragraph from being swallowed into the title.
    while (
        len(run) <= _HEADING_WRAP_LINES
        and full[index + len(run) - 1]
        and index + len(run) < len(lines)
        and not full[index + len(run)]
        and _heading_text(lines[index : index + len(run) + 1])
    ):
        run = lines[index : index + len(run) + 1]
    return run


def split_embedded_headings(blocks: list[TextBlock]) -> list[TextBlock]:
    """Cut section headings out of the paragraph blocks that swallowed them.

    Many journal templates emit a heading and the body text beneath it as one
    layout block, so the heading owns none of the emphasis a heading normally
    carries and is far too long to pass any "short line" test. Left alone, the
    document ends up with no sections at all, and the only heading-shaped blocks
    left on the page — running heads, chart titles, table column headers — are
    promoted in their place.

    A heading is recognised here by layout facts, none of them specific to a
    publisher: the line matches a heading pattern and it starts at the left edge
    of its column. A numbered heading ("3.1 ChatGPT and Education environment")
    carries enough evidence in the pattern alone — prose almost never opens a
    line with a digit followed by a capitalised word. An *unnumbered* heading is
    an ordinary word such as "Method", so it additionally has to begin a
    paragraph: the line above it must stop short of the column edge, which
    justified body text does only at the end of a paragraph.
    """
    out: list[TextBlock] = []
    for block in blocks:
        lines = block.lines
        if block.element_hint is not None or block.rotated or len(lines) < 2:
            out.append(block)
            continue

        # The median, not the maximum: a block that mixes a centred figure
        # caption with a narrower text column has one line far wider than the
        # rest, and measuring against it would make every body line look short.
        right = statistics.median(b[2] for _, b in lines)
        left = min(b[0] for _, b in lines)
        full = [b[2] >= right - _FULL_LINE_SLACK for _, b in lines]
        # Wide word spacing in justified text makes some PDFs report a single
        # line as several, each holding one word. Those fragments sit mid-column;
        # a heading always starts at the column's left edge.
        at_left = [b[0] <= left + _FULL_LINE_SLACK for _, b in lines]

        pieces: list[tuple[list, bool]] = []
        index = 0
        while index < len(lines):
            # The block's own first line is always a paragraph start, and may be
            # indented by a drop cap or a first-line indent, so the left-edge
            # test only applies to lines inside the block.
            heading = _heading_run(lines, index, full, at_left)
            if heading:
                pieces.append((heading, True))
                index += len(heading)
                continue
            if pieces and not pieces[-1][1]:
                pieces[-1][0].append(lines[index])
            else:
                pieces.append(([lines[index]], False))
            index += 1

        if len(pieces) < 2:
            out.append(block)
            continue
        out.extend(
            _sub_block(block, run, hint=None, heading=is_heading) for run, is_heading in pieces
        )
    return out


def split_display_equations(blocks: list[TextBlock]) -> list[TextBlock]:
    """Separate display equations that the PDF placed inside a paragraph block.

    Running text starts at the block's left edge; a display equation is indented
    from both edges. A run of such lines that reads as mathematics (it has an
    ``=``) becomes its own equation block, and the prose around it stays prose.
    Uses layout only, so it applies to any paper.
    """
    out: list[TextBlock] = []
    for block in blocks:
        lines = block.lines
        if block.element_hint is not None or block.rotated or len(lines) < 3:
            out.append(block)
            continue
        x0, _, x1, _ = block.bbox
        margin = (x1 - x0) * _DISPLAY_INDENT
        display = [b[0] - x0 >= margin and x1 - b[2] >= margin for _, b in lines]
        if all(display) or not any(display):
            out.append(block)
            continue

        pieces: list[tuple[list, bool]] = []
        for line, is_display in zip(lines, display, strict=True):
            if pieces and pieces[-1][1] == is_display:
                pieces[-1][0].append(line)
            else:
                pieces.append(([line], is_display))

        # Inline sub- and superscripts also come out as indented "lines", but they
        # share a row with the prose around them; display math does not.
        prose = [b for (_, b), shown in zip(lines, display, strict=True) if not shown]

        def own_rows(run: list, prose: list = prose) -> bool:
            return not any(
                p[1] < r[3] - _ROW_OVERLAP and r[1] < p[3] - _ROW_OVERLAP
                for _, r in run
                for p in prose
            )

        split: list[TextBlock] = []
        pending: list = []
        for run, is_display in pieces:
            text = "\n".join(t for t, _ in run)
            if (
                is_display
                and "=" in text
                and len(text.split()) <= _DISPLAY_MAX_WORDS
                and own_rows(run)
            ):
                if pending:
                    split.append(_sub_block(block, pending, hint=None))
                    pending = []
                split.append(_sub_block(block, run, hint=ElementType.EQUATION))
            else:
                pending.extend(run)
        if pending:
            split.append(_sub_block(block, pending, hint=None))
        out.extend(split if len(split) > 1 else [block])
    return out


def looks_like_table(text: str) -> bool:
    """Detect a tabular block by numeric density and column-like spacing."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    numeric_lines = 0
    for line in lines:
        cells = re.split(r"\s{2,}|\t|\s\|\s", line.strip())
        numbers = sum(1 for c in cells if re.fullmatch(r"[-+]?\d[\d,.]*%?", c.strip()))
        if len(cells) >= 3 and numbers >= 2:
            numeric_lines += 1
    return numeric_lines >= 2


def looks_like_table_body(text: str) -> bool:
    """Short cell-like lines without sentence punctuation, e.g. a table of words.

    Only consulted directly after a table caption, where it separates the table
    body from the prose that follows the table.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2 or any(len(ln) > 90 for ln in lines):
        return False
    sentence_ends = sum(1 for ln in lines if ln.endswith((".", ";", ":")))
    lengths = sorted(len(ln) for ln in lines)
    return sentence_ends <= len(lines) // 4 and lengths[len(lengths) // 2] <= 50


def extract_element_id(text: str) -> tuple[ElementType, str | None]:
    """Return the element type and stable identifier implied by a caption."""
    first_line = text.strip().split("\n")[0]
    if m := _TABLE_CAPTION.match(first_line):
        return ElementType.TABLE, f"table-{m.group(2).lower()}"
    if m := _FIGURE_CAPTION.match(first_line):
        return ElementType.FIGURE, f"figure-{m.group(2).lower()}"
    if m := _ALGORITHM_CAPTION.match(first_line):
        return ElementType.FIGURE, f"algorithm-{m.group(2).lower()}"
    return ElementType.UNKNOWN, None


def is_reference_section(section: str | None) -> bool:
    if not section:
        return False
    lowered = section.lower()
    return "reference" in lowered or "bibliograph" in lowered


def classify_block(
    block: TextBlock,
    *,
    section: str | None,
    equation_counter: int,
) -> tuple[ElementType, str | None]:
    """Assign an element type and identifier to one layout block.

    Order matters: an explicit caption wins over every statistical signal, and
    the references section wins over table-like numeric density (bibliographies
    are full of year and page numbers).
    """
    text = block.text

    element_type, element_id = extract_element_id(text)
    if element_type is not ElementType.UNKNOWN:
        return element_type, element_id

    if is_reference_section(section) and _REFERENCE_ENTRY.search(text):
        return ElementType.REFERENCE, None

    if block.element_hint is ElementType.EQUATION or looks_like_equation(text):
        number = _EQUATION_NUMBER.search(text.strip().split("\n")[-1])
        if number:
            digits = re.sub(r"[^\d a-z]", "", number.group(0)).strip()
            return ElementType.EQUATION, f"equation-{digits}"
        return ElementType.EQUATION, f"equation-p{block.page}-{equation_counter}"

    if looks_like_table(text):
        return ElementType.TABLE, None

    return ElementType.PARAGRAPH, None


# --- layout-driven structure -------------------------------------------------

#: A line counts as belonging to a region when this much of its area is inside.
_REGION_OVERLAP = 0.5
#: Kinds whose text is page furniture rather than document content.
_FURNITURE_KINDS = frozenset({"header", "footer"})


def _line_region(
    bbox: tuple[float, float, float, float], regions: list
) -> str | None:
    """The kind of the region a line sits in, or ``None`` if it sits in none.

    A line belongs to a region when most of it lies inside, *or* when its centre
    does. The second test matters: a detector draws its box tightly around the
    glyphs, while PyMuPDF reports a line box with the full leading above and
    below, so a one-digit folio can sit squarely inside a footer and still fall
    just under an area threshold.

    Where regions overlap, the one covering the most of the line wins; ties are
    broken by the region's reading order, so the answer never depends on the
    order the model happened to report them in.
    """
    x0, y0, x1, y1 = bbox
    area = max((x1 - x0) * (y1 - y0), 1e-6)
    centre = ((x0 + x1) / 2, (y0 + y1) / 2)
    best: tuple[float, int, str] | None = None
    for region in regions:
        rx0, ry0, rx1, ry1 = region.bbox
        width = min(x1, rx1) - max(x0, rx0)
        height = min(y1, ry1) - max(y0, ry0)
        if width <= 0 or height <= 0:
            continue
        covered = (width * height) / area
        encloses_centre = rx0 <= centre[0] <= rx1 and ry0 <= centre[1] <= ry1
        if covered < _REGION_OVERLAP and not encloses_centre:
            continue
        candidate = (-covered, region.order, region.kind)
        if best is None or candidate < best:
            best = candidate
    return best[2] if best else None


def apply_page_layout(blocks: list[TextBlock], layout) -> list[TextBlock]:
    """Re-cut a page's blocks along the regions a layout model found.

    PyMuPDF reports the blocks the PDF's content stream declares, which on many
    templates fuse a heading, its paragraph and a caption into one object. The
    layout model was asked what the *page* contains, so its regions are used
    here as the authority: each line is assigned to the region it sits in, runs
    of lines sharing a region become their own block, lines inside a running
    header or footer are dropped, and a run inside a title region is marked as a
    heading regardless of how it was set.

    Blocks the model has nothing to say about are returned untouched, so a page
    it failed on degrades to the geometric heuristics rather than losing text.
    """
    regions = [r for r in getattr(layout, "regions", []) if r.kind in LAYOUT_KINDS]
    if not regions:
        return blocks

    out: list[TextBlock] = []
    for block in blocks:
        if block.element_hint is not None or block.rotated or not block.lines:
            out.append(block)
            continue

        kinds = [_line_region(bbox, regions) for _, bbox in block.lines]
        if all(kind is None for kind in kinds):
            out.append(block)
            continue

        runs: list[tuple[list, str | None]] = []
        for line, kind in zip(block.lines, kinds, strict=True):
            if runs and runs[-1][1] == kind:
                runs[-1][0].append(line)
            else:
                runs.append(([line], kind))

        pieces = [
            _sub_block(block, run, hint=None, heading=kind == "title")
            for run, kind in runs
            if kind not in _FURNITURE_KINDS
        ]
        # Dropping every line of a block would silently delete content, so a
        # block that is entirely furniture is the only case where nothing is
        # emitted; anything else keeps at least one piece.
        out.extend(pieces)
    return out
