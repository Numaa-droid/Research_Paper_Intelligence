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
from .pdf_parser import TextBlock

# --- Section headings ------------------------------------------------------

#: "3 Method", "4.2 Main Results", "A.1 Proofs"
_NUMBERED_HEADING = re.compile(
    r"^\s*((?:\d+|[A-Z])(?:\.\d+){0,3})\.?\s+([A-Z][^\n]{2,80})\s*$"
)

_CANONICAL_SECTIONS = (
    "abstract", "introduction", "background", "related work", "preliminaries",
    "method", "methods", "methodology", "approach", "model", "architecture",
    "experiments", "experimental setup", "experimental results", "results",
    "evaluation", "ablation", "ablations", "ablation study", "analysis",
    "discussion", "limitations", "conclusion", "conclusions", "future work",
    "acknowledgments", "acknowledgements", "references", "bibliography",
    "appendix", "supplementary material", "broader impact", "ethics statement",
)

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
_MATH_SYMBOLS = set("=+−-*/^_∑∫∏√≤≥≈≠∈∀∃∇∂λμσθαβγδεπφψωΣΔΩ|⟨⟩‖⊤")
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
    if not text or len(text) > 120 or text.count("\n") > 1:
        return None

    first_line = text.split("\n")[0].strip()
    bigger = body_font_size > 0 and block.font_size >= body_font_size * 1.08
    emphasised = bigger or block.is_bold

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
    if plain and (emphasised or len(first_line) <= 40):
        return DetectedHeading(
            block.page,
            None,
            plain.group(1).title(),
            first_line,
            1,
            block.block_no,
            is_canonical=True,
        )

    # Bold, short, title-case, no terminal punctuation -> probably a heading.
    if emphasised and len(first_line.split()) <= 8 and not first_line.endswith((".", ",", ";")):
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
    symbols = sum(1 for c in stripped if c in _MATH_SYMBOLS)
    digits = sum(1 for c in stripped if c.isdigit())
    total = max(len(stripped), 1)

    numbered = bool(_EQUATION_NUMBER.search(stripped.split("\n")[-1]))
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

    if looks_like_equation(text):
        number = _EQUATION_NUMBER.search(text.strip().split("\n")[-1])
        if number:
            digits = re.sub(r"[^\d a-z]", "", number.group(0)).strip()
            return ElementType.EQUATION, f"equation-{digits}"
        return ElementType.EQUATION, f"equation-p{block.page}-{equation_counter}"

    if looks_like_table(text):
        return ElementType.TABLE, None

    return ElementType.PARAGRAPH, None
