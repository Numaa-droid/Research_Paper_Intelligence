"""Where the paper defines a symbol's shape or value, found without a model.

A math agent shown three passages marks ``W^Q_i`` "not defined in the paper"
when page 5 says ``W^Q_i ∈ ℝ^{d_model×d_k}``. Papers define symbols in a few
fixed forms, and each is readable straight from the text:

* membership:      ``W^Q_i ∈ ℝ^{d_model×d_k}``, ``x_i ∈ ℝ^d``
* value chains:    ``d_k = d_v = d_model/h = 64``, ``h = 8``
* dimension words: "queries and keys of dimension d_k", "the dimensionality of
  input and output is d_model"

The last form names things in words, so it is matched against the symbol's
meaning as the agent gave it ("Queries matrix" → queries). PDF extraction
scatters sub- and superscripts (``W Q i``), so symbols are compared with
spacing, ``_``, ``^`` and braces removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..domain.enums import ElementType
from ..retrieval.index import PaperIndex

#: A shape or meaning field that says the paper does not give one.
PLACEHOLDER = re.compile(
    r"^\W*(?:not\s+(?:\w+\s+){0,2}(?:defined|stated|specified|given|provided|reported)\b.*"
    r"|unknown|unspecified|undefined|n/?a|none|—|-)\W*$",
    re.IGNORECASE,
)
_MEMBERSHIP = re.compile(
    r"(?<![\w])(?P<sym>[^\W\d_][^\W_]{0,5}(?:\s[^\W_]{1,3}){0,2})\s*∈\s*(?:R|ℝ)\s*\^?\s*\{?"
    r"(?P<shape>[^\W_][\w]*(?:\s*[×x·]\s*[^\W_][\w]*)*)"
)
_LEADING_WORD = re.compile(r"^(?:and|or|with|where|are|is|the|of|let|for|all|each|any)\s+", re.IGNORECASE)
#: "d_k = d_v = d_model/h = 64": spaced equals signs, a number at the end.
_CHAIN = re.compile(
    r"(?P<lhs>(?:(?<![\w/^])[^\s=,;()]+\s+=\s+)+)"
    r"(?P<val>-?\d+(?:\.\d+)?(?:\^-?\d+)?)(?!\w|\.\d)"
)
_IDENT = re.compile(r"^[^\W\d][\w]{0,12}$")
_NOUNS = r"(?P<nouns>[a-z]+(?:\s*,\s*[a-z]+)*(?:\s*,?\s*and\s+[a-z]+)?)"
_OF_DIMENSION = re.compile(rf"{_NOUNS}\s+of\s+dimension\s+(?P<dim>[^\W\d][\w]*)")
_DIMENSIONALITY = re.compile(
    rf"dimensionality\s+of\s+(?:the\s+)?{_NOUNS}\s+is\s+(?P<dim>[^\W\d][\w]*)"
)
_WORD = re.compile(r"[a-z]+")
#: Words that open a meaning without naming the thing ("First bias vector"), and
#: nouns too generic to tie a symbol to a sentence ("vectors of dimension d").
_MEANING_SKIP = frozenset(
    "the a an first second third fourth final learned learnable trainable output-side "
    "vector vectors matrix matrices tensor tensors dimension dimensions number size scalar "
    "parameter parameters weight weights".split()
)


def symbol_key(symbol: str) -> str:
    """``WQ_i``, ``W^Q_i``, ``W Q i`` and ``W_{i}^{Q}`` order-insensitively aside, as one key."""
    text = re.sub(r"\\(?:mathrm|text|mathbf|boldsymbol)\b", "", symbol)
    return re.sub(r"[\s_^{}\\$]", "", text).lower()


def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


@dataclass(slots=True)
class Definition:
    text: str
    page: int
    others: list[tuple[str, int]]

    def describe(self) -> str:
        out = f"{self.text} (p. {self.page})"
        if self.others:
            out += "; also " + "; ".join(f"{t} (p. {p})" for t, p in self.others)
        return out


class SymbolDefinitions:
    """Symbol definitions stated in the paper's text, in document order."""

    def __init__(self, index: PaperIndex) -> None:
        self._by_symbol: dict[str, list[tuple[str, int]]] = {}
        self._by_noun: dict[str, list[tuple[str, int]]] = {}
        for chunk in index.chunks:
            if chunk.element_type is ElementType.TABLE:
                continue
            text = " ".join(chunk.text.split())
            for match in _MEMBERSHIP.finditer(text):
                symbol = _LEADING_WORD.sub("", match.group("sym"))
                shape = match.group("shape").replace(" ", "")
                self._add(self._by_symbol, symbol_key(symbol), f"∈ ℝ^{shape}", chunk.page)
            for match in _CHAIN.finditer(text):
                parts = [p.strip() for p in match.group("lhs").split("=") if p.strip()]
                chain = " = ".join([*parts, match.group("val")])
                for part in parts:
                    if _IDENT.match(part):
                        self._add(self._by_symbol, symbol_key(part), chain, chunk.page)
            for pattern in (_OF_DIMENSION, _DIMENSIONALITY):
                for match in pattern.finditer(text):
                    for noun in _WORD.findall(match.group("nouns")):
                        if noun != "and":
                            self._add(self._by_noun, _stem(noun), f"dimension {match.group('dim')}", chunk.page)

    @staticmethod
    def _add(table: dict[str, list[tuple[str, int]]], key: str, text: str, page: int) -> None:
        entries = table.setdefault(key, [])
        if all(existing != text for existing, _ in entries):
            entries.append((text, page))

    def lookup(self, symbol: str, meaning: str | None = None) -> Definition | None:
        """The paper's own definition of ``symbol``, or None if it gives none."""
        entries = self._by_symbol.get(symbol_key(symbol))
        if not entries and meaning:
            # Only the word the meaning is about: "Queries matrix" -> queries,
            # "First bias vector" -> bias, never the generic "vector".
            head = next((w for w in _WORD.findall(meaning.lower()) if w not in _MEANING_SKIP), None)
            entries = self._by_noun.get(_stem(head)) if head else None
        if not entries:
            return None
        (text, page), *others = entries
        return Definition(text, page, others[:2])
