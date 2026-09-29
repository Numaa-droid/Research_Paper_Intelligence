"""Terms of art a summary uses that the paper never does.

The free-text parts of a report (problem, training procedure, objective) are
written by a model from quoted evidence, and a model fills gaps from general
knowledge: the paper says "label smoothing", the summary says "cross-entropy
loss". Whether a term occurs in the paper is a string question. Only the words
most likely to carry such an import are checked — hyphenated compounds and
acronyms — because ordinary paraphrase vocabulary legitimately differs from the
paper's wording.
"""

from __future__ import annotations

import re

from ..retrieval.index import PaperIndex

_COMPOUND = re.compile(r"\b[A-Za-z]{2,}(?:-[A-Za-z]{2,})+\b")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}[a-z]?\b")
_SQUASH = re.compile(r"[^a-z0-9]+")
_SUFFIXES = ("ally", "ly", "ing", "ed", "es", "s")


def _squash(text: str) -> str:
    return _SQUASH.sub("", text.lower())


class PaperVocabulary:
    """The paper's text, squashed so hyphenation and line breaks do not matter."""

    def __init__(self, index: PaperIndex) -> None:
        self._text = _squash(index.full_text())

    def _occurs(self, term: str) -> bool:
        squashed = _squash(term)
        # "auto-regressively" is the paper's "auto-regressive": inflections count.
        stems = [squashed] + [
            squashed[: -len(suffix)] for suffix in _SUFFIXES if squashed.endswith(suffix)
        ]
        return any((stem == squashed or len(stem) >= 4) and stem in self._text for stem in stems)

    def unstated_terms(self, text: str | None) -> list[str]:
        """Compounds and acronyms in ``text`` that occur nowhere in the paper."""
        found: list[str] = []
        for match in [*_COMPOUND.finditer(text or ""), *_ACRONYM.finditer(text or "")]:
            term = match.group(0)
            if not self._occurs(term) and term not in found:
                found.append(term)
        return found
