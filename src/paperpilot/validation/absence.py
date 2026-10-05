"""Claims that the paper does not say something, checked against the paper.

An agent sees only the passages retrieved for it, so "training data and batching
details are not stated" means "not in what I was shown". When the paper has a
section titled *Training Data and Batching*, the report would contradict both
the paper and itself. Absence is checkable without a model: take the subject of
the claim and look for it in the paper's own headings, then (for a subject of two
or more words) in its sentences. Found means the claim is wrong, and the report
says where the paper covers it instead. Not found leaves the claim alone: it may
well be true.

Only statements about what a text *says* are treated as absence claims ("not
stated", "not described", "details not provided in the excerpt"). A sentence such
as "no other help is available" reports the paper's content and is left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..domain.enums import ElementType
from ..retrieval.index import PaperIndex

_STRONG = (
    r"stated|specified|described|reported|mentioned|detailed|discussed|defined|explained"
    r"|documented|quantified"
)
#: Verbs that also describe a paper's content ("no guarantee is provided"):
#: counted only when the sentence says *where* it is missing ("in the excerpt").
_WEAK = r"provided|given|available|included|presented|listed|shown"
_TEXT_REF = re.compile(
    r"\b(?:in|from|within)\s+(?:the\s+)?(?:provided\s+|given\s+|retrieved\s+|available\s+)?"
    r"(?:context|excerpts?|passages?|text|paper|document|sources?|snippets?)\b|\bhere\b",
    re.IGNORECASE,
)
_HEDGE = r"(?:explicitly\s+|clearly\s+|directly\s+|further\s+|yet\s+|fully\s+)?"
#: "<subject> is/are not (explicitly) stated…"
_COPULA = re.compile(
    rf"^(?P<s>.+?)\s+(?:is|are|was|were)\s+not\s+{_HEDGE}(?P<v>{_STRONG}|{_WEAK})\b",
    re.IGNORECASE,
)
#: "No <subject> is/are reported…"
_NO = re.compile(
    rf"^no\s+(?P<s>.+?)\s+(?:is|are|was|were)\s+{_HEDGE}(?P<v>{_STRONG}|{_WEAK})\b",
    re.IGNORECASE,
)
#: "The paper does not report <subject>", "… have not yet reported <subject>"
_ACTIVE = re.compile(
    rf"\b(?:does|do|did|has|have)\s+not\s+{_HEDGE}{_HEDGE}"
    r"(?P<v>state|specify|describe|report|mention|detail|discuss|define|explain|provide|give)"
    r"(?:s|d|ed)?\s+(?P<s>[^,.;()]+)",
    re.IGNORECASE,
)
#: "Not (explicitly) stated <subject>" or a bare "Not explicitly stated".
_BARE = re.compile(rf"^not\s+{_HEDGE}(?P<v>{_STRONG}|{_WEAK})\b(?P<s>.*)$", re.IGNORECASE)
#: "(not detailed here)", "(details not provided in the excerpt)"
_PAREN = re.compile(
    rf"\s*\((?:the\s+)?(?:details?|specifics?|values?)?\s*(?:are\s+|is\s+)?not\s+{_HEDGE}"
    rf"(?:{_STRONG}|{_WEAK})\b[^)]*\)",
    re.IGNORECASE,
)
#: Where the noun phrase that a parenthetical refers to ends.
_VERB = re.compile(
    r"\b(?:is|are|was|were|be|been|has|have|had|can|may|will|would|should|does|do|did"
    r"|varies|vary|varied|uses|use|used|employs|employed|follows|follow|followed|includes"
    r"|include|involves|consists|relies|applies|applied|takes|uses)\b",
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_WORD = re.compile(r"[a-z][a-z-]*", re.IGNORECASE)
_STOP = frozenset(
    "a an the of and or for in on to with by at from as its their this that these those any "
    "some such other into over about regarding concerning how what which whether there it "
    "details detail detailed information info specific specifics exact explicit explicitly "
    "further additional more particular paper papers author authors study text context excerpt "
    "excerpts passage passages value values aspects description descriptions "
    "full complete precise yet technique techniques method methods approach approaches "
    "procedure procedures strategy strategies scheme schemes setting settings".split()
)
_SKIP_SECTIONS = re.compile(r"^\W*(?:references|bibliography|acknowledg)", re.IGNORECASE)
#: A sentence that names a subject only to say it is missing, planned or hoped
#: for ("results will be reported in future work") does not cover it.
_NOT_COVERING = re.compile(
    r"\b(?:not|no|never|without|neither|nor|will|would|shall|future|plan|plans|planned|propose"
    r"|proposed|intend|intends|expected|expect|could|might|may|hope|aim|aims|leave|left)\b",
    re.IGNORECASE,
)
#: What may precede "does not report X" for the whole sentence to be the claim.
_PLAIN_SUBJECT = re.compile(r"^(?:\w+\s+){0,3}\w*\s*$")


def _stem(word: str) -> str:
    word = word.lower().strip("-")
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text) if w.lower() not in _STOP and len(w) > 1}


@dataclass(slots=True)
class Contradiction:
    """An absence claim the paper itself refutes."""

    claim: str         # the text that says the paper is silent
    subject: str       # what it says the paper is silent about
    where: str         # where the paper covers it
    partial: bool      # True when only a parenthetical was the claim
    remove: bool = True  # False when the claim is one part of a longer sentence

    def describe(self) -> str:
        return f"the model wrote “{self.claim}”, but the paper covers {self.subject} in {self.where}"


class PaperCoverage:
    """The paper's headings and sentences, for checking claims that it is silent."""

    def __init__(self, index: PaperIndex) -> None:
        self._headings: list[tuple[set[str], str, int]] = []
        seen: set[str] = set()
        for chunk in index.chunks:
            for heading in (chunk.section, chunk.subsection):
                if heading and heading not in seen and not _SKIP_SECTIONS.match(heading):
                    seen.add(heading)
                    self._headings.append((_terms(heading), heading, chunk.page))
        self._sentences: list[tuple[set[str], str, int]] = []
        for chunk in index.chunks:
            if chunk.element_type is ElementType.TABLE or _SKIP_SECTIONS.match(chunk.section or ""):
                continue
            where = f"§{chunk.citation_section}" if chunk.citation_section else "the text"
            for sentence in _SENTENCES.split(" ".join(chunk.text.split())):
                if not _NOT_COVERING.search(sentence):
                    self._sentences.append((_terms(sentence), where, chunk.page))

    def covers(self, subject: str) -> str | None:
        """Where the paper covers ``subject``, or None."""
        wanted = _terms(subject)
        if not wanted:
            return None
        for terms, heading, page in self._headings:
            if wanted <= terms:
                return f"§{heading} (p. {page})"
        if len(wanted) >= 2:
            for terms, where, page in self._sentences:
                if wanted <= terms:
                    return f"{where} (p. {page})"
        return None

    def check(self, text: str | None, *, subject: str | None = None) -> tuple[str, list[Contradiction]]:
        """``text`` with refuted absence claims removed, and what was removed.

        ``subject`` names what a field is about, for a bare "Not explicitly stated".
        """
        if not text:
            return text or "", []
        kept: list[str] = []
        found: list[Contradiction] = []
        for sentence in _SENTENCES.split(text.strip()):
            claim = sentence.strip()
            contradiction = self._check_sentence(claim, subject)
            if contradiction is None:
                kept.append(sentence)
                continue
            found.append(contradiction)
            if contradiction.partial:
                kept.append(_PAREN.sub("", sentence))
            elif not contradiction.remove:
                # The sentence says more than the claim; keep it and let the note correct it.
                kept.append(sentence)
        return " ".join(s.strip() for s in kept if s.strip()), found

    def _check_sentence(self, sentence: str, field_subject: str | None) -> Contradiction | None:
        body = sentence.rstrip(" .")
        paren = _PAREN.search(body)
        if paren:
            head = body[: paren.start()]
            verb = _VERB.search(head)
            noun_phrase = head[: verb.start()] if verb else head
            where = self.covers(noun_phrase)
            if where:
                subject = re.sub(r"^(?:the|a|an)\s+", "", noun_phrase.strip(), flags=re.IGNORECASE)
                return Contradiction(paren.group(0).strip(), subject.lower(), where, partial=True)
            return None
        for pattern in (_COPULA, _NO, _ACTIVE, _BARE):
            match = pattern.search(body) if pattern is _ACTIVE else pattern.match(body)
            if not match:
                continue
            verb = match.group("v").lower()
            if re.fullmatch(_WEAK, verb) and not _TEXT_REF.search(body):
                return None
            subject = _TEXT_REF.sub(" ", match.group("s")).strip(" :,-") or (field_subject or "")
            where = self.covers(subject)
            if where:
                subject = re.sub(r"^(?:the|a|an)\s+", "", subject, flags=re.IGNORECASE)
                whole = pattern is not _ACTIVE or (
                    _PLAIN_SUBJECT.match(body[: match.start()]) is not None
                    and match.end() >= len(body) - 1
                )
                return Contradiction(
                    sentence.strip(), subject.lower(), where, partial=False, remove=whole
                )
            return None
        return None
