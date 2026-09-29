"""The review-protocol checklist: what a literature review says about how it was done.

A review's reproducibility is its protocol — where it searched, for what, over
which period, how it chose what to keep and how it synthesised it. Each item is
looked for in the sentences where the paper speaks for itself (sentences that
report a cited work describe that work's protocol, not this one's). A found item
cites the sentence; a missing one is a gap in the review, which the report lists
as a limitation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..domain.enums import CitationQuality, ElementType
from ..domain.models import Evidence
from ..retrieval.index import PaperIndex
from .attribution import attribution_map

_ITEMS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "Sources searched named",
        "the databases or venues searched are not named",
        re.compile(
            r"\b(?:Scopus|Web\s+of\s+Science|WoS|PubMed|MEDLINE|Embase|IEEE\s*Xplore|"
            r"ACM\s+Digital\s+Library|Google\s+Scholar|arXiv|DBLP|ScienceDirect|SpringerLink|"
            r"Semantic\s+Scholar|PsycINFO|CINAHL|Cochrane|ERIC|JSTOR|SSRN)\b"
        ),
    ),
    (
        "Search terms stated",
        "the search terms or query are not stated",
        re.compile(
            r"\bsearch(?:ed|ing)?\s+(?:for|of)\b|\bkeywords?\b(?!\s*:)|\bsearch\s+(?:string|terms?|"
            r"quer(?:y|ies))\b|\bthe\s+terms?\s+[\"“A-Z]",
            re.IGNORECASE,
        ),
    ),
    (
        "Time window stated",
        "the period covered by the search is not stated",
        re.compile(
            r"\b(?:between|from|period|during|since|until|published\s+in)\b[^.]{0,60}?"
            r"\b(?:19|20)\d{2}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Number of records reported",
        "the number of records found or kept is not reported",
        re.compile(
            r"\b\d[\d,.]*\s+(?:unique\s+)?(?:materials|articles|papers|records|studies|"
            r"publications|documents|items|works|sources|hits)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Inclusion/exclusion criteria stated",
        "no inclusion or exclusion criteria are stated",
        re.compile(
            r"\b(?:inclusion|exclusion|eligibility|selection)\s+criteri|\b(?:were|was)\s+"
            r"(?:included|excluded)\b|\bincluded\s+(?:only|if)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Screening process described",
        "how records were screened (duplicates, titles/abstracts, full texts) is not described",
        re.compile(
            r"\bPRISMA\b|\bscreen(?:ed|ing)\b|\bduplicates?\b|\bfull[\s-]texts?\b|"
            r"\btitles?\s+and\s+abstracts?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Quality or risk-of-bias assessment",
        "the quality or risk of bias of the reviewed works is not assessed",
        re.compile(
            r"\bquality\s+(?:assess|apprais|criteri|score)|\brisk\s+of\s+bias\b|"
            r"\bcritical\s+appraisal\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Synthesis method stated",
        "how the reviewed works were grouped or synthesised is not stated",
        re.compile(
            r"\bthematic\b|\bnarrative\s+synthesis\b|\bmeta-?analy|\bcod(?:ed|ing)\s+"
            r"(?:scheme|the|each)|\bcategori[sz](?:ed|ation)\b|\btaxonomy\b|\bbibliometric\b|"
            r"\bclassif(?:ied|ication)\s+(?:of|into|by)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(slots=True)
class ProtocolItem:
    name: str
    found: bool
    #: What is missing, phrased as a limitation of the review.
    gap: str
    evidence: list[Evidence]


def review_protocol(index: PaperIndex, *, max_evidence: int = 2) -> list[ProtocolItem]:
    """Each protocol item, with the paper's own sentences that state it."""
    amap = attribution_map(index)
    own = [s for s in amap.sentences if s.voice != "reported"]
    items: list[ProtocolItem] = []
    for name, gap, pattern in _ITEMS:
        hits = [s for s in own if pattern.search(s.text)][:max_evidence]
        evidence = []
        for sentence in hits:
            chunk = index.get(sentence.chunk_id)
            evidence.append(
                Evidence(
                    page=sentence.page,
                    section=sentence.section,
                    source_type=chunk.element_type if chunk else ElementType.PARAGRAPH,
                    element_id=chunk.element_id if chunk else None,
                    chunk_id=sentence.chunk_id,
                    quote=sentence.text[:400],
                    confidence=1.0,
                    citation_quality=CitationQuality.EXACT,
                    validation_notes=["sentence selected by the deterministic review-protocol check"],
                )
            )
        items.append(ProtocolItem(name=name, found=bool(hits), gap=gap, evidence=evidence))
    return items
