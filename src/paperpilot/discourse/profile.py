"""What kind of paper this is, decided from its text.

The kind changes what the report's sections *mean*. The "methodology" of a
literature review is its search and synthesis procedure, not the procedure of a
study it summarises; its "dataset" is the corpus it searched. Getting the kind
wrong is how a review ends up described as an experiment.

The decision uses only signals that can be counted:

* the title and keywords ("review", "survey", "meta-analysis", …);
* the share of body sentences that report cited works (see
  :mod:`paperpilot.discourse.attribution`);
* sentences where the paper, in its own voice, trains, evaluates or proposes;
* theorem/proof structure.

Every signal that fired is recorded, so the plan and the report can say *why*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..domain.enums import ElementType
from ..domain.models import PaperKind
from ..retrieval.index import PaperIndex
from .attribution import attribution_map

_REVIEW_WORDS = re.compile(
    r"\b(?:review|survey|overview|meta-?analys[ie]s|systematic\s+mapping|literature|"
    r"scoping|state[\s-]of[\s-]the[\s-]art|bibliometric)\b",
    re.IGNORECASE,
)
_OWN_EMPIRICAL = re.compile(
    r"\b(?:we|our)\b[^.]{0,80}?\b(?:train|trained|training|evaluat\w*|experiment\w*|propos\w*|"
    r"achiev\w*|outperform\w*|benchmark\w*|implement\w*|fine-tun\w*|ablat\w*|measur\w*|"
    r"collect\w*|recruit\w*|survey(?:ed)?|interview\w*|simulat\w*|compar\w*)\b",
    re.IGNORECASE,
)
_THEOREM = re.compile(r"\b(?:theorem|lemma|proposition|corollary|proof)\b", re.IGNORECASE)
_EXPERIMENT_HEADING = re.compile(
    r"\b(?:experiment|evaluation|results?|ablation|benchmark)", re.IGNORECASE
)

#: A titled review needs at least this share of reporting sentences…
TITLED_REVIEW_SHARE = 0.15
#: …an untitled one this much, and few own-voice experiments.
UNTITLED_REVIEW_SHARE = 0.35
#: Distinct cited works a review reports, at least.
MIN_REVIEWED_WORKS = 5


@dataclass(slots=True)
class PaperProfile:
    kind: PaperKind
    signals: list[str] = field(default_factory=list)
    reported_share: float = 0.0
    reviewed_works: int = 0
    own_empirical_sentences: int = 0


def profile_paper(index: PaperIndex) -> PaperProfile:
    """Classify the paper; deterministic for a given index."""
    metadata = index.metadata
    amap = attribution_map(index)
    share = amap.reported_share
    works = len(amap.cited_sources)
    own_empirical = sum(
        1 for s in amap.sentences if s.voice != "reported" and _OWN_EMPIRICAL.search(s.text)
    )
    headings = index.sections()
    theorem_hits = sum(1 for h in headings if _THEOREM.search(h)) + sum(
        1 for s in amap.sentences if s.voice != "reported" and _THEOREM.search(s.text[:40])
    )
    experiment_headings = [h for h in headings if _EXPERIMENT_HEADING.search(h)]

    frame = " ".join([metadata.title or "", " ".join(metadata.keywords or [])])
    titled = _REVIEW_WORDS.search(frame)

    signals = [
        f"{share:.0%} of body sentences report a cited work ({works} distinct work(s))",
        f"{own_empirical} sentence(s) where the paper itself trains, evaluates, proposes or measures",
    ]
    if titled:
        signals.insert(0, f"the title or keywords say “{titled.group(0)}”")
    if experiment_headings:
        signals.append("experiment/result heading(s): " + ", ".join(experiment_headings[:3]))
    if theorem_hits:
        signals.append(f"{theorem_hits} theorem/lemma/proof marker(s)")

    kind: PaperKind
    if works >= MIN_REVIEWED_WORKS and (
        (titled and share >= TITLED_REVIEW_SHARE)
        or (share >= UNTITLED_REVIEW_SHARE and own_empirical < 3)
    ):
        kind = "review"
    elif theorem_hits >= 3 and own_empirical <= 2:
        kind = "theoretical"
    elif own_empirical >= 2 or experiment_headings or _numeric_tables(index):
        kind = "empirical"
    else:
        kind = "conceptual"

    return PaperProfile(
        kind=kind,
        signals=signals,
        reported_share=share,
        reviewed_works=works,
        own_empirical_sentences=own_empirical,
    )


def _numeric_tables(index: PaperIndex) -> bool:
    """A table with several numeric cells is a results table in all but name."""
    for chunk in index.by_element_type(ElementType.TABLE):
        if len(re.findall(r"\b\d+(?:\.\d+)\b", chunk.text)) >= 6:
            return True
    return False
