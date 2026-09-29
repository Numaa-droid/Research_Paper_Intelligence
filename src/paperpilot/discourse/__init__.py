"""Deterministic discourse analysis: who is speaking in each sentence of a paper.

A quote can be verbatim and still be misread. "In [20] the authors found X" is in
the paper, but X is a finding of work [20], not of the paper. Whether a sentence
reports another work is a question about its wording and its place in the text,
so it is answered here without a model, once per paper, and every later stage
(the agents' context, the reconcile pass, the report) reads the same answer.
"""

from .attribution import (
    AttributionMap,
    SentenceVoice,
    attribution_map,
    evidence_attribution,
    text_attribution,
)
from .profile import PaperKind, PaperProfile, profile_paper
from .protocol import ProtocolItem, review_protocol
from .structure import heading_numbering_issues, heading_sequence

__all__ = [
    "AttributionMap",
    "PaperKind",
    "PaperProfile",
    "ProtocolItem",
    "SentenceVoice",
    "attribution_map",
    "evidence_attribution",
    "heading_numbering_issues",
    "heading_sequence",
    "profile_paper",
    "review_protocol",
    "text_attribution",
]
