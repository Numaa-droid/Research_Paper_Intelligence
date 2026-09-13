"""Provider-independent evaluation of PaperPilot runs."""

from .harness import EvaluationHarness, PaperExpectation, load_expectations
from .metrics import (
    EvaluationMetrics,
    aggregate,
    citation_precision,
    citation_recall,
    claim_coverage,
    evaluate_record,
    numeric_accuracy_score,
    unsupported_claim_rate,
    verification_accuracy,
)

__all__ = [
    "EvaluationHarness",
    "EvaluationMetrics",
    "PaperExpectation",
    "aggregate",
    "citation_precision",
    "citation_recall",
    "claim_coverage",
    "evaluate_record",
    "load_expectations",
    "numeric_accuracy_score",
    "unsupported_claim_rate",
    "verification_accuracy",
]
