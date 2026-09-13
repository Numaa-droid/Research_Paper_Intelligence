"""Experiment Agent: datasets, baselines, metrics, ablations — and the numbers.

Every reported number is re-checked against its own evidence deterministically
before it reaches the report, so a value the model transcribed wrongly is
downgraded rather than printed as fact.
"""

from __future__ import annotations

from typing import Any

from ..domain.enums import ElementType, FindingStatus
from ..domain.models import ExperimentResult, Finding, MetricResult
from ..retrieval.retriever import RetrievalQuery, RetrievalResult
from ..validation.evidence import validate_evidence
from ..validation.numeric import check_claim_numbers, numeric_accuracy
from ..validation.schema import as_str_dict, as_str_list, coerce_status, parse_findings
from .base import AgentContext, BaseAgent, evidence_from_payload, require_dict
from .prompts.registry import EXPERIMENTS

_EXPERIMENT_SECTIONS = ["experiment", "evaluation", "result", "setup", "ablation", "analysis"]


class ExperimentAgent(BaseAgent):
    name = "experiments"
    prompt = EXPERIMENTS
    context_limit = 16

    def queries(self, context: AgentContext) -> list[RetrievalQuery]:
        priority = [s for s in context.priority_sections if s] or _EXPERIMENT_SECTIONS
        return [
            RetrievalQuery(text="datasets train validation test splits", top_k=4),
            RetrievalQuery(text="baselines compared against", top_k=4),
            RetrievalQuery(text="evaluation metrics reported", top_k=3),
            # Tables are where the results live; ask for them explicitly.
            RetrievalQuery(
                text="main results table comparison numbers",
                top_k=5,
                element_types=[ElementType.TABLE],
            ),
            RetrievalQuery(text="ablation study removing component effect", top_k=4),
            RetrievalQuery(
                text="hyperparameters learning rate batch size epochs seeds",
                top_k=3,
            ),
            RetrievalQuery(
                text="statistical significance confidence interval standard deviation",
                top_k=2,
            ),
            RetrievalQuery(
                text="error analysis failure cases where the method underperforms",
                top_k=3,
                sections=priority,
            ),
        ]

    def build_result(
        self, payload: dict[str, Any], context: AgentContext, results: list[RetrievalResult]
    ) -> tuple[list[Finding], dict[str, Any], list[str]]:
        data = require_dict(payload, self.name)
        findings = parse_findings(data.get("findings"), agent=self.name)

        reported: list[MetricResult] = []
        for item in data.get("reported_results") or []:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric") or "").strip()
            value = str(item.get("value") or "").strip()
            if not (metric and value):
                continue
            evidence = evidence_from_payload(item.get("evidence"), agent=self.name)
            for piece in evidence:
                validate_evidence(piece, context.index)
            reported.append(
                MetricResult(
                    metric=metric,
                    value=value,
                    system=str(item.get("system") or "").strip() or None,
                    dataset=str(item.get("dataset") or "").strip() or None,
                    split=str(item.get("split") or "").strip() or None,
                    is_best=bool(item.get("is_best")),
                    evidence=evidence,
                )
            )

        unsupported_numbers = self._check_reported_numbers(reported)

        result = ExperimentResult(
            datasets=as_str_list(data.get("datasets")),
            splits=str(data.get("splits") or ""),
            baselines=as_str_list(data.get("baselines")),
            metrics=as_str_list(data.get("metrics")),
            reported_results=reported,
            ablations=as_str_list(data.get("ablations")),
            hyperparameters=as_str_dict(data.get("hyperparameters")),
            error_analysis=str(data.get("error_analysis") or ""),
            statistical_significance=str(data.get("statistical_significance") or ""),
            conclusions_supported=coerce_status(data.get("conclusions_supported")),
            conclusions_assessment=str(data.get("conclusions_assessment") or ""),
            findings=findings,
        )

        open_questions: list[str] = []
        if unsupported_numbers:
            open_questions.append(
                "These reported values could not be located in the cited evidence and "
                "should be treated as unverified: " + ", ".join(sorted(unsupported_numbers))
            )
        if result.conclusions_supported is FindingStatus.UNKNOWN:
            open_questions.append(
                "Whether the experiments support the paper's conclusions could not be "
                "determined from the retrieved evidence."
            )

        payload_out = result.model_dump(mode="json", exclude={"findings"})
        payload_out["unverified_numbers"] = sorted(unsupported_numbers)
        return findings, payload_out, open_questions

    def _check_reported_numbers(self, reported: list[MetricResult]) -> set[str]:
        """Deterministically confirm each reported value appears in its evidence."""
        unsupported: set[str] = set()
        for metric in reported:
            checks = check_claim_numbers(metric.value, metric.evidence)
            if numeric_accuracy(checks) < 1.0:
                label = f"{metric.system or 'unknown system'} {metric.metric}={metric.value}"
                unsupported.add(label)
                for evidence in metric.evidence:
                    evidence.validation_notes.append(
                        f"value '{metric.value}' was not found in this quote"
                    )
        return unsupported
