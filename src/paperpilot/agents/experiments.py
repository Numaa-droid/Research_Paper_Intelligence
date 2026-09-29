"""Experiment Agent: datasets, baselines, metrics, ablations — and the numbers.

Every reported number is re-checked against its own evidence deterministically
before it reaches the report, so a value the model transcribed wrongly is
downgraded rather than printed as fact.
"""

from __future__ import annotations

from typing import Any

from ..domain.enums import ElementType, FindingStatus
from ..domain.models import Evidence, ExperimentResult, Finding, MetricResult
from ..response.reader import DocumentReader, clean_text, parse_bool, quotes_from_text
from ..response.schema import Citation
from ..retrieval.retriever import RetrievalQuery
from ..validation.consistency import find_value_conflicts
from ..validation.evidence import validate_evidence
from ..validation.numeric import check_claim_numbers, extract_numbers, numeric_accuracy
from ..validation.values import locate_value, value_in_text
from .base import AgentContext, BaseAgent
from .grounding import Grounder, parse_findings, parse_status
from .prompts.registry import EXPERIMENTS

_RESULT_COLUMNS: dict[str, tuple[str, ...]] = {
    "system": ("model", "method", "approach"),
    "dataset": ("benchmark", "data"),
    "split": ("subset", "set"),
    "metric": ("measure",),
    "value": ("score", "result", "number"),
    "best": ("is best", "best result"),
    "source": ("sources", "citation", "ref"),
    "quote": ("evidence", "verbatim quote", "verbatim text"),
}


def _cell(record: dict, name: str) -> str:
    cell = record.get(name)
    return clean_text(cell.value) if cell is not None else ""

_EXPERIMENT_SECTIONS = ["experiment", "evaluation", "result", "setup", "ablation", "analysis"]


class ExperimentAgent(BaseAgent):
    name = "experiments"
    prompt = EXPERIMENTS
    context_limit = 20
    anchor_element_types = (ElementType.TABLE,)
    anchor_section_hints = ("experiment", "evaluation", "result", "metric", "dataset", "benchmark", "baseline", "ablation", "setup", "hyperparameter", "implementation", "training")

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
        self, reader: DocumentReader, context: AgentContext, grounder: Grounder
    ) -> tuple[list[Finding], dict[str, Any], list[str]]:
        findings = parse_findings(reader, grounder, category="experiments")

        reported: list[MetricResult] = []
        skipped_values: list[str] = []
        for table in reader.tables(reader.section("reported results", "results")):
            for record in reader.records(table, _RESULT_COLUMNS):
                metric = _cell(record, "metric")
                value = _cell(record, "value")
                if not (metric and value):
                    continue
                if not extract_numbers(value, skip_trivial=False):
                    # "Not numeric, hypothesized highest" is a description, not a
                    # result; a results table only holds values the paper reports.
                    skipped_values.append(f"{_cell(record, 'system') or 'unknown system'} {metric}")
                    continue
                system = _cell(record, "system") or None
                evidence = self._row_evidence(record, value, grounder)
                for piece in evidence:
                    validate_evidence(piece, context.index)
                if not any(p.is_verified and value_in_text(value, p.quote) for p in evidence):
                    # The row cited a caption or a nearby sentence; find the line of
                    # the paper that actually holds this value for this system.
                    cited = [
                        chunk
                        for name in ("source", "quote")
                        if record.get(name) is not None
                        for c in record[name].citations
                        if (chunk := grounder.sources.chunk(c.label)) is not None
                    ]
                    located = locate_value(value, context.index, name=system, cited=cited)
                    if located is not None:
                        validate_evidence(located, context.index)
                        evidence.insert(0, located)
                reported.append(
                    MetricResult(
                        metric=metric,
                        value=value,
                        system=system,
                        dataset=_cell(record, "dataset") or None,
                        split=_cell(record, "split") or None,
                        is_best=bool(parse_bool(_cell(record, "best")))
                        or bool(record.get("value") and record["value"].strong),
                        evidence=evidence,
                    )
                )

        conclusions = reader.section("conclusions", "conclusions supported", "conclusion")
        conclusion_fields = reader.fields(conclusions)
        hyperparameters = reader.key_value_map(reader.section("hyperparameters"))

        unsupported_numbers = self._check_reported_numbers(reported)
        conflicts = find_value_conflicts(
            [m for m in reported if f"{m.system or 'unknown system'} {m.metric}={m.value}"
             not in unsupported_numbers],
            context.index,
        )

        result = ExperimentResult(
            datasets=reader.items(reader.section("datasets", "dataset")),
            splits=reader.text(reader.section("splits", "data splits")),
            baselines=reader.items(reader.section("baselines")),
            metrics=reader.items(reader.section("metrics", "evaluation metrics")),
            reported_results=reported,
            ablations=reader.items(reader.section("ablations", "ablation study")),
            hyperparameters=hyperparameters,
            error_analysis=reader.text(reader.section("error analysis")),
            statistical_significance=reader.text(reader.section("statistical significance")),
            conclusions_supported=parse_status(
                reader.field_text(conclusion_fields, "supported", "status", "conclusions supported")
            ),
            conclusions_assessment=reader.paragraphs(conclusions),
            findings=findings,
        )

        # What the pipeline itself could not do is a note on how the report was
        # produced, not a question the paper leaves open: it goes in the
        # payload's diagnostics, which the report prints under its provenance.
        diagnostics: list[str] = []
        if skipped_values:
            diagnostics.append(
                "These rows had no numeric value in the paper's text and were left out of "
                "the results table: " + ", ".join(skipped_values)
            )
        if unsupported_numbers:
            diagnostics.append(
                "These reported values could not be located in the cited evidence and "
                "should be treated as unverified: " + ", ".join(sorted(unsupported_numbers))
            )
        if result.conclusions_supported is FindingStatus.UNKNOWN:
            diagnostics.append(
                "Whether the experiments support the paper's conclusions could not be "
                "determined from the retrieved evidence."
            )

        # A value the paper states twice, differently, is a question about the paper.
        open_questions = [conflict.describe() for conflict in conflicts]

        payload_out = result.model_dump(mode="json", exclude={"findings"})
        payload_out["unverified_numbers"] = sorted(unsupported_numbers)
        payload_out["value_conflicts"] = [conflict.to_dict() for conflict in conflicts]
        payload_out["diagnostics"] = diagnostics
        return findings, payload_out, open_questions

    @staticmethod
    def _row_evidence(record: dict, value: str, grounder: Grounder) -> list[Evidence]:
        """The row's quote if it has one, else the cited source line holding the value."""
        labels = [
            c.label
            for name in ("source", "quote")
            if record.get(name) is not None
            for c in record[name].citations
        ]
        quote_cell = record.get("quote")
        if quote_cell is not None and quote_cell.value:
            refs = quotes_from_text(quote_cell.text)
            for ref in refs:
                if not ref.citations:
                    ref.citations = [Citation(label=label, raw=f"[{label}]") for label in labels]
            evidence = grounder.evidence_list(refs)
            if evidence:
                return evidence
        return grounder.evidence_for_value(value, labels)

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
