"""Methodology Agent: how the proposed method actually works."""

from __future__ import annotations

from typing import Any

from ..domain.enums import ElementType
from ..domain.models import Finding, MethodologyResult, MethodologyStep
from ..retrieval.retriever import RetrievalQuery, RetrievalResult
from ..validation.schema import as_str_list, parse_findings
from .base import AgentContext, BaseAgent, evidence_from_payload, require_dict
from .prompts.registry import METHODOLOGY

_METHOD_SECTIONS = ["method", "approach", "model", "architecture", "framework", "algorithm"]


class MethodologyAgent(BaseAgent):
    name = "methodology"
    prompt = METHODOLOGY
    context_limit = 14

    def queries(self, context: AgentContext) -> list[RetrievalQuery]:
        priority = [s for s in context.priority_sections if s] or _METHOD_SECTIONS
        return [
            RetrievalQuery(text="proposed method architecture and pipeline", top_k=5),
            RetrievalQuery(
                text="model inputs outputs and components",
                top_k=4,
                sections=priority,
            ),
            RetrievalQuery(text="training procedure objective optimizer schedule", top_k=4),
            RetrievalQuery(text="inference procedure at test time decoding", top_k=3),
            RetrievalQuery(
                text="design decisions and differences from prior approaches", top_k=3
            ),
            RetrievalQuery(
                text="algorithm pseudocode steps figure of the architecture",
                top_k=3,
                element_types=[ElementType.FIGURE, ElementType.EQUATION],
            ),
        ]

    def build_result(
        self, payload: dict[str, Any], context: AgentContext, results: list[RetrievalResult]
    ) -> tuple[list[Finding], dict[str, Any], list[str]]:
        data = require_dict(payload, self.name)
        findings = parse_findings(data.get("findings"), agent=self.name)

        steps: list[MethodologyStep] = []
        for order, item in enumerate(data.get("pipeline_steps") or [], start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            description = str(item.get("description") or "").strip()
            if not (name or description):
                continue
            steps.append(
                MethodologyStep(
                    order=int(item.get("order") or order),
                    name=name or f"Step {order}",
                    description=description,
                    evidence=evidence_from_payload(item.get("evidence"), agent=self.name),
                    is_inferred=bool(item.get("is_inferred")),
                )
            )

        # Step evidence is part of the report, so it must be validated too.
        from ..validation.evidence import validate_evidence

        for step in steps:
            for evidence in step.evidence:
                validate_evidence(evidence, context.index)

        pseudocode = data.get("pseudocode")
        result = MethodologyResult(
            inputs=str(data.get("inputs") or ""),
            outputs=str(data.get("outputs") or ""),
            architecture=str(data.get("architecture") or ""),
            training_procedure=str(data.get("training_procedure") or ""),
            inference_procedure=str(data.get("inference_procedure") or ""),
            design_decisions=as_str_list(data.get("design_decisions")),
            comparison_with_prior_work=str(data.get("comparison_with_prior_work") or ""),
            pipeline_steps=sorted(steps, key=lambda s: s.order),
            pseudocode=str(pseudocode).strip() if pseudocode else None,
            findings=findings,
        )
        return findings, result.model_dump(mode="json", exclude={"findings"}), []
