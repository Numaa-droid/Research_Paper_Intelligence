"""Methodology Agent: how the proposed method actually works."""

from __future__ import annotations

import re
from typing import Any

from ..domain.enums import ElementType
from ..domain.models import Finding, MethodologyResult, MethodologyStep
from ..response.reader import DocumentReader, clean_text, parse_bool
from ..retrieval.retriever import RetrievalQuery
from ..validation.evidence import validate_evidence
from .base import AgentContext, BaseAgent
from .grounding import Grounder, parse_findings
from .prompts.registry import METHODOLOGY

#: "Step 2: Encoder", "2. Encoder", "Step 2 — Encoder", "Encoder".
_STEP_HEADING = re.compile(r"^\s*(?:step\s*)?(\d+)?\s*[:.)\-–—]?\s*(.*)$", re.IGNORECASE)

_METHOD_SECTIONS = ["method", "approach", "model", "architecture", "framework", "algorithm"]


class MethodologyAgent(BaseAgent):
    name = "methodology"
    prompt = METHODOLOGY
    context_limit = 14
    anchor_section_hints = ("method", "approach", "architecture", "framework", "algorithm", "pipeline", "model")

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
        self, reader: DocumentReader, context: AgentContext, grounder: Grounder
    ) -> tuple[list[Finding], dict[str, Any], list[str]]:
        findings = parse_findings(reader, grounder, category="methodology")
        steps = self._steps(reader, grounder)

        # Step evidence is part of the report, so it must be validated too.
        for step in steps:
            for evidence in step.evidence:
                validate_evidence(evidence, context.index)

        pseudocode_section = reader.section("pseudocode", "algorithm")
        code = reader.code(pseudocode_section)
        pseudocode = code.text.strip() if code is not None else reader.text(pseudocode_section)

        result = MethodologyResult(
            inputs=reader.text(reader.section("inputs", "input")),
            outputs=reader.text(reader.section("outputs", "output")),
            architecture=reader.text(reader.section("architecture", "model architecture")),
            training_procedure=reader.text(reader.section("training procedure", "training")),
            inference_procedure=reader.text(reader.section("inference procedure", "inference")),
            design_decisions=reader.items(reader.section("design decisions")),
            comparison_with_prior_work=reader.text(
                reader.section("comparison with prior work", "prior work", "related work")
            ),
            pipeline_steps=steps,
            pseudocode=pseudocode or None,
            findings=findings,
        )
        return findings, result.model_dump(mode="json", exclude={"findings"}), []

    @staticmethod
    def _steps(reader: DocumentReader, grounder: Grounder) -> list[MethodologyStep]:
        section = reader.section("pipeline steps", "pipeline", "steps")
        steps: list[MethodologyStep] = []
        for order, sub in enumerate(reader.subsections(section), start=1):
            match = _STEP_HEADING.match(sub.heading)
            number = int(match.group(1)) if match and match.group(1) else order
            name = clean_text(match.group(2) if match else sub.heading) or f"Step {order}"
            fields = reader.fields(sub)
            # Models write the description as prose, as "- Description: …", or as a
            # plain bullet under "- Inferred: no"; any of them is the description.
            description = (
                reader.paragraphs(sub)
                or reader.field_text(fields, "description", "what it does")
                or "\n".join(
                    clean_text(item.text) for item in reader.list_items(sub) if not item.key
                ).strip()
            )
            steps.append(
                MethodologyStep(
                    order=max(1, number),
                    name=name,
                    description=description,
                    evidence=grounder.evidence_list(reader.quotes(sub)),
                    is_inferred=bool(parse_bool(reader.field_text(fields, "inferred", "is inferred"))),
                )
            )
        if steps:
            return sorted(steps, key=lambda s: s.order)

        # Fallback: a numbered list, "1. Name: description".
        for order, item in enumerate(reader.list_items(section), start=1):
            name = clean_text(item.key) if item.key else f"Step {order}"
            description = clean_text(item.value if item.key else item.text)
            if description:
                steps.append(MethodologyStep(order=order, name=name, description=description))
        return steps
