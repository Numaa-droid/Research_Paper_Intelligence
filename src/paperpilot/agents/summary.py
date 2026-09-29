"""Summary Agent: the grounded single-agent baseline.

This agent alone produces a usable report. It exists partly as a baseline: if the
multi-agent pipeline does not beat it on citation precision and claim coverage in
the evaluation harness, the extra agents are not earning their cost.
"""

from __future__ import annotations

from typing import Any

from ..domain.models import Finding, SummaryResult
from ..response.reader import DocumentReader
from ..retrieval.retriever import RetrievalQuery
from .base import AgentContext, BaseAgent
from .grounding import Grounder, parse_findings
from .prompts.registry import SUMMARY

_QUERIES = (
    "what problem does this paper address and why does it matter",
    "main contribution and novelty of the proposed approach",
    "dataset used for training and evaluation",
    "headline experimental results and improvement over baselines",
    "limitations, assumptions and threats to validity stated by the authors",
)


class SummaryAgent(BaseAgent):
    name = "summary"
    prompt = SUMMARY
    context_limit = 14
    anchor_section_hints = ("abstract", "introduction", "contribution", "conclusion", "limitation")

    def queries(self, context: AgentContext) -> list[RetrievalQuery]:
        queries = [RetrievalQuery(text=q, top_k=4) for q in _QUERIES]
        # The abstract and conclusion carry the authors' own framing; make sure
        # they are present even if they did not win on similarity.
        queries.append(
            RetrievalQuery(
                text="abstract conclusion summary of contributions",
                top_k=4,
                sections=["abstract", "conclusion", "introduction"],
            )
        )
        return queries

    def build_result(
        self, reader: DocumentReader, context: AgentContext, grounder: Grounder
    ) -> tuple[list[Finding], dict[str, Any], list[str]]:
        findings = parse_findings(reader, grounder, category="summary")
        summary = SummaryResult(
            research_problem=reader.text(reader.section("research problem", "problem")),
            motivation=reader.text(reader.section("motivation")),
            main_contribution=reader.text(
                reader.section("main contribution", "contributions", "contribution")
            ),
            dataset=reader.text(reader.section("dataset", "datasets", "data")),
            proposed_method=reader.text(reader.section("proposed method", "method", "approach")),
            main_results=reader.text(reader.section("main results", "results")),
            limitations=reader.text(reader.section("limitations")),
            assumptions=reader.text(reader.section("assumptions")),
            concise_summary=reader.text(reader.section("concise summary", "summary", "tl dr")),
            unresolved_questions=reader.items(
                reader.section("unresolved questions", "open questions")
            ),
            findings=findings,
        )
        return findings, summary.model_dump(mode="json", exclude={"findings"}), (
            summary.unresolved_questions
        )
