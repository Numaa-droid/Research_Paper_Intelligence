"""Summary Agent: the grounded single-agent baseline.

This agent alone produces a usable report. It exists partly as a baseline: if the
multi-agent pipeline does not beat it on citation precision and claim coverage in
the evaluation harness, the extra agents are not earning their cost.
"""

from __future__ import annotations

from typing import Any

from ..domain.models import Finding, SummaryResult
from ..retrieval.retriever import RetrievalQuery, RetrievalResult
from ..validation.schema import as_str_list, parse_findings
from .base import AgentContext, BaseAgent, require_dict
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
        self, payload: dict[str, Any], context: AgentContext, results: list[RetrievalResult]
    ) -> tuple[list[Finding], dict[str, Any], list[str]]:
        data = require_dict(payload, self.name)
        findings = parse_findings(data.get("findings"), agent=self.name)
        summary = SummaryResult(
            research_problem=str(data.get("research_problem") or ""),
            motivation=str(data.get("motivation") or ""),
            main_contribution=str(data.get("main_contribution") or ""),
            dataset=str(data.get("dataset") or ""),
            proposed_method=str(data.get("proposed_method") or ""),
            main_results=str(data.get("main_results") or ""),
            limitations=str(data.get("limitations") or ""),
            assumptions=str(data.get("assumptions") or ""),
            concise_summary=str(data.get("concise_summary") or ""),
            unresolved_questions=as_str_list(data.get("unresolved_questions")),
            findings=findings,
        )
        return findings, summary.model_dump(mode="json", exclude={"findings"}), (
            summary.unresolved_questions
        )
