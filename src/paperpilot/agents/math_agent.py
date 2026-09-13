"""Math Agent: activated only when the paper actually contains mathematics.

Activation is a deterministic decision made from the index, not a judgement call
delegated to a model: a paper with two inline symbols does not need a Math Agent,
and running one anyway would spend tokens to produce filler.
"""

from __future__ import annotations

from typing import Any

from ..domain.enums import ElementType
from ..domain.models import EquationExplanation, Finding, MathResult
from ..retrieval.retriever import RetrievalQuery, RetrievalResult
from ..validation.evidence import validate_evidence
from ..validation.schema import as_str_dict, as_str_list, parse_findings
from .base import AgentContext, BaseAgent, evidence_from_payload, require_dict
from .prompts.registry import MATH

#: Words that indicate a real mathematical treatment rather than incidental notation.
_MATH_KEYWORDS = (
    "loss function", "objective function", "we minimize", "we minimise", "we maximize",
    "gradient", "derivative", "theorem", "lemma", "proof", "we define", "denote",
    "probability distribution", "expectation", "optimization problem", "converges",
    "upper bound", "lower bound", "regularization", "regularisation",
)

_VALID_REPRESENTATIONS = {"latex", "extracted_text", "unavailable"}


def detect_math_content(context: AgentContext, *, min_equations: int = 3) -> tuple[bool, str]:
    """Decide whether the Math Agent should run, and say why.

    Returns ``(activate, reason)``. The reason is surfaced to the user so the
    decision is inspectable rather than mysterious.
    """
    index = context.index
    equations = index.by_element_type(ElementType.EQUATION)
    equation_count = len(equations)

    if equation_count >= min_equations:
        return True, f"{equation_count} equations detected (threshold {min_equations})"

    text = index.full_text().lower()
    keyword_hits = [k for k in _MATH_KEYWORDS if k in text]
    if equation_count >= 1 and len(keyword_hits) >= 2:
        return True, (
            f"{equation_count} equation(s) plus mathematical discussion "
            f"({', '.join(keyword_hits[:3])})"
        )
    if len(keyword_hits) >= 4:
        return True, f"substantial mathematical discussion ({', '.join(keyword_hits[:4])})"

    return False, (
        f"only {equation_count} equation(s) and {len(keyword_hits)} mathematical "
        f"keyword(s) detected; below the threshold of {min_equations} equations"
    )


class MathAgent(BaseAgent):
    name = "math"
    prompt = MATH
    context_limit = 14

    def __init__(self, provider, *, min_equations: int = 3) -> None:
        super().__init__(provider)
        self.min_equations = min_equations

    def should_run(self, context: AgentContext) -> tuple[bool, str | None]:
        activate, reason = detect_math_content(context, min_equations=self.min_equations)
        return activate, None if activate else reason

    def queries(self, context: AgentContext) -> list[RetrievalQuery]:
        return [
            # Every detected equation, regardless of similarity to a query.
            RetrievalQuery(
                text="equation formula definition",
                top_k=10,
                element_types=[ElementType.EQUATION],
            ),
            RetrievalQuery(text="loss function training objective we minimize", top_k=4),
            RetrievalQuery(text="notation symbols variables denote where", top_k=4),
            RetrievalQuery(text="derivation proof theorem lemma we show that", top_k=4),
            RetrievalQuery(text="dimensions shapes of the representations", top_k=3),
        ]

    def build_result(
        self, payload: dict[str, Any], context: AgentContext, results: list[RetrievalResult]
    ) -> tuple[list[Finding], dict[str, Any], list[str]]:
        data = require_dict(payload, self.name)
        findings = parse_findings(data.get("findings"), agent=self.name)

        equations: list[EquationExplanation] = []
        open_questions: list[str] = []

        for item in data.get("equations") or []:
            if not isinstance(item, dict):
                continue
            try:
                page = int(item.get("page"))
            except (TypeError, ValueError):
                continue
            if not context.index.has_page(page):
                open_questions.append(
                    f"An equation was reported on page {page}, which is outside this "
                    f"document (1-{context.index.page_count}); it was discarded."
                )
                continue

            source_text = str(item.get("source_text") or "").strip()
            representation = str(item.get("representation") or "extracted_text").lower()
            if representation not in _VALID_REPRESENTATIONS:
                representation = "extracted_text"

            # An equation with no recoverable source text is marked unavailable
            # rather than being reconstructed from the explanation.
            if not source_text:
                source_text = "(equation could not be extracted from the PDF)"
                representation = "unavailable"

            evidence = evidence_from_payload(item.get("evidence"), agent=self.name)
            for piece in evidence:
                validate_evidence(piece, context.index)

            if representation != "unavailable":
                verified = self._source_text_is_real(source_text, page, context)
                if not verified:
                    representation = "unavailable"
                    open_questions.append(
                        f"The equation text reported for page {page} could not be matched "
                        "against the extracted PDF text and is shown as unavailable."
                    )

            equations.append(
                EquationExplanation(
                    equation_id=str(item.get("equation_id") or "").strip() or None,
                    page=page,
                    source_text=source_text,
                    representation=representation,  # type: ignore[arg-type]
                    purpose=str(item.get("purpose") or ""),
                    variables=as_str_dict(item.get("variables")),
                    shapes=as_str_dict(item.get("shapes")),
                    derivation_stated_by_authors=str(
                        item.get("derivation_stated_by_authors") or ""
                    ),
                    agent_interpretation=str(item.get("agent_interpretation") or ""),
                    interpretation_differs=bool(item.get("interpretation_differs")),
                    intuition=str(item.get("intuition") or ""),
                    evidence=evidence,
                )
            )

        result = MathResult(
            equations=equations,
            loss_functions=as_str_list(data.get("loss_functions")),
            optimization_objective=str(data.get("optimization_objective") or ""),
            notation=as_str_dict(data.get("notation")),
            findings=findings,
        )
        return findings, result.model_dump(mode="json", exclude={"findings"}), open_questions

    @staticmethod
    def _source_text_is_real(source_text: str, page: int, context: AgentContext) -> bool:
        """Confirm the quoted equation exists on the cited page."""
        from ..validation.quotes import match_quote

        return match_quote(source_text, context.index.page_text(page), threshold=0.7).found
