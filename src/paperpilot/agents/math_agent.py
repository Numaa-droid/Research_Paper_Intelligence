"""Math Agent: activated only when the paper actually contains mathematics.

Activation is a deterministic decision made from the index, not a judgement call
delegated to a model: a paper with two inline symbols does not need a Math Agent,
and running one anyway would spend tokens to produce filler.
"""

from __future__ import annotations

import re
from typing import Any

from ..domain.enums import ElementType
from ..domain.models import EquationExplanation, Evidence, Finding, MathResult
from ..response import normalize_key
from ..response.reader import DocumentReader, QuoteRef, clean_text, parse_bool, parse_int
from ..retrieval.retriever import RetrievalQuery
from ..validation.evidence import validate_evidence
from ..validation.quotes import match_quote
from .base import AgentContext, BaseAgent
from .grounding import Grounder, parse_findings
from .prompts.registry import MATH

#: Words that indicate a real mathematical treatment rather than incidental notation.
_MATH_KEYWORDS = (
    "loss function", "objective function", "we minimize", "we minimise", "we maximize",
    "gradient", "derivative", "theorem", "lemma", "proof", "we define", "denote",
    "probability distribution", "expectation", "optimization problem", "converges",
    "upper bound", "lower bound", "regularization", "regularisation",
)

_VALID_REPRESENTATIONS = {"latex", "extracted_text", "unavailable"}
_EQUATION_HEADING = re.compile(r"^\s*(?:equation|eq\.?)\s*", re.IGNORECASE)
_NUMBERED_ID = re.compile(r"^equation-(\d+[a-z]?)$")


def equation_label(element_id: str | None, page: int) -> str:
    """``equation-2`` -> ``Equation (2)``; an unnumbered one is named by its page."""
    match = _NUMBERED_ID.match(element_id or "")
    return f"Equation ({match.group(1)})" if match else f"Unnumbered equation (p. {page})"


def _equation_number(label: str | None) -> float:
    match = re.search(r"\((\d+)", label or "")
    return float(match.group(1)) if match else float("inf")


def _verbatim_evidence(chunk, context: AgentContext) -> Evidence:
    evidence = Evidence(
        page=chunk.page,
        section=chunk.citation_section,
        source_type=chunk.element_type,
        element_id=chunk.element_id,
        chunk_id=chunk.chunk_id,
        quote=chunk.text,
        confidence=0.9,
    )
    validate_evidence(evidence, context.index)
    return evidence
_SYMBOL_COLUMNS: dict[str, tuple[str, ...]] = {
    "symbol": ("variable", "notation", "term"),
    "meaning": ("description", "definition"),
    "shape": ("dimension", "dimensions", "shapes", "size"),
}


def detect_math_content(context: AgentContext, *, min_equations: int = 3) -> tuple[bool, str]:
    """Decide whether the Math Agent should run, and say why.

    Returns ``(activate, reason)``. The reason is surfaced to the user so the
    decision is inspectable rather than mysterious.
    """
    index = context.index
    equation_count = len(
        {c.element_id or c.chunk_id for c in index.by_element_type(ElementType.EQUATION)}
    )

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
    anchor_element_types = (ElementType.EQUATION,)

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
        self, reader: DocumentReader, context: AgentContext, grounder: Grounder
    ) -> tuple[list[Finding], dict[str, Any], list[str]]:
        findings = parse_findings(reader, grounder, category="math")

        equations: list[EquationExplanation] = []
        open_questions: list[str] = []
        covered: set[str] = set()

        for sub in reader.subsections(reader.section("equations", "key equations")):
            fields = reader.fields(sub)
            source_item = reader.field(fields, "source", "sources", "citation")
            labels = [c.label for c in source_item.citations] if source_item else []
            refs = reader.quotes(sub)
            labels += [label for ref in refs for label in ref.labels if label not in labels]

            cited_chunks = [c for c in map(grounder.sources.chunk, labels) if c is not None]
            equation_chunk = next(
                (c for c in cited_chunks if c.element_type is ElementType.EQUATION), None
            )
            if equation_chunk is not None:
                # The cited source *is* the equation: its page, number and text
                # come from the index, so a model cannot mis-copy or mis-number it.
                covered.add(equation_chunk.chunk_id)
                equations.append(
                    self._from_index(equation_chunk, reader, sub, fields, grounder, refs, context)
                )
                continue

            page = parse_int(reader.field_text(fields, "page"))
            if page is None:
                cited = cited_chunks[0] if cited_chunks else None
                if cited is None:
                    open_questions.append(
                        f"'{sub.heading}' cited no source from the context and was discarded."
                    )
                    continue
                page = cited.page
            if not context.index.has_page(page):
                open_questions.append(
                    f"An equation was reported on page {page}, which is outside this "
                    f"document (1-{context.index.page_count}); it was discarded."
                )
                continue

            block = reader.code(sub)
            source_text = (
                block.text.strip()
                if block is not None
                else reader.field_text(fields, "source text", "equation", "formula")
            )
            representation = normalize_key(reader.field_text(fields, "representation")).replace(
                " ", "_"
            )
            if representation not in _VALID_REPRESENTATIONS:
                language = getattr(block, "language", None)
                representation = "latex" if language in ("latex", "tex", "math") else "extracted_text"

            # An equation with no recoverable source text is marked unavailable
            # rather than being reconstructed from the explanation.
            if not source_text:
                source_text = "(equation could not be extracted from the PDF)"
                representation = "unavailable"

            evidence = grounder.evidence_list(refs)
            if not evidence and representation != "unavailable":
                evidence = grounder.evidence_list(
                    [QuoteRef(text=source_text, citations=source_item.citations if source_item else [])]
                )
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

            variables, shapes = self._symbols(reader, sub)
            equations.append(
                EquationExplanation(
                    equation_id=clean_text(_EQUATION_HEADING.sub("", sub.heading)) or sub.heading,
                    page=page,
                    source_text=source_text,
                    representation=representation,  # type: ignore[arg-type]
                    purpose=reader.field_text(fields, "purpose"),
                    variables=variables,
                    shapes=shapes,
                    derivation_stated_by_authors=reader.field_text(
                        fields, "derivation stated by authors", "derivation"
                    ),
                    agent_interpretation=reader.field_text(
                        fields, "agent interpretation", "interpretation"
                    ),
                    interpretation_differs=bool(
                        parse_bool(reader.field_text(fields, "interpretation differs"))
                    ),
                    intuition=reader.field_text(fields, "intuition"),
                    evidence=evidence,
                )
            )

        for chunk in context.index.by_element_type(ElementType.EQUATION):
            if chunk.chunk_id in covered or any(
                e.page == chunk.page and match_quote(e.source_text, chunk.text).found
                for e in equations
                if e.representation != "unavailable"
            ):
                continue
            label = equation_label(chunk.element_id, chunk.page)
            open_questions.append(
                f"{label} was detected in the paper but not explained; it is listed verbatim."
            )
            equations.append(
                EquationExplanation(
                    equation_id=label,
                    page=chunk.page,
                    source_text=chunk.text,
                    representation="extracted_text",
                    evidence=[_verbatim_evidence(chunk, context)],
                )
            )
        equations.sort(key=lambda e: (e.page, _equation_number(e.equation_id)))

        result = MathResult(
            equations=equations,
            loss_functions=reader.items(reader.section("loss functions", "loss function")),
            optimization_objective=reader.text(
                reader.section("optimization objective", "optimisation objective", "objective")
            ),
            notation=reader.key_value_map(reader.section("notation")),
            findings=findings,
        )
        return findings, result.model_dump(mode="json", exclude={"findings"}), open_questions

    def _from_index(
        self, chunk, reader, section, fields, grounder, refs, context
    ) -> EquationExplanation:
        evidence = grounder.evidence_list(refs) or [_verbatim_evidence(chunk, context)]
        for piece in evidence:
            validate_evidence(piece, context.index)
        variables, shapes = self._symbols(reader, section)
        return EquationExplanation(
            equation_id=equation_label(chunk.element_id, chunk.page),
            page=chunk.page,
            source_text=chunk.text,
            representation="extracted_text",
            purpose=reader.field_text(fields, "purpose"),
            variables=variables,
            shapes=shapes,
            derivation_stated_by_authors=reader.field_text(
                fields, "derivation stated by authors", "derivation"
            ),
            agent_interpretation=reader.field_text(fields, "agent interpretation", "interpretation"),
            interpretation_differs=bool(
                parse_bool(reader.field_text(fields, "interpretation differs"))
            ),
            intuition=reader.field_text(fields, "intuition"),
            evidence=evidence,
        )

    @staticmethod
    def _symbols(reader: DocumentReader, section) -> tuple[dict[str, str], dict[str, str]]:
        variables: dict[str, str] = {}
        shapes: dict[str, str] = {}
        for table in reader.tables(section):
            for record in reader.records(table, _SYMBOL_COLUMNS):
                symbol = record["symbol"].value.strip() if record["symbol"] is not None else ""
                if not symbol:
                    continue
                if record["meaning"] is not None and record["meaning"].value:
                    variables[symbol] = clean_text(record["meaning"].value)
                if record["shape"] is not None and clean_text(record["shape"].value):
                    shapes[symbol] = clean_text(record["shape"].value)
        return variables, shapes

    @staticmethod
    def _source_text_is_real(source_text: str, page: int, context: AgentContext) -> bool:
        """Confirm the quoted equation exists on the cited page."""
        return match_quote(source_text, context.index.page_text(page), threshold=0.7).found
