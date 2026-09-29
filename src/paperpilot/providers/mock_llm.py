"""Deterministic, offline model provider.

This is not a stub that returns fixed strings. It reads the retrieved evidence it
is given and builds its answers out of verbatim sentences from that evidence,
which means:

* the whole pipeline — including quote validation, numeric checking and citation
  rendering — is exercised end to end with no credentials and no network;
* the integration test asserts on real grounding behaviour rather than on a
  canned fixture;
* it is honest about its own limits. It deliberately emits one ungrounded claim
  per run so that the Verification Agent has something to catch. A verifier that
  is never given a bad claim has not been tested;
* it answers in the same Markdown layout the prompts ask real models for, so the
  deterministic Markdown parser is exercised on every offline run.

It is a test double. It is never a good analysis of a paper, and the CLI and API
say so in their output when it is the active provider.
"""

from __future__ import annotations

import re
from typing import Any

from .llm import LLMProvider, LLMRequest, LLMResponse, TokenUsage

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")

#: A claim the mock always emits with no evidence, so the verifier has a real
#: failure to detect. Chosen to be obviously foreign to any input paper.
UNGROUNDED_CLAIM = (
    "The proposed method achieves 99.9% accuracy on ImageNet and has been deployed "
    "in production at three major technology companies."
)


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text or "") if p.strip()]
    return [p for p in parts if len(p) > 25]


def _score(text: str, keywords: set[str]) -> int:
    return len(_tokens(text) & keywords)


class MockLLMProvider(LLMProvider):
    """Evidence-grounded deterministic provider."""

    name = "mock"

    def __init__(self, model: str = "mock-deterministic-v1") -> None:
        self._model = model
        self.calls: list[LLMRequest] = []

    @property
    def model(self) -> str:
        return self._model

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        handler, render = {
            "summary": (self._summary, _render_summary),
            "methodology": (self._methodology, _render_methodology),
            "experiments": (self._experiments, _render_experiments),
            "math": (self._math, _render_math),
            "coverage": (self._coverage, _render_coverage),
            "verification": (self._verification, _render_verification),
            "planner": (self._planner, _render_planner),
        }.get(request.task, (self._fallback, _render_fallback))

        text = render(handler(request))
        return LLMResponse(
            text=text,
            model=self._model,
            provider=self.name,
            usage=TokenUsage(
                prompt_tokens=max(1, len(request.user) // 4),
                completion_tokens=max(1, len(text) // 4),
            ),
            latency_ms=0.0,
        )

    # --- grounding helpers ------------------------------------------------

    @staticmethod
    def _best_chunk(evidence: list[dict], keywords: str, **filters) -> dict | None:
        """Pick the evidence chunk that best matches ``keywords``."""
        wanted = _tokens(keywords)
        pool = [
            e
            for e in evidence
            if all(e.get(key) == value for key, value in filters.items())
        ]
        if not pool:
            pool = evidence
        if not pool:
            return None
        best = max(pool, key=lambda e: _score(e.get("text", ""), wanted))
        return best if _score(best.get("text", ""), wanted) > 0 else pool[0]

    @staticmethod
    def _quote_from(chunk: dict | None, keywords: str, *, max_chars: int = 320) -> str | None:
        """Take a verbatim sentence from ``chunk`` that matches ``keywords``."""
        if not chunk:
            return None
        text = chunk.get("text", "")
        wanted = _tokens(keywords)
        candidates = _sentences(text) or ([text] if text.strip() else [])
        if not candidates:
            return None
        best = max(candidates, key=lambda s: _score(s, wanted))
        return best[:max_chars].strip()

    def _evidence_object(self, chunk: dict | None, quote: str | None) -> list[dict]:
        if not chunk or not quote:
            return []
        return [{"source_id": chunk.get("source_id"), "quote": quote}]

    def _grounded_finding(
        self,
        evidence: list[dict],
        keywords: str,
        *,
        claim_prefix: str,
        category: str,
        impact: float = 0.6,
        **filters,
    ) -> dict | None:
        chunk = self._best_chunk(evidence, keywords, **filters)
        quote = self._quote_from(chunk, keywords)
        if not quote:
            return None
        return {
            "claim": f"{claim_prefix}: {quote}"[:400],
            "explanation": (
                f"Taken from {chunk.get('section') or 'the paper'} on page {chunk.get('page')}."
            ),
            "category": category,
            "status": "supported",
            "confidence": 0.8,
            "impact": impact,
            "evidence": self._evidence_object(chunk, quote),
        }

    @staticmethod
    def _ungrounded_finding(category: str) -> dict:
        return {
            "claim": UNGROUNDED_CLAIM,
            "explanation": (
                "Deliberately ungrounded claim emitted by the deterministic mock provider "
                "so that the Verification Agent has an unsupported claim to catch."
            ),
            "category": category,
            "status": "supported",  # asserted confidently on purpose
            "confidence": 0.9,
            "impact": 0.95,
            "evidence": [],
        }

    # --- task handlers ----------------------------------------------------

    def _summary(self, request: LLMRequest) -> dict[str, Any]:
        evidence = request.evidence
        findings = [
            f
            for f in (
                self._grounded_finding(
                    evidence,
                    "problem challenge requires difficult",
                    claim_prefix="The paper addresses this problem",
                    category="problem",
                    impact=0.7,
                ),
                self._grounded_finding(
                    evidence,
                    "we propose introduce contribution novel approach",
                    claim_prefix="The paper's stated contribution",
                    category="contribution",
                    impact=0.8,
                ),
                self._grounded_finding(
                    evidence,
                    "dataset evaluate benchmark data",
                    claim_prefix="Evaluation data",
                    category="dataset",
                    impact=0.6,
                ),
                self._grounded_finding(
                    evidence,
                    "results improves outperforms accuracy f1 score reduces",
                    claim_prefix="Reported result",
                    category="results",
                    impact=0.9,
                ),
                self._grounded_finding(
                    evidence,
                    "limitation does not we did not future work only",
                    claim_prefix="Stated limitation",
                    category="limitations",
                    impact=0.5,
                ),
            )
            if f
        ]
        findings.append(self._ungrounded_finding("results"))

        def field(keywords: str) -> str:
            chunk = self._best_chunk(evidence, keywords)
            return self._quote_from(chunk, keywords) or ""

        return {
            "research_problem": field("problem challenge requires difficult"),
            "motivation": field("motivation important matters because"),
            "main_contribution": field("we propose introduce contribution novel"),
            "dataset": field("dataset benchmark evaluate data"),
            "proposed_method": field("method architecture model approach pipeline"),
            "main_results": field("results improves outperforms score reduces"),
            "limitations": field("limitation does not future work only"),
            "assumptions": field("assume assumption requires given that"),
            "concise_summary": " ".join(
                filter(
                    None,
                    [
                        field("we propose introduce contribution novel"),
                        field("results improves outperforms score reduces"),
                    ],
                )
            )[:900],
            "unresolved_questions": [
                "Deterministic mock provider: this summary is assembled from retrieved "
                "sentences and is not a substitute for a real model.",
            ],
            "findings": findings,
        }

    def _methodology(self, request: LLMRequest) -> dict[str, Any]:
        evidence = request.evidence
        steps = []
        for order, keywords in enumerate(
            (
                "input takes given document question",
                "architecture encoder decoder layers module",
                "training trained loss optimizer objective",
                "inference test time output prediction",
            ),
            start=1,
        ):
            chunk = self._best_chunk(evidence, keywords)
            quote = self._quote_from(chunk, keywords)
            if not quote:
                continue
            steps.append(
                {
                    "order": order,
                    "name": keywords.split()[0].title(),
                    "description": quote,
                    "is_inferred": False,
                    "evidence": self._evidence_object(chunk, quote),
                }
            )

        findings = [
            f
            for f in (
                self._grounded_finding(
                    evidence,
                    "architecture encoder module component layer",
                    claim_prefix="Architecture",
                    category="methodology",
                    impact=0.7,
                ),
                self._grounded_finding(
                    evidence,
                    "training trained loss optimizer learning rate",
                    claim_prefix="Training procedure",
                    category="methodology",
                    impact=0.7,
                ),
            )
            if f
        ]
        findings.append(
            {
                "claim": (
                    "The implementation likely uses mixed-precision training on 8 GPUs, "
                    "though the paper does not state this."
                ),
                "explanation": "Interpretation by the mock provider, not stated by the authors.",
                "category": "methodology",
                "status": "inferred",
                "confidence": 0.35,
                "impact": 0.4,
                "evidence": [],
            }
        )

        def field(keywords: str) -> str:
            return self._quote_from(self._best_chunk(evidence, keywords), keywords) or ""

        return {
            "inputs": field("input takes given document question"),
            "outputs": field("output produces returns prediction span"),
            "architecture": field("architecture encoder decoder layers module"),
            "training_procedure": field("training trained loss optimizer objective"),
            "inference_procedure": field("inference test time decoding"),
            "design_decisions": [d for d in (field("we set we use we choose"),) if d],
            "comparison_with_prior_work": field("prior work unlike previous compared"),
            "pipeline_steps": steps,
            "pseudocode": None,
            "findings": findings,
        }

    def _experiments(self, request: LLMRequest) -> dict[str, Any]:
        evidence = request.evidence
        tables = [e for e in evidence if e.get("element_type") == "table"]
        # Prefer the table *body* over its caption: the caption chunk shares the
        # element id but contains no numbers to read.
        table = (
            max(tables, key=lambda t: len(re.findall(r"\d+\.\d+", t.get("text", ""))))
            if tables
            else None
        )

        reported = []
        if table:
            # Read real numbers out of the real table text.
            for line in table.get("text", "").split("\n"):
                numbers = re.findall(r"\b\d+\.\d+\b", line)
                label = line.split()[0] if line.split() else ""
                if numbers and label and not label[0].isdigit():
                    reported.append(
                        {
                            "metric": "reported value",
                            "value": numbers[0],
                            "system": label,
                            "dataset": None,
                            "split": None,
                            "is_best": False,
                            "evidence": self._evidence_object(
                                table, line.strip()[:320] or table.get("text", "")[:320]
                            ),
                        }
                    )

        findings = [
            f
            for f in (
                self._grounded_finding(
                    evidence,
                    "dataset benchmark evaluate splits",
                    claim_prefix="Experimental setup",
                    category="experiments",
                    impact=0.7,
                ),
                self._grounded_finding(
                    evidence,
                    "baseline compared against dense previous",
                    claim_prefix="Baselines",
                    category="experiments",
                    impact=0.7,
                ),
                self._grounded_finding(
                    evidence,
                    "ablation removing without component reduces",
                    claim_prefix="Ablation",
                    category="experiments",
                    impact=0.8,
                ),
            )
            if f
        ]
        if table:
            quote = self._quote_from(table, "results comparison model") or table.get("text", "")
            findings.append(
                {
                    "claim": f"Main results table: {quote}"[:400],
                    "explanation": f"Copied from {table.get('element_id') or 'the results table'}.",
                    "category": "experiments",
                    "status": "supported",
                    "confidence": 0.85,
                    "impact": 0.9,
                    "evidence": self._evidence_object(table, quote),
                }
            )

        return {
            "datasets": [],
            "splits": self._quote_from(
                self._best_chunk(evidence, "train validation test split"),
                "train validation test split",
            )
            or "",
            "baselines": [],
            "metrics": [],
            "reported_results": reported,
            "ablations": [
                a
                for a in (
                    self._quote_from(
                        self._best_chunk(evidence, "ablation removing without"),
                        "ablation removing without",
                    ),
                )
                if a
            ],
            "hyperparameters": {},
            "error_analysis": "",
            "statistical_significance": self._quote_from(
                self._best_chunk(evidence, "significance confidence interval seed"),
                "significance confidence interval seed",
            )
            or "",
            "conclusions_supported": "partially_supported",
            "conclusions_assessment": (
                "Assessed by the deterministic mock provider from retrieved text only."
            ),
            "findings": findings,
        }

    def _math(self, request: LLMRequest) -> dict[str, Any]:
        evidence = request.evidence
        equations = []
        for chunk in [e for e in evidence if e.get("element_type") == "equation"][:6]:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            equations.append(
                {
                    "equation_id": chunk.get("element_id"),
                    "page": chunk.get("page"),
                    "source_text": text,
                    "representation": "extracted_text",
                    "purpose": "Equation extracted from the paper.",
                    "variables": {},
                    "shapes": {},
                    "derivation_stated_by_authors": "",
                    "agent_interpretation": (
                        "The deterministic mock provider does not interpret mathematics; "
                        "the equation is preserved verbatim."
                    ),
                    "interpretation_differs": True,
                    "intuition": "",
                    "evidence": self._evidence_object(chunk, text[:320]),
                }
            )

        findings = [
            f
            for f in (
                self._grounded_finding(
                    evidence,
                    "loss objective minimize maximize",
                    claim_prefix="Optimization objective",
                    category="math",
                    impact=0.7,
                ),
            )
            if f
        ]
        return {
            "equations": equations,
            "loss_functions": [],
            "optimization_objective": self._quote_from(
                self._best_chunk(evidence, "loss objective minimize"),
                "loss objective minimize",
            )
            or "",
            "notation": {},
            "findings": findings,
        }

    def _verification(self, request: LLMRequest) -> dict[str, Any]:
        """Lexically ground the claim, then defer to the deterministic checks."""
        claim = str(request.context.get("claim", ""))
        evidence = request.evidence
        numeric_checks = request.context.get("numeric_checks") or []
        numbers_missing = any(not c.get("found_in_evidence") for c in numeric_checks)

        claim_tokens = _tokens(claim) - {
            "the", "a", "an", "of", "and", "is", "are", "in", "on", "to", "for", "that",
            "this", "with", "by", "from", "it", "as", "at", "be", "has", "have",
        }
        best = None
        best_overlap = 0.0
        for item in evidence:
            overlap = len(claim_tokens & _tokens(item.get("text", "")))
            ratio = overlap / max(len(claim_tokens), 1)
            if ratio > best_overlap:
                best_overlap = ratio
                best = item

        if best is None or best_overlap < 0.25:
            return {
                "status": "unsupported",
                "confidence": 0.15,
                "explanation": (
                    "No retrieved passage shares enough content with the claim to support "
                    f"it (best lexical overlap {best_overlap:.0%})."
                ),
                "supporting_evidence": [],
                "contradicting_evidence": [],
            }

        quote = self._quote_from(best, claim) or (best.get("text") or "")[:320]
        if numbers_missing:
            return {
                "status": "partially_supported",
                "confidence": 0.5,
                "explanation": (
                    "The retrieved evidence is on topic, but at least one number in the "
                    "claim does not appear in it."
                ),
                "supporting_evidence": self._evidence_object(best, quote),
                "contradicting_evidence": [],
            }

        status = "supported" if best_overlap >= 0.5 else "partially_supported"
        return {
            "status": status,
            "confidence": round(min(0.95, 0.4 + best_overlap), 4),
            "explanation": (
                f"The claim overlaps {best_overlap:.0%} of its content words with a passage "
                f"on page {best.get('page')}"
                + (f" in {best.get('section')}" if best.get("section") else "")
                + "."
            ),
            "supporting_evidence": self._evidence_object(best, quote),
            "contradicting_evidence": [],
        }

    def _planner(self, request: LLMRequest) -> dict[str, Any]:
        structure = request.context.get("structure", {})
        equations = int(structure.get("equation_count", 0))
        tables = int(structure.get("table_count", 0))
        sections = [str(s) for s in structure.get("sections", [])]

        def has(*needles: str) -> bool:
            return any(n in s.lower() for s in sections for n in needles)

        return {
            "run_summary": True,
            "run_methodology": has("method", "approach", "model", "architecture") or True,
            "run_experiments": has("experiment", "result", "evaluation") or tables > 0,
            "run_math": equations >= int(structure.get("math_min_equations", 3)),
            "run_table_analysis": tables > 0,
            "priority_sections": [s for s in sections if not s.lower().startswith("reference")][:6],
            "verification_threshold": 0.85,
            "max_retries": 1,
            "rationale": {
                "run_math": f"{equations} equation(s) detected in the index",
                "run_experiments": f"{tables} table(s) detected in the index",
                "provider": "plan produced by the deterministic mock provider",
            },
        }

    def _coverage(self, request: LLMRequest) -> dict[str, Any]:
        """One grounded finding per source: its first sentence, quoted verbatim."""
        findings = []
        for chunk in request.evidence:
            quote = next(iter(_sentences(chunk.get("text", ""))), "").strip()[:320]
            if len(quote) < 20:
                continue
            findings.append(
                {
                    "claim": f"The paper states: {quote}"[:400],
                    "explanation": f"Read by the coverage sweep from page {chunk.get('page')}.",
                    "category": "coverage",
                    "status": "supported",
                    "confidence": 0.7,
                    "impact": 0.4,
                    "evidence": self._evidence_object(chunk, quote),
                }
            )
        return {"findings": findings}

    def _fallback(self, request: LLMRequest) -> dict[str, Any]:
        return {"findings": [], "note": f"mock provider has no handler for task '{request.task}'"}


# --- Markdown rendering ------------------------------------------------------
# The mock writes exactly the layout the prompts in agents/prompts/registry.py
# request, the way a compliant model would.


def _one_line(text: object) -> str:
    return " ".join(str(text or "").split())


def _prose(text: object) -> str:
    value = str(text or "").strip()
    return value.replace("\n\n", "\n") if value else "Not stated."


def _cell(text: object) -> str:
    return _one_line(text).replace("|", "\\|")


def _yes_no(value: object) -> str:
    return "yes" if value else "no"


def _quotes(evidence: list[dict]) -> str:
    lines = []
    for item in evidence:
        quote = str(item.get("quote") or "").strip().replace("\n\n", "\n")
        if quote:
            label = f"[{item['source_id']}] " if item.get("source_id") else ""
            lines.append("> " + label + quote.replace("\n", "\n> "))
    return "\n\n".join(lines)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {_one_line(i)}" for i in items if str(i).strip()) or "Not stated."


def _findings(findings: list[dict]) -> str:
    parts = ["## Findings"]
    for number, finding in enumerate(findings, start=1):
        block = [
            f"### Finding {number}",
            f"- Claim: {_one_line(finding['claim'])}",
            f"- Status: {finding.get('status', 'unknown')}",
            f"- Category: {finding.get('category') or 'general'}",
            f"- Confidence: {finding.get('confidence', 0.5)}",
            f"- Impact: {finding.get('impact', 0.5)}",
            f"- Explanation: {_one_line(finding.get('explanation'))}",
        ]
        quotes = _quotes(finding.get("evidence") or [])
        parts.append("\n".join(block) + (f"\n\n{quotes}" if quotes else ""))
    return "\n\n".join(parts)


def _sections(pairs: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"## {heading}\n{body}" for heading, body in pairs)


def _render_summary(data: dict) -> str:
    return (
        _sections(
            [
                ("Research Problem", _prose(data["research_problem"])),
                ("Motivation", _prose(data["motivation"])),
                ("Main Contribution", _prose(data["main_contribution"])),
                ("Dataset", _prose(data["dataset"])),
                ("Proposed Method", _prose(data["proposed_method"])),
                ("Main Results", _prose(data["main_results"])),
                ("Limitations", _prose(data["limitations"])),
                ("Assumptions", _prose(data["assumptions"])),
                ("Concise Summary", _prose(data["concise_summary"])),
                ("Unresolved Questions", _bullets(data["unresolved_questions"])),
            ]
        )
        + "\n\n"
        + _findings(data["findings"])
    )


def _render_methodology(data: dict) -> str:
    steps = ["## Pipeline Steps"]
    for step in data["pipeline_steps"]:
        quotes = _quotes(step["evidence"])
        steps.append(
            f"### Step {step['order']}: {_one_line(step['name'])}\n"
            f"- Inferred: {_yes_no(step['is_inferred'])}\n\n"
            f"{_prose(step['description'])}" + (f"\n\n{quotes}" if quotes else "")
        )
    pseudocode = data.get("pseudocode")
    return "\n\n".join(
        [
            _sections(
                [
                    ("Inputs", _prose(data["inputs"])),
                    ("Outputs", _prose(data["outputs"])),
                    ("Architecture", _prose(data["architecture"])),
                    ("Training Procedure", _prose(data["training_procedure"])),
                    ("Inference Procedure", _prose(data["inference_procedure"])),
                    ("Design Decisions", _bullets(data["design_decisions"])),
                    ("Comparison With Prior Work", _prose(data["comparison_with_prior_work"])),
                ]
            ),
            "\n\n".join(steps),
            "## Pseudocode\n" + (f"```text\n{pseudocode}\n```" if pseudocode else "Not stated."),
            _findings(data["findings"]),
        ]
    )


def _render_experiments(data: dict) -> str:
    rows = [
        "| System | Dataset | Split | Metric | Value | Best | Source | Quote |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in data["reported_results"]:
        evidence = (item.get("evidence") or [{}])[0]
        rows.append(
            "| "
            + " | ".join(
                [
                    _cell(item.get("system")),
                    _cell(item.get("dataset")),
                    _cell(item.get("split")),
                    _cell(item["metric"]),
                    _cell(item["value"]),
                    _yes_no(item.get("is_best")),
                    f"[{evidence['source_id']}]" if evidence.get("source_id") else "",
                    _cell(evidence.get("quote")),
                ]
            )
            + " |"
        )
    hyper = ["| Name | Value |", "| --- | --- |"] + [
        f"| {_cell(k)} | {_cell(v)} |" for k, v in data["hyperparameters"].items()
    ]
    return (
        _sections(
            [
                ("Datasets", _bullets(data["datasets"])),
                ("Splits", _prose(data["splits"])),
                ("Baselines", _bullets(data["baselines"])),
                ("Metrics", _bullets(data["metrics"])),
                ("Reported Results", "\n".join(rows)),
                ("Ablations", _bullets(data["ablations"])),
                ("Hyperparameters", "\n".join(hyper)),
                ("Error Analysis", _prose(data["error_analysis"])),
                ("Statistical Significance", _prose(data["statistical_significance"])),
                (
                    "Conclusions",
                    f"- Supported: {data['conclusions_supported']}\n\n"
                    + _prose(data["conclusions_assessment"]),
                ),
            ]
        )
        + "\n\n"
        + _findings(data["findings"])
    )


def _render_math(data: dict) -> str:
    equations = ["## Equations"]
    for equation in data["equations"]:
        evidence = equation.get("evidence") or []
        source = evidence[0].get("source_id") if evidence else None
        lines = [f"### Equation {equation.get('equation_id') or ''}".rstrip()]
        if source:
            lines.append(f"- Source: [{source}]")
        lines += [
            f"- Page: {equation['page']}",
            f"- Representation: {equation['representation']}",
            f"- Purpose: {_one_line(equation['purpose'])}",
            f"- Derivation stated by authors: {_one_line(equation['derivation_stated_by_authors']) or 'Not stated.'}",
            f"- Agent interpretation: {_one_line(equation['agent_interpretation'])}",
            f"- Interpretation differs: {_yes_no(equation['interpretation_differs'])}",
            f"- Intuition: {_one_line(equation['intuition']) or 'Not stated.'}",
            "",
            f"```text\n{equation['source_text']}\n```",
        ]
        quotes = _quotes(evidence)
        equations.append("\n".join(lines) + (f"\n\n{quotes}" if quotes else ""))
    notation = ["| Symbol | Meaning |", "| --- | --- |"] + [
        f"| {_cell(k)} | {_cell(v)} |" for k, v in data["notation"].items()
    ]
    return "\n\n".join(
        [
            "\n\n".join(equations),
            _sections(
                [
                    ("Loss Functions", _bullets(data["loss_functions"])),
                    ("Optimization Objective", _prose(data["optimization_objective"])),
                    ("Notation", "\n".join(notation)),
                ]
            ),
            _findings(data["findings"]),
        ]
    )


def _render_verification(data: dict) -> str:
    return _sections(
        [
            ("Verdict", f"- Status: {data['status']}\n- Confidence: {data['confidence']}"),
            ("Explanation", _prose(data["explanation"])),
            ("Supporting Evidence", _quotes(data["supporting_evidence"]) or "None."),
            ("Contradicting Evidence", _quotes(data["contradicting_evidence"]) or "None."),
        ]
    )


def _render_planner(data: dict) -> str:
    rows = ["| Agent | Run | Reason |", "| --- | --- | --- |"]
    for agent in ("summary", "methodology", "experiments", "math", "table_analysis"):
        reason = data["rationale"].get(f"run_{agent}", "")
        rows.append(f"| {agent} | {_yes_no(data[f'run_{agent}'])} | {_cell(reason)} |")
    return _sections(
        [
            ("Agents", "\n".join(rows)),
            ("Priority Sections", _bullets(data["priority_sections"])),
        ]
    )


def _render_coverage(data: dict) -> str:
    table = "## Settings and Values\n| Name | Value | Source | Quote |\n| --- | --- | --- | --- |"
    return table + "\n\n" + _findings(data.get("findings") or [])


def _render_fallback(data: dict) -> str:
    return f"## Note\n{data.get('note', '')}\n\n## Findings\nNot stated."
