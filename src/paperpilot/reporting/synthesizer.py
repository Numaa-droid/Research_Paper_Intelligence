"""Report synthesis.

The report is assembled deterministically from validated structures — there is no
model call here at all. That is the point: the final artefact cannot introduce a
claim, a number or a citation that did not survive verification, because the
synthesizer has nothing to work from except validated objects.

Unsupported, contradicted and inferred findings are given their own section
rather than being dropped, so what the system could *not* establish is as visible
as what it could.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime

from ..discourse.attribution import (
    AttributionMap,
    attribution_map,
    describe_sources,
    evidence_attribution,
    text_attribution,
)
from ..discourse.protocol import review_protocol
from ..discourse.structure import heading_numbering_issues, heading_sequence
from ..domain.enums import AgentName, ElementType, FindingStatus
from ..domain.models import (
    AgentResult,
    AgentUsage,
    AnalysisPlan,
    Evidence,
    Finding,
    PaperMetadata,
    ProgressEvent,
    VerificationResult,
)
from ..retrieval.index import PaperIndex
from ..validation.terms import PaperVocabulary
from .citations import citation_key, format_citations, format_quote

_STATUS_LABEL = {
    FindingStatus.SUPPORTED: "Supported",
    FindingStatus.PARTIALLY_SUPPORTED: "Partially supported",
    FindingStatus.UNSUPPORTED: "Unsupported",
    FindingStatus.CONTRADICTED: "Contradicted",
    FindingStatus.INFERRED: "Inferred (not stated by the authors)",
    FindingStatus.UNKNOWN: "Unknown",
}

_WEAK = (FindingStatus.UNSUPPORTED, FindingStatus.CONTRADICTED, FindingStatus.UNKNOWN)


def _esc(text: str | None) -> str:
    """Collapse whitespace and neutralise stray Markdown pipes for tables."""
    if not text:
        return ""
    return " ".join(str(text).split()).replace("|", "\\|")


#: What a model writes into a field when the paper does not state the value.
#: It is prose about an absence, so it is emptied out rather than carried
#: forward: left as-is it reads as content, and the reproducibility checklist
#: ticks "found in the paper" for a sentence saying it was not.
_NOT_STATED = re.compile(
    r"^\W*(?:(?:not|none)\b[\w\s]{0,40}?\b(?:stated|specified|described|reported|mentioned"
    r"|given|provided|available|applicable|found|discussed|detailed|listed)"
    r"|not\s+applicable|none|n/?a|unknown|unspecified|unclear|no\s+\w+\s+(?:found|reported))\W*$",
    re.IGNORECASE,
)


def _stated(value):
    """Strip "the paper does not say" placeholders out of an agent field.

    Returns the same shape it was given — string, list or mapping — so callers
    can keep testing it for truth, but with disclaimers removed.
    """
    if isinstance(value, str):
        text = value.strip()
        return "" if _NOT_STATED.match(text) else text
    if isinstance(value, (list, tuple)):
        return [item for item in value if not isinstance(item, str) or _stated(item)]
    return value


def _attribution_tag(attribution: str | None, sources) -> str:
    """"(reported from cited work [20])" for a statement that is another work's."""
    if attribution == "cited":
        return f" _(reported from cited work {describe_sources(sources)})_"
    if attribution == "mixed":
        return f" _(partly reported from cited work {describe_sources(sources)})_"
    return ""


def _bullet(finding: Finding) -> str:
    citation = format_citations(finding.evidence)
    tag = _attribution_tag(finding.attribution, finding.cited_sources)
    line = f"- **{_esc(finding.claim)}** {citation}{tag}"
    if finding.explanation:
        line += f"\n  {_esc(finding.explanation)}"
    return line


def _combined_attribution(pairs: list[tuple[str, tuple[str, ...]]]) -> tuple[str, list[str]]:
    """One attribution for several pieces of evidence (a step, a result row)."""
    known = [(kind, sources) for kind, sources in pairs if kind != "unknown"]
    sources: list[str] = []
    for _, found in known:
        sources.extend(x for x in found if x not in sources)
    if not known:
        return "unknown", []
    if all(kind == "cited" for kind, _ in known):
        return "cited", sources
    if any(kind in ("cited", "mixed") for kind, _ in known):
        return "mixed", sources
    return "own", []


#: Section titles that change meaning with the kind of paper. A review's
#: "methodology" is its search and synthesis, its "dataset" the corpus searched.
_KIND_TITLES = {
    "review": {
        "methodology": "## 4. Methodology — how the review was conducted",
        "setup": "## 6. Dataset and experimental setup — the corpus the review searched",
        "reproducibility": "## 9. Reproducibility checklist — review protocol",
    },
}
_DEFAULT_TITLES = {
    "methodology": "## 4. Methodology",
    "setup": "## 6. Dataset and experimental setup",
    "reproducibility": "## 9. Reproducibility checklist",
}


class ReportSynthesizer:
    """Renders the final Markdown report."""

    def __init__(self, *, include_appendix: bool = True, max_appendix_items: int = 60) -> None:
        self.include_appendix = include_appendix
        self.max_appendix_items = max_appendix_items

    # --- public API -------------------------------------------------------

    def synthesize(
        self,
        *,
        metadata: PaperMetadata,
        index: PaperIndex,
        plan: AnalysisPlan | None,
        agent_results: dict[str, AgentResult],
        findings: list[Finding],
        verification_results: list[VerificationResult],
        events: list[ProgressEvent] | None = None,
        usage: list[AgentUsage] | None = None,
        retry_count: int = 0,
    ) -> str:
        self._vocabulary = PaperVocabulary(index)
        self._index = index
        self._amap: AttributionMap = attribution_map(index)
        self._kind = plan.paper_kind if plan else "empirical"
        # Findings folded into another as duplicates stay in their agent's
        # result but are no longer part of the analysis; each live finding is
        # printed once across the narrative sections.
        self._live = {id(f) for f in findings}
        self._rendered: set[int] = set()
        sections: list[str] = [
            self._header(metadata),
            self._executive_summary(agent_results, findings),
            self._problem_and_motivation(agent_results),
            self._contributions(agent_results, findings),
            self._methodology(agent_results),
            self._mathematics(agent_results),
            self._dataset_and_setup(agent_results),
            self._results(agent_results, findings),
            self._limitations(agent_results, findings),
            self._reproducibility(agent_results, metadata),
            self._rest_of_paper(agent_results),
            self._verified_findings(findings),
            self._weak_findings(findings, verification_results),
            self._open_questions(agent_results, findings),
            self._how_this_was_produced(plan, agent_results, verification_results, retry_count, usage),
        ]
        if self.include_appendix:
            sections.append(self._evidence_appendix(findings, verification_results))
        sections.append(self._footer(metadata, index))
        return "\n\n".join(s for s in sections if s and s.strip()) + "\n"

    # --- sections ---------------------------------------------------------

    def _header(self, metadata: PaperMetadata) -> str:
        lines = ["# PaperMind Report", ""]
        lines.append(f"**Paper:** {_esc(metadata.title) or '_title not detected_'}")
        if metadata.authors:
            lines.append(f"**Authors:** {_esc(', '.join(metadata.authors))}")
        if metadata.year:
            lines.append(f"**Year:** {metadata.year}")
        if metadata.doi:
            lines.append(f"**DOI:** {metadata.doi}")
        lines.append(f"**Source file:** `{_esc(metadata.filename) or 'unknown'}`")
        lines.append(f"**Pages:** {metadata.page_count} &nbsp;|&nbsp; **Indexed chunks:** {metadata.chunk_count}")
        lines.append(
            f"**Detected elements:** {metadata.table_count} tables, "
            f"{metadata.figure_count} figures, {metadata.equation_count} equations"
        )
        if metadata.ocr_page_count:
            lines.append(
                f"**OCR:** {metadata.ocr_page_count} page(s) were read by OCR; quotes from "
                "those pages may be approximate."
            )
        lines.append("")
        lines.append(
            "> Page numbers are **1-based PDF page indices**, not printed page labels. "
            "Every citation below was checked against the indexed text of the paper; "
            "statements that could not be grounded are listed explicitly rather than removed."
        )
        return "\n".join(lines)

    def _executive_summary(
        self, agent_results: dict[str, AgentResult], findings: list[Finding]
    ) -> str:
        summary = self._payload(agent_results, AgentName.SUMMARY)
        body = self._prose(summary.get("concise_summary")) if summary else ""

        supported = sum(1 for f in findings if f.status is FindingStatus.SUPPORTED)
        partial = sum(1 for f in findings if f.status is FindingStatus.PARTIALLY_SUPPORTED)
        weak = sum(1 for f in findings if f.status in _WEAK)
        inferred = sum(1 for f in findings if f.status is FindingStatus.INFERRED)

        lines = ["## 1. Executive summary", ""]
        lines.append(body or "_The summary agent did not return a usable summary._")
        lines.append("")
        lines.append(
            f"**Grounding at a glance:** {len(findings)} findings — "
            f"{supported} supported, {partial} partially supported, "
            f"{inferred} inferred, {weak} unsupported or contradicted."
        )
        return "\n".join(lines)

    def _problem_and_motivation(self, agent_results: dict[str, AgentResult]) -> str:
        summary = self._payload(agent_results, AgentName.SUMMARY)
        if not summary:
            return ""
        lines = ["## 2. Research problem and motivation", ""]
        lines.append(f"**Problem.** {self._prose(summary.get('research_problem')) or '_not established_'}")
        lines.append("")
        lines.append(f"**Motivation.** {self._prose(summary.get('motivation')) or '_not established_'}")
        return "\n".join(lines)

    def _contributions(
        self, agent_results: dict[str, AgentResult], findings: list[Finding]
    ) -> str:
        summary = self._payload(agent_results, AgentName.SUMMARY)
        lines = ["## 3. Contributions", ""]
        if summary and summary.get("main_contribution"):
            lines.append(self._prose(summary["main_contribution"]))
            lines.append("")
        candidates = [
            f for f in findings if (f.category or "").lower() in ("contribution", "problem")
        ]
        # A contribution is something the paper does. A statement whose
        # evidence only reports a cited work cannot be one, however it is worded.
        contribution_findings = [f for f in candidates if f.attribution != "cited"]
        moved = len(candidates) - len(contribution_findings)
        if contribution_findings:
            lines.extend(self._bullets(contribution_findings))
        elif not (summary and summary.get("main_contribution")):
            lines.append("_No contribution claims could be grounded in the paper._")
        if moved:
            lines.append("")
            lines.append(
                f"_{moved} statement(s) offered as contributions only report what cited works "
                "found, so they are not listed as this paper's contributions (see §11)._"
            )
        return "\n".join(lines)

    def _methodology(self, agent_results: dict[str, AgentResult]) -> str:
        result = agent_results.get(AgentName.METHODOLOGY.value)
        lines = [self._title("methodology"), ""]
        if result is None or result.status != "ok":
            lines.append(self._not_run(result, "methodology"))
            return "\n".join(lines)

        payload = result.payload
        for label, key in (
            ("Inputs", "inputs"),
            ("Outputs", "outputs"),
            ("Architecture", "architecture"),
            ("Training", "training_procedure"),
            ("Inference", "inference_procedure"),
            ("Relation to prior work", "comparison_with_prior_work"),
        ):
            value = self._prose(payload.get(key))
            if value:
                lines.append(f"**{label}.** {value}")
                lines.append("")

        steps = payload.get("pipeline_steps") or []
        if steps:
            lines.append("### Execution flow")
            lines.append("")
            rendered_steps = []
            attributions = []
            for step in steps:
                evidence = [Evidence(**e) for e in step.get("evidence", [])]
                kind, sources = self._evidence_attribution(evidence)
                if evidence:
                    attributions.append(kind)
                marker = " _(inferred, not stated by the authors)_" if step.get("is_inferred") else ""
                tag = (
                    f" _(procedure of cited work {describe_sources(sources)}, not this paper's)_"
                    if kind == "cited"
                    else _attribution_tag(kind, sources)
                )
                rendered_steps.append(
                    f"{step.get('order', 0)}. **{_esc(step.get('name'))}** — "
                    f"{_esc(step.get('description'))}{marker} {format_citations(evidence)}{tag}"
                )
            cited_steps = sum(1 for kind in attributions if kind == "cited")
            if attributions and cited_steps == len(attributions):
                lines.append(
                    "> ⚠️ **Every step below is evidenced only by passages in which the paper "
                    "reports another work's procedure.** They describe studies the paper cites, "
                    "not the paper's own method."
                )
                lines.append("")
            elif cited_steps:
                lines.append(
                    f"> ⚠️ {cited_steps} of {len(attributions)} step(s) are evidenced only by "
                    "passages reporting a cited work's procedure; they are marked below."
                )
                lines.append("")
            lines.extend(rendered_steps)
            lines.append("")

        decisions = payload.get("design_decisions") or []
        if decisions:
            lines.append("### Design decisions")
            lines.append("")
            lines.extend(f"- {_esc(d)}" for d in decisions)
            lines.append("")

        if payload.get("pseudocode"):
            lines.append("### Pseudocode")
            lines.append("")
            lines.append("```text")
            lines.append(str(payload["pseudocode"]))
            lines.append("```")

        method_findings = self._unrendered(result.findings)
        if method_findings:
            lines.append("### Methodology findings")
            lines.append("")
            lines.extend(self._bullets(method_findings))
        return "\n".join(lines)

    def _mathematics(self, agent_results: dict[str, AgentResult]) -> str:
        result = agent_results.get(AgentName.MATH.value)
        lines = ["## 5. Mathematical explanation", ""]
        if result is None or result.status == "skipped":
            reason = result.skip_reason if result else "the math agent did not run"
            lines.append(
                f"_The Math Agent was **not activated** for this paper: {_esc(reason)}._"
            )
            return "\n".join(lines)
        if result.status != "ok":
            lines.append(self._not_run(result, "math"))
            return "\n".join(lines)

        payload = result.payload
        if payload.get("optimization_objective"):
            lines.append(f"**Optimisation objective.** {self._prose(payload['optimization_objective'])}")
            lines.append("")

        equations = payload.get("equations") or []
        if not equations:
            lines.append("_No equations could be extracted legibly from this paper._")
        for equation in equations:
            identifier = equation.get("equation_id") or f"page {equation.get('page')}"
            page_note = "" if "(p." in identifier else f" (p. {equation.get('page')})"
            lines.append(f"### {_esc(identifier)}{page_note}")
            lines.append("")
            if equation.get("representation") == "unavailable":
                lines.append(
                    "> **Equation unavailable.** This equation did not survive PDF "
                    "extraction legibly and has deliberately not been reconstructed."
                )
            else:
                lines.append("```text")
                lines.append(str(equation.get("source_text", "")))
                lines.append("```")
                lines.append(
                    f"_Preserved as: {equation.get('representation', 'extracted_text')}_"
                )
            lines.append("")
            if equation.get("purpose"):
                lines.append(f"**Purpose.** {_esc(equation['purpose'])}")
            variables = equation.get("variables") or {}
            if variables:
                lines.append("")
                lines.append("| Symbol | Meaning |")
                lines.append("| --- | --- |")
                for symbol, meaning in variables.items():
                    lines.append(f"| `{_esc(symbol)}` | {_esc(meaning)} |")
            shapes = equation.get("shapes") or {}
            if shapes:
                lines.append("")
                lines.append(
                    "**Shapes.** "
                    + ", ".join(f"`{_esc(k)}`: {_esc(v)}" for k, v in shapes.items())
                )
            if equation.get("derivation_stated_by_authors"):
                lines.append("")
                lines.append(
                    f"**Derivation stated by the authors.** "
                    f"{_esc(equation['derivation_stated_by_authors'])}"
                )
            if equation.get("agent_interpretation"):
                lines.append("")
                marker = (
                    " ⚠️ _This is the system's reading and differs from, or goes beyond, "
                    "what the authors state._"
                    if equation.get("interpretation_differs")
                    else ""
                )
                lines.append(
                    f"**System interpretation.** {_esc(equation['agent_interpretation'])}{marker}"
                )
            evidence = [Evidence(**e) for e in equation.get("evidence", [])]
            if evidence:
                lines.append("")
                lines.append(f"Evidence: {format_citations(evidence)}")
            lines.append("")
        return "\n".join(lines)

    def _dataset_and_setup(self, agent_results: dict[str, AgentResult]) -> str:
        result = agent_results.get(AgentName.EXPERIMENTS.value)
        lines = [self._title("setup"), ""]
        if result is None or result.status != "ok":
            lines.append(self._not_run(result, "experiments"))
            return "\n".join(lines)

        payload = result.payload
        for label, items in (
            ("Datasets", _stated(payload.get("datasets") or [])),
            ("Splits", [s] if (s := _stated(payload.get("splits"))) else []),
            ("Baselines", _stated(payload.get("baselines") or [])),
            ("Metrics", _stated(payload.get("metrics") or [])),
        ):
            own, cited = self._split_by_attribution(items)
            text = _esc(", ".join(own)) or "_not established_"
            lines.append(f"**{label}.** {text}")
            if cited:
                lines.append("")
                lines.append(
                    "  _Named by the agent but stated only where the paper reports cited "
                    "works, so not this paper's own:_ "
                    + "; ".join(f"{_esc(item)}{tag}" for item, tag in cited)
                )
            lines.append("")
        lines.pop()

        hyperparameters = payload.get("hyperparameters") or {}
        if hyperparameters:
            lines.append("")
            lines.append("**Hyperparameters.**")
            lines.append("")
            lines.append("| Name | Value |")
            lines.append("| --- | --- |")
            for name, value in hyperparameters.items():
                lines.append(f"| {_esc(name)} | {_esc(value)} |")
        return "\n".join(lines)

    def _results(self, agent_results: dict[str, AgentResult], findings: list[Finding]) -> str:
        result = agent_results.get(AgentName.EXPERIMENTS.value)
        lines = ["## 7. Results and ablations", ""]
        if result is None or result.status != "ok":
            lines.append(self._not_run(result, "experiments"))
            return "\n".join(lines)

        payload = result.payload
        reported = payload.get("reported_results") or []
        own_rows: list[str] = []
        cited_rows: list[str] = []
        for item in reported:
            evidence = [Evidence(**e) for e in item.get("evidence", [])]
            kind, sources = self._evidence_attribution(evidence)
            row = (
                f"| {_esc(item.get('system')) or '—'} "
                f"| {_esc(item.get('metric'))} "
                f"| {_esc(item.get('value'))} "
                f"| {_esc(item.get('dataset')) or '—'} "
                f"| {format_citations(evidence, limit=2)} |"
            )
            if kind == "cited":
                cited_rows.append(row[:-1] + f"| {describe_sources(sources)} |")
            else:
                own_rows.append(row)
        if own_rows:
            lines.append("| System | Metric | Value | Dataset | Evidence |")
            lines.append("| --- | --- | --- | --- | --- |")
            lines.extend(own_rows)
            lines.append("")
        if cited_rows:
            lines.append(
                "**Numbers the paper reports from cited works.** These are results of the "
                "works it cites, stated in passages that report them — not results of this paper."
            )
            lines.append("")
            lines.append("| System | Metric | Value | Dataset | Evidence | Cited work |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            lines.extend(cited_rows)
            lines.append("")
        inventory = self._unread_elements(reported)
        if inventory:
            lines.extend(inventory)
            lines.append("")

        unverified = payload.get("unverified_numbers") or []
        if unverified:
            lines.append(
                "> ⚠️ **Unverified values.** The following reported values could not be "
                "located in their own cited evidence and must not be relied on: "
                + ", ".join(f"`{_esc(u)}`" for u in unverified)
                + "."
            )
            lines.append("")

        conflicts = payload.get("value_conflicts") or []
        if conflicts:
            lines.append(
                "> ⚠️ **The paper reports these results with two different values.** Both "
                "are shown; neither was chosen for you."
            )
            lines.append("")
            lines.append("| Result | Reported value | Where | Differing value stated | Where |")
            lines.append("| --- | --- | --- | --- | --- |")
            for conflict in conflicts:
                subject = " ".join(
                    _esc(conflict.get(k)) for k in ("system", "metric") if conflict.get(k)
                )
                if conflict.get("dataset"):
                    subject += f" ({_esc(conflict['dataset'])})"
                reported = [Evidence(**e) for e in conflict.get("reported_evidence", [])]
                stated = Evidence(**conflict["stated_evidence"])
                lines.append(
                    f"| {subject} | {_esc(conflict.get('reported_value'))} "
                    f"| {format_citations(reported, limit=2)} "
                    f"| {_esc(conflict.get('stated_value'))} | {format_citations([stated])} |"
                )
            lines.append("")

        ablations = payload.get("ablations") or []
        if ablations:
            lines.append("### Ablations")
            lines.append("")
            lines.extend(f"- {_esc(a)}" for a in ablations)
            lines.append("")

        if payload.get("statistical_significance"):
            lines.append(
                f"**Statistical significance.** {_esc(payload['statistical_significance'])}"
            )
            lines.append("")
        if payload.get("error_analysis"):
            lines.append(f"**Error analysis.** {_esc(payload['error_analysis'])}")
            lines.append("")

        assessment = payload.get("conclusions_supported")
        if assessment:
            label = _STATUS_LABEL.get(FindingStatus(assessment), assessment) if assessment in {
                s.value for s in FindingStatus
            } else assessment
            lines.append(
                f"**Do the experiments support the paper's conclusions?** {label}. "
                f"{_esc(payload.get('conclusions_assessment'))}"
            )
        return "\n".join(lines)

    def _limitations(self, agent_results: dict[str, AgentResult], findings: list[Finding]) -> str:
        summary = self._payload(agent_results, AgentName.SUMMARY)
        lines = ["## 8. Limitations and threats to validity", ""]
        stated = self._prose(summary.get("limitations")) if summary else ""
        lines.append(
            f"**Stated by the authors.** {stated or '_no limitations were found in the paper_'}"
        )
        lines.append("")
        if self._kind == "review":
            gaps = [item for item in review_protocol(self._index) if not item.found]
            if gaps:
                lines.append(
                    "**Gaps in the review's protocol** _(a deterministic check of the paper's own "
                    "sentences; see §9)_. The paper does not say how it chose or weighed the "
                    "works it reviews:"
                )
                lines.append("")
                lines.extend(f"- {_esc(item.gap)}." for item in gaps)
                lines.append("")
        assumptions = self._prose(summary.get("assumptions")) if summary else ""
        if assumptions:
            lines.append(f"**Assumptions.** {assumptions}")
            lines.append("")
        limitation_findings = self._unrendered(
            [f for f in findings if (f.category or "").lower() in ("limitations", "limitation")]
        )
        own = [f for f in limitation_findings if f.attribution != "cited"]
        cited = [f for f in limitation_findings if f.attribution == "cited"]
        if own:
            lines.extend(self._bullets(own))
            lines.append("")
        if cited:
            lines.append(
                "**Limitations the paper reports from cited works.** These are what the works "
                "it cites say about their subject, not limitations of this paper's own work:"
            )
            lines.append("")
            lines.extend(self._bullets(cited))
            lines.append("")
        lines.append(
            "**Threats to validity of this report.** This analysis reads only the "
            "submitted PDF. It cannot check the authors' claims against the outside "
            "world, reproduce any experiment, or detect an error that the paper itself "
            "does not reveal. Section 12 lists what it could not ground, and section 14 what it read."
        )
        return "\n".join(lines)

    def _reproducibility(
        self, agent_results: dict[str, AgentResult], metadata: PaperMetadata
    ) -> str:
        if self._kind == "review":
            return self._review_protocol_checklist()
        experiments = agent_results.get(AgentName.EXPERIMENTS.value)
        methodology = agent_results.get(AgentName.METHODOLOGY.value)
        exp_payload = experiments.payload if experiments and experiments.status == "ok" else {}
        method_payload = methodology.payload if methodology and methodology.status == "ok" else {}

        # Every row goes through _stated: a field holding "Not explicitly
        # stated." is the agent reporting an absence, and must tick the box
        # the same way an empty field does.
        # Datasets, baselines and metrics count only when they are the paper's
        # own: a cited study's dataset does not make this paper reproducible.
        def own(key: str) -> bool:
            return bool(self._split_by_attribution(_stated(exp_payload.get(key) or []))[0])

        checks = [
            ("Datasets identified", own("datasets")),
            ("Splits described", bool(_stated(exp_payload.get("splits")))),
            ("Baselines identified", own("baselines")),
            ("Metrics identified", own("metrics")),
            ("Hyperparameters reported", bool(_stated(exp_payload.get("hyperparameters")))),
            (
                "Training procedure described",
                bool(_stated(method_payload.get("training_procedure"))),
            ),
            (
                "Inference procedure described",
                bool(_stated(method_payload.get("inference_procedure"))),
            ),
            (
                "Statistical significance reported",
                bool(_stated(exp_payload.get("statistical_significance"))),
            ),
            ("Ablations reported", bool(_stated(exp_payload.get("ablations")))),
        ]
        lines = [self._title("reproducibility"), ""]
        lines.append(
            "_Each row records whether PaperMind could find the item **in this PDF**. "
            "An unchecked box means it was not found, not that it does not exist._"
        )
        lines.append("")
        lines.append("| Item | Found in the paper |")
        lines.append("| --- | --- |")
        for label, present in checks:
            lines.append(f"| {label} | {'✅ yes' if present else '❌ not found'} |")
        return "\n".join(lines)

    def _rest_of_paper(self, agent_results: dict[str, AgentResult]) -> str:
        """What the coverage sweep found in the parts no specialist read."""
        result = agent_results.get("coverage")
        lines = ["## 10. Details from the rest of the paper", ""]
        if result is None or result.status == "skipped":
            reason = result.skip_reason if result else "the coverage sweep did not run"
            lines.append(f"_Nothing to add: {_esc(reason)}._")
            return "\n".join(lines)
        if result.status != "ok":
            lines.append(self._not_run(result, "coverage"))
            return "\n".join(lines)

        payload = result.payload
        lines.append(
            "_Every part of the paper that none of the specialists above was shown was read "
            "here, in document order, so that nothing the paper states is lost to retrieval._"
        )
        lines.append("")

        details = [d for d in payload.get("details") or [] if d.get("value_found")]
        if details:
            lines.append("| Setting or value | Value | Evidence |")
            lines.append("| --- | --- | --- |")
            for detail in details:
                evidence = [Evidence(**e) for e in detail.get("evidence", [])]
                kind, sources = self._evidence_attribution(evidence)
                lines.append(
                    f"| {_esc(detail.get('name'))} | {_esc(detail.get('value'))} "
                    f"| {format_citations(evidence, limit=2)}{_attribution_tag(kind, sources)} |"
                )
            lines.append("")

        grounded = [
            f
            for f in self._unrendered(result.findings)
            if f.status in (FindingStatus.SUPPORTED, FindingStatus.PARTIALLY_SUPPORTED)
            and f.has_evidence
        ]
        by_section: dict[str, list[Finding]] = defaultdict(list)
        for finding in grounded:
            section = next((e.section for e in finding.evidence if e.is_verified and e.section), None)
            by_section[section or "Other"].append(finding)
        order = {name: position for position, name in enumerate(payload.get("sections_read") or [])}
        for section in sorted(by_section, key=lambda name: order.get(name, len(order))):
            lines.append(f"### {_esc(section)}")
            lines.append("")
            lines.extend(self._bullets(by_section[section]))
            lines.append("")
        if not details and not grounded:
            lines.append("_The sweep found nothing further that could be grounded in the paper._")
        return "\n".join(lines).rstrip()

    def _verified_findings(self, findings: list[Finding]) -> str:
        verified = [
            f
            for f in findings
            if f.status in (FindingStatus.SUPPORTED, FindingStatus.PARTIALLY_SUPPORTED)
        ]
        lines = ["## 11. Verified findings", ""]
        if not verified:
            lines.append("_No finding reached a supported or partially supported status._")
            return "\n".join(lines)

        lines.append(
            f"{len(verified)} finding(s) were checked against the paper and grounded."
        )
        lines.append("")
        by_status: dict[FindingStatus, list[Finding]] = defaultdict(list)
        for finding in verified:
            by_status[finding.status].append(finding)

        for status in (FindingStatus.SUPPORTED, FindingStatus.PARTIALLY_SUPPORTED):
            group = by_status.get(status)
            if not group:
                continue
            lines.append(f"### {_STATUS_LABEL[status]} ({len(group)})")
            lines.append("")
            for finding in sorted(group, key=lambda f: -f.confidence):
                lines.append(
                    f"- **{_esc(finding.claim)}** {format_citations(finding.evidence)}"
                    f"{_attribution_tag(finding.attribution, finding.cited_sources)} "
                    f"<sub>confidence {finding.confidence:.2f} · {finding.agent}</sub>"
                )
                if finding.verification_note:
                    lines.append(f"  - _Verifier:_ {_esc(finding.verification_note)}")
            lines.append("")
        return "\n".join(lines)

    def _weak_findings(
        self, findings: list[Finding], verification_results: list[VerificationResult]
    ) -> str:
        weak = [f for f in findings if f.status in _WEAK or f.status is FindingStatus.INFERRED]
        lines = ["## 12. Unsupported, contradicted and inferred findings", ""]
        lines.append(
            "_These statements were produced during analysis but could **not** be "
            "grounded in the paper. They are shown so that the failure is visible; they "
            "should not be treated as facts about the paper._"
        )
        lines.append("")
        if not weak:
            lines.append("_Every finding in this run was grounded in the paper._")
            return "\n".join(lines)

        by_status: dict[FindingStatus, list[Finding]] = defaultdict(list)
        for finding in weak:
            by_status[finding.status].append(finding)

        verification_by_claim = {r.claim.strip().lower(): r for r in verification_results}

        for status in (
            FindingStatus.CONTRADICTED,
            FindingStatus.UNSUPPORTED,
            FindingStatus.INFERRED,
            FindingStatus.UNKNOWN,
        ):
            group = by_status.get(status)
            if not group:
                continue
            lines.append(f"### {_STATUS_LABEL[status]} ({len(group)})")
            lines.append("")
            for finding in group:
                lines.append(f"- **{_esc(finding.claim)}** <sub>{finding.agent}</sub>")
                result = verification_by_claim.get(finding.claim.strip().lower())
                if result is not None and result.status is not finding.status:
                    # A later check (a retry's evidence, a deterministic cap) changed
                    # the status; the earlier verdict's reasoning no longer applies.
                    result = None
                why = ""
                if result is not None:
                    # The deterministic checks are what settled this verdict, so
                    # they lead. The verifier's own reading can argue the exact
                    # opposite — it called the claim supported and a quote that
                    # could not be found in the paper overruled it — and printed
                    # alone it explains the reverse of what the section reports.
                    if result.deterministic_notes:
                        why = "; ".join(result.deterministic_notes)
                        if result.explanation:
                            why += (
                                " — the verifier's own reading, which these checks "
                                f"overruled: {result.explanation}"
                            )
                    else:
                        why = result.explanation
                if not why:
                    why = finding.verification_note or ""
                if why:
                    lines.append(f"  - _Why:_ {_esc(why)}")
                if result and result.numeric_checks:
                    missing = [c.value for c in result.numeric_checks if not c.found_in_evidence]
                    if missing:
                        lines.append(
                            "  - _Numbers not found in the paper:_ "
                            + ", ".join(f"`{_esc(m)}`" for m in missing)
                        )
                if result and result.contradicting_evidence:
                    lines.append(
                        "  - _Contradicting evidence:_ "
                        + format_citations(result.contradicting_evidence)
                    )
            lines.append("")
        return "\n".join(lines)

    def _open_questions(
        self, agent_results: dict[str, AgentResult], findings: list[Finding]
    ) -> str:
        questions: list[str] = []
        for result in agent_results.values():
            questions.extend(result.open_questions)
        summary = self._payload(agent_results, AgentName.SUMMARY)
        if summary:
            questions.extend(summary.get("unresolved_questions") or [])

        seen: set[str] = set()
        unique = []
        for question in questions:
            key = question.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(question)

        lines = ["## 13. Open questions", ""]
        if not unique:
            lines.append("_No open questions were recorded._")
            return "\n".join(lines)
        lines.extend(f"- {_esc(q)}" for q in unique)
        return "\n".join(lines)

    def _how_this_was_produced(
        self,
        plan: AnalysisPlan | None,
        agent_results: dict[str, AgentResult],
        verification_results: list[VerificationResult],
        retry_count: int,
        usage: list[AgentUsage] | None,
    ) -> str:
        lines = ["## 14. How this report was produced", ""]
        if plan:
            lines.append(
                f"**Plan** (created by: {plan.created_by}). Agents selected: "
                f"{', '.join(plan.enabled_agents())}."
            )
            lines.append("")
            lines.append(
                f"**Paper type.** Analysed as a **{plan.paper_kind}** paper, decided "
                "deterministically from its text: "
                + "; ".join(_esc(s) for s in plan.paper_kind_signals)
                + ". Section titles, the reproducibility checklist and each agent's reference "
                "guide follow from this."
            )
            lines.append("")
            if plan.rationale:
                lines.append("| Decision | Reason |")
                lines.append("| --- | --- |")
                for key, reason in plan.rationale.items():
                    lines.append(f"| `{_esc(key)}` | {_esc(reason)} |")
                lines.append("")

        lines.append("| Agent | Status | Findings | Attempt | Detail |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, result in agent_results.items():
            detail = result.skip_reason or result.error or ""
            lines.append(
                f"| {name} | {result.status} | {len(result.findings)} | "
                f"{result.attempt} | {_esc(detail)} |"
            )
        lines.append("")

        coverage = agent_results.get("coverage")
        if coverage is not None and coverage.payload.get("content_chunks"):
            data = coverage.payload
            total = data["content_chunks"]
            read = data.get("read_by_specialists", 0) + data.get("read_by_sweep", 0)
            lines.append(
                f"**Coverage.** {read} of {total} content chunks were read by a model: "
                f"{data.get('read_by_specialists', 0)} by the specialists and "
                f"{data.get('read_by_sweep', 0)} by the coverage sweep."
            )
            if data.get("unread_sections"):
                lines.append(
                    "Not read, because the sweep budget ran out: "
                    + ", ".join(f"§{_esc(s)}" for s in data["unread_sections"])
                    + "."
                )
            lines.append("")

        reported = [
            f
            for result in agent_results.values()
            for f in result.findings
            if id(f) in getattr(self, "_live", set()) and f.attribution == "cited"
        ]
        if reported:
            lines.append(
                f"**Attribution.** {len(reported)} finding(s) rest only on passages in which the "
                "paper reports a cited work; each is marked “reported from cited work [N]”. "
                "Attribution follows the paper's own wording (“In [N] the authors …”, "
                "“According to them …”) and is decided without a model."
            )
            lines.append("")

        issues = heading_numbering_issues(heading_sequence(self._index))
        if issues:
            lines.append(
                "**Structure checks.** As extracted from the PDF: "
                + "; ".join(_esc(i) for i in issues)
                + ". Section references to these numbers are ambiguous."
            )
            lines.append("")

        diagnostics = [
            d for result in agent_results.values() for d in result.payload.get("diagnostics") or []
        ]
        if diagnostics:
            lines.append("**Pipeline notes.**")
            lines.append("")
            lines.extend(f"- {_esc(d)}" for d in diagnostics)
            lines.append("")

        if verification_results:
            counts: dict[str, int] = defaultdict(int)
            for result in verification_results:
                counts[result.status.value] += 1
            lines.append(
                "**Verification.** "
                + ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
                + f" across {len(verification_results)} high-impact claim(s)."
            )
            lines.append("")

        lines.append(
            f"**Retries.** {retry_count} retry round(s) were run"
            + (
                f" (budget: {plan.max_retries})." if plan else "."
            )
        )
        if usage:
            total = sum(u.total_tokens for u in usage)
            costs = [u.estimated_cost_usd for u in usage if u.estimated_cost_usd is not None]
            models = sorted({u.model for u in usage})
            lines.append("")
            lines.append(
                f"**Model usage.** {total} tokens across {len(usage)} call(s); "
                f"model(s): {', '.join(models)}."
                + (f" Estimated cost: ${sum(costs):.4f}." if costs else "")
            )
            if any(m.startswith("mock") for m in models):
                lines.append("")
                lines.append(
                    "> ⚠️ **This run used the deterministic mock provider.** The mock assembles "
                    "answers from retrieved sentences to exercise the pipeline offline. It is "
                    "not a language model and this report is not a real analysis of the paper. "
                    "Set `PAPERPILOT_LLM_PROVIDER=openai_compatible` for a genuine run."
                )
        return "\n".join(lines)

    def _evidence_appendix(
        self, findings: list[Finding], verification_results: list[VerificationResult]
    ) -> str:
        evidence: list[tuple[Evidence, str]] = []
        for finding in findings:
            for item in finding.evidence:
                evidence.append((item, finding.claim))
        for result in verification_results:
            for item in result.supporting_evidence + result.contradicting_evidence:
                evidence.append((item, result.claim))

        seen: set[tuple] = set()
        unique: list[tuple[Evidence, str]] = []
        for item, claim in evidence:
            key = (item.page, item.chunk_id, item.quote[:80])
            if key in seen:
                continue
            seen.add(key)
            unique.append((item, claim))

        unique.sort(key=lambda pair: citation_key(pair[0]))
        lines = ["## 15. Evidence appendix", ""]
        if not unique:
            lines.append("_No evidence was collected._")
            return "\n".join(lines)

        lines.append(
            f"Every quote used anywhere in this report, in page order "
            f"({len(unique)} item(s), showing up to {self.max_appendix_items})."
        )
        lines.append("")
        for index, (item, claim) in enumerate(unique[: self.max_appendix_items], start=1):
            from .citations import format_citation

            citation = format_citation(item, include_quality=True)
            lines.append(f"**E{index}.** {citation} — supports: _{_esc(claim)[:160]}_")
            lines.append("")
            lines.append(format_quote(item))
            if item.validation_notes:
                lines.append("")
                lines.append(
                    "_Validation notes:_ " + "; ".join(_esc(n) for n in item.validation_notes)
                )
            lines.append("")
        return "\n".join(lines)

    def _footer(self, metadata: PaperMetadata, index: PaperIndex) -> str:
        generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        return (
            "---\n\n"
            f"_Generated by PaperMind on {generated} from `{_esc(metadata.filename)}` "
            f"(document id `{metadata.document_id}`, {index.page_count} pages, "
            f"{len(index.chunks)} indexed chunks)._\n\n"
            "_PaperMind is a domain-specific research-paper analysis system. It is not "
            "general intelligence, it does not have knowledge of this paper beyond the "
            "submitted PDF, and it does not verify the paper's claims against the outside "
            "world — only against the paper's own text._"
        )

    # --- helpers ----------------------------------------------------------

    def _title(self, key: str) -> str:
        return _KIND_TITLES.get(self._kind, {}).get(key, _DEFAULT_TITLES[key])

    def _unrendered(self, findings: list[Finding]) -> list[Finding]:
        """Live findings not yet printed in an earlier narrative section."""
        live = getattr(self, "_live", None)
        rendered = getattr(self, "_rendered", set())
        return [
            f for f in findings if (live is None or id(f) in live) and id(f) not in rendered
        ]

    def _bullets(self, findings: list[Finding]) -> list[str]:
        out = []
        for finding in self._unrendered(findings):
            self._rendered.add(id(finding))
            out.append(_bullet(finding))
        return out

    def _evidence_attribution(self, evidence: list[Evidence]) -> tuple[str, list[str]]:
        verified = [e for e in evidence if e.is_verified] or evidence
        return _combined_attribution([evidence_attribution(e, self._amap) for e in verified])

    def _split_by_attribution(self, items: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
        """Items the paper states itself, and items it states only from cited works."""
        own: list[str] = []
        cited: list[tuple[str, str]] = []
        for item in items:
            kind, sources = text_attribution(item, self._amap)
            if kind == "cited":
                cited.append((item, _attribution_tag(kind, sources)))
            else:
                own.append(item)
        return own, cited

    def _unread_elements(self, reported: list[dict]) -> list[str]:
        """Tables and figures of the paper that no reported result cites.

        A table is the paper's own evidence even when no agent turned it into
        rows, so its extracted text is shown rather than left out.
        """
        cited_elements = {
            e.get("element_id") for item in reported for e in item.get("evidence", []) if e.get("element_id")
        }
        tables: dict[str, list] = {}
        figures: list = []
        for chunk in self._index.chunks:
            if chunk.element_type is ElementType.TABLE and chunk.element_id:
                tables.setdefault(chunk.element_id, []).append(chunk)
            elif chunk.element_type is ElementType.FIGURE and chunk.element_id:
                if chunk.element_id not in {c.element_id for c in figures}:
                    figures.append(chunk)
        lines: list[str] = []
        unread = {eid: chunks for eid, chunks in tables.items() if eid not in cited_elements}
        if unread:
            lines.append("### Tables not read into the results above")
            lines.append("")
            lines.append(
                "_The paper's own tables that no reported result cites. Their extracted text is "
                "shown as-is; a table that extracted poorly may read out of order._"
            )
            lines.append("")
            for chunks in unread.values():
                first = chunks[0]
                caption = _esc(first.text.split("\n")[0])[:160]
                body = " ".join(" ".join(c.text.split()) for c in chunks)[:900]
                lines.append(f"**{caption}** [p. {first.page}]")
                lines.append("")
                lines.append(f"> {_esc(body)}")
                lines.append("")
        if figures:
            lines.append("### Figures in the paper")
            lines.append("")
            for chunk in figures:
                caption = _esc(chunk.text.split("\n")[0])[:160]
                lines.append(f"- {caption} [p. {chunk.page}]")
        return lines

    def _review_protocol_checklist(self) -> str:
        lines = [self._title("reproducibility"), ""]
        lines.append(
            "_For a review, reproducibility means its protocol. Each row is looked for in the "
            "sentences where the paper speaks for itself — a sentence reporting a cited study "
            "describes that study, not this review. Found rows cite the sentence._"
        )
        lines.append("")
        lines.append("| Item | Found in the paper | Where |")
        lines.append("| --- | --- | --- |")
        for item in review_protocol(self._index):
            where = format_citations(item.evidence, limit=2) if item.found else ""
            lines.append(
                f"| {item.name} | {'✅ yes' if item.found else '❌ not found'} | {where} |"
            )
        return "\n".join(lines)

    def _prose(self, text: str | None) -> str:
        """Escape model-written prose and flag terms of art the paper never uses."""
        body = _esc(text)
        vocabulary = getattr(self, "_vocabulary", None)
        terms = vocabulary.unstated_terms(text) if body and vocabulary else []
        if terms:
            body += (
                " ⚠️ _Not stated in the paper: "
                + ", ".join(f"“{_esc(t)}”" for t in terms)
                + " — this wording is the system's, not the authors'._"
            )
        return body

    @staticmethod
    def _payload(agent_results: dict[str, AgentResult], agent: AgentName) -> dict:
        result = agent_results.get(agent.value)
        return result.payload if result and result.status == "ok" else {}

    @staticmethod
    def _not_run(result: AgentResult | None, name: str) -> str:
        if result is None:
            return f"_The {name} agent did not run._"
        if result.status == "skipped":
            return f"_The {name} agent was skipped: {_esc(result.skip_reason)}._"
        if result.status == "failed":
            return f"_The {name} agent failed: {_esc(result.error)}._"
        return f"_The {name} agent produced no usable output._"
