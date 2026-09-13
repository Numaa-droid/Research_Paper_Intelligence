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

from collections import defaultdict
from datetime import UTC, datetime

from ..domain.enums import AgentName, FindingStatus
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


def _bullet(finding: Finding) -> str:
    citation = format_citations(finding.evidence)
    line = f"- **{_esc(finding.claim)}** {citation}"
    if finding.explanation:
        line += f"\n  {_esc(finding.explanation)}"
    return line


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
        lines = ["# PaperPilot Report", ""]
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
        body = _esc(summary.get("concise_summary")) if summary else ""

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
        lines.append(f"**Problem.** {_esc(summary.get('research_problem')) or '_not established_'}")
        lines.append("")
        lines.append(f"**Motivation.** {_esc(summary.get('motivation')) or '_not established_'}")
        return "\n".join(lines)

    def _contributions(
        self, agent_results: dict[str, AgentResult], findings: list[Finding]
    ) -> str:
        summary = self._payload(agent_results, AgentName.SUMMARY)
        lines = ["## 3. Contributions", ""]
        if summary and summary.get("main_contribution"):
            lines.append(_esc(summary["main_contribution"]))
            lines.append("")
        contribution_findings = [
            f for f in findings if (f.category or "").lower() in ("contribution", "problem")
        ]
        if contribution_findings:
            lines.extend(_bullet(f) for f in contribution_findings)
        elif not (summary and summary.get("main_contribution")):
            lines.append("_No contribution claims could be grounded in the paper._")
        return "\n".join(lines)

    def _methodology(self, agent_results: dict[str, AgentResult]) -> str:
        result = agent_results.get(AgentName.METHODOLOGY.value)
        lines = ["## 4. Methodology", ""]
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
            value = _esc(payload.get(key))
            if value:
                lines.append(f"**{label}.** {value}")
                lines.append("")

        steps = payload.get("pipeline_steps") or []
        if steps:
            lines.append("### Execution flow")
            lines.append("")
            for step in steps:
                evidence = [Evidence(**e) for e in step.get("evidence", [])]
                marker = " _(inferred, not stated by the authors)_" if step.get("is_inferred") else ""
                lines.append(
                    f"{step.get('order', 0)}. **{_esc(step.get('name'))}** — "
                    f"{_esc(step.get('description'))}{marker} {format_citations(evidence)}"
                )
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

        if result.findings:
            lines.append("### Methodology findings")
            lines.append("")
            lines.extend(_bullet(f) for f in result.findings)
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
            lines.append(f"**Optimisation objective.** {_esc(payload['optimization_objective'])}")
            lines.append("")

        equations = payload.get("equations") or []
        if not equations:
            lines.append("_No equations could be extracted legibly from this paper._")
        for equation in equations:
            identifier = equation.get("equation_id") or f"page {equation.get('page')}"
            lines.append(f"### {_esc(identifier)} (p. {equation.get('page')})")
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
        lines = ["## 6. Dataset and experimental setup", ""]
        if result is None or result.status != "ok":
            lines.append(self._not_run(result, "experiments"))
            return "\n".join(lines)

        payload = result.payload
        datasets = payload.get("datasets") or []
        baselines = payload.get("baselines") or []
        metrics = payload.get("metrics") or []

        lines.append(f"**Datasets.** {_esc(', '.join(datasets)) or '_not established_'}")
        lines.append("")
        lines.append(f"**Splits.** {_esc(payload.get('splits')) or '_not established_'}")
        lines.append("")
        lines.append(f"**Baselines.** {_esc(', '.join(baselines)) or '_not established_'}")
        lines.append("")
        lines.append(f"**Metrics.** {_esc(', '.join(metrics)) or '_not established_'}")

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
        if reported:
            lines.append("| System | Metric | Value | Dataset | Evidence |")
            lines.append("| --- | --- | --- | --- | --- |")
            for item in reported:
                evidence = [Evidence(**e) for e in item.get("evidence", [])]
                lines.append(
                    f"| {_esc(item.get('system')) or '—'} "
                    f"| {_esc(item.get('metric'))} "
                    f"| {_esc(item.get('value'))} "
                    f"| {_esc(item.get('dataset')) or '—'} "
                    f"| {format_citations(evidence, limit=2)} |"
                )
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
        stated = _esc(summary.get("limitations")) if summary else ""
        lines.append(
            f"**Stated by the authors.** {stated or '_no limitations were found in the paper_'}"
        )
        lines.append("")
        assumptions = _esc(summary.get("assumptions")) if summary else ""
        if assumptions:
            lines.append(f"**Assumptions.** {assumptions}")
            lines.append("")
        limitation_findings = [
            f for f in findings if (f.category or "").lower() in ("limitations", "limitation")
        ]
        if limitation_findings:
            lines.extend(_bullet(f) for f in limitation_findings)
            lines.append("")
        lines.append(
            "**Threats to validity of this report.** This analysis reads only the "
            "submitted PDF. It cannot check the authors' claims against the outside "
            "world, reproduce any experiment, or detect an error that the paper itself "
            "does not reveal. Section 12 lists what it could not ground."
        )
        return "\n".join(lines)

    def _reproducibility(
        self, agent_results: dict[str, AgentResult], metadata: PaperMetadata
    ) -> str:
        experiments = agent_results.get(AgentName.EXPERIMENTS.value)
        methodology = agent_results.get(AgentName.METHODOLOGY.value)
        exp_payload = experiments.payload if experiments and experiments.status == "ok" else {}
        method_payload = methodology.payload if methodology and methodology.status == "ok" else {}

        checks = [
            ("Datasets identified", bool(exp_payload.get("datasets"))),
            ("Splits described", bool(exp_payload.get("splits"))),
            ("Baselines identified", bool(exp_payload.get("baselines"))),
            ("Metrics identified", bool(exp_payload.get("metrics"))),
            ("Hyperparameters reported", bool(exp_payload.get("hyperparameters"))),
            ("Training procedure described", bool(method_payload.get("training_procedure"))),
            ("Inference procedure described", bool(method_payload.get("inference_procedure"))),
            (
                "Statistical significance reported",
                bool(exp_payload.get("statistical_significance")),
            ),
            ("Ablations reported", bool(exp_payload.get("ablations"))),
        ]
        lines = ["## 9. Reproducibility checklist", ""]
        lines.append(
            "_Each row records whether PaperPilot could find the item **in this PDF**. "
            "An unchecked box means it was not found, not that it does not exist._"
        )
        lines.append("")
        lines.append("| Item | Found in the paper |")
        lines.append("| --- | --- |")
        for label, present in checks:
            lines.append(f"| {label} | {'✅ yes' if present else '❌ not found'} |")
        return "\n".join(lines)

    def _verified_findings(self, findings: list[Finding]) -> str:
        verified = [
            f
            for f in findings
            if f.status in (FindingStatus.SUPPORTED, FindingStatus.PARTIALLY_SUPPORTED)
        ]
        lines = ["## 10. Verified findings", ""]
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
                    f"- **{_esc(finding.claim)}** {format_citations(finding.evidence)} "
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
        lines = ["## 11. Unsupported, contradicted and inferred findings", ""]
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
                if result and result.explanation:
                    lines.append(f"  - _Why:_ {_esc(result.explanation)}")
                elif finding.verification_note:
                    lines.append(f"  - _Why:_ {_esc(finding.verification_note)}")
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

        lines = ["## 12. Open questions", ""]
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
        lines = ["## 13. How this report was produced", ""]
        if plan:
            lines.append(
                f"**Plan** (created by: {plan.created_by}). Agents selected: "
                f"{', '.join(plan.enabled_agents())}."
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
        lines = ["## 14. Evidence appendix", ""]
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
            f"_Generated by PaperPilot on {generated} from `{_esc(metadata.filename)}` "
            f"(document id `{metadata.document_id}`, {index.page_count} pages, "
            f"{len(index.chunks)} indexed chunks)._\n\n"
            "_PaperPilot is a domain-specific research-paper analysis system. It is not "
            "general intelligence, it does not have knowledge of this paper beyond the "
            "submitted PDF, and it does not verify the paper's claims against the outside "
            "world — only against the paper's own text._"
        )

    # --- helpers ----------------------------------------------------------

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
