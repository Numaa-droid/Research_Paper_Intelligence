"""The brief: a paper in two pages.

The full report is exhaustive, which is right for checking a paper and wrong
for understanding one: a reader handed fifteen sections may as well read the
PDF. The brief is what to read first: the problem, what the paper does, what it
found, the takeaways, and how far to trust them.

It is built from exactly the same validated structures as the full report, by
the same checks (:meth:`ReportSynthesizer._prose` flags unstated terms and
numbers), with no model call. It only *selects and shortens*: it can leave
something out, but it cannot say anything the full report does not.
"""

from __future__ import annotations

import re

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
from .citations import format_citations
from .synthesizer import _STATUS_LABEL, _WEAK, ReportSynthesizer, _esc, _stated

#: Word budget for the whole brief: about two printed pages with the table.
TARGET_WORDS = 900

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(\[])")

#: Finding categories in the order a reader wants them as takeaways.
_TAKEAWAY_ORDER = ("contribution", "results", "result", "method", "methodology", "problem")


def clip(text: str | None, *, sentences: int, max_chars: int = 600) -> str:
    """The first ``sentences`` sentences of ``text``, cut at ``max_chars``."""
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    kept = " ".join(_SENTENCE.split(text)[:sentences])
    if len(kept) > max_chars:
        kept = kept[: max_chars - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return kept


_CAPTION = re.compile(r"\b(?:table|figure|fig\.)\s*\d+\s*:", re.IGNORECASE)


def _readable(claim: str) -> bool:
    """Whether a finding reads as a takeaway, not a caption or a flattened table."""
    numbers = re.findall(r"\d+(?:\.\d+)?", claim)
    return not _CAPTION.search(claim) and len(numbers) <= 6


def _key(sentence: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()


class BriefSynthesizer(ReportSynthesizer):
    """Renders the two-page brief."""

    def _fresh(self, text: str) -> str:
        """``text`` without sentences an earlier section already said.

        The summary's fields overlap (its one-paragraph summary usually restates
        the contribution and the headline result); in two pages, saying a thing
        twice is a sentence of something else not said.
        """
        kept = []
        for sentence in _SENTENCE.split(text or ""):
            key = _key(sentence)
            if key and key not in self._said:
                self._said.add(key)
                kept.append(sentence)
        return " ".join(kept)

    def _already_said(self, text: str) -> bool:
        key = _key(text)
        return any(key in said or said in key for said in self._said if len(said) > 30)

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
        self._prepare(index=index, plan=plan, findings=findings)
        self._said: set[str] = set()
        summary = self._payload(agent_results, AgentName.SUMMARY)
        sections = [
            self._brief_header(metadata),
            self._in_short(summary),
            self._the_problem(summary),
            self._what_it_does(summary, agent_results),
            self._what_it_found(summary, agent_results),
            self._takeaways(findings),
            self._trust(summary, agent_results, findings, verification_results),
            self._more(),
        ]
        return "\n\n".join(s for s in sections if s and s.strip()) + "\n"

    # --- sections ---------------------------------------------------------

    def _brief_header(self, metadata: PaperMetadata) -> str:
        title = _esc(metadata.title) or _esc(metadata.filename) or "Untitled paper"
        facts = []
        if metadata.authors:
            authors = metadata.authors
            facts.append(
                _esc(", ".join(authors[:3]) + (f" and {len(authors) - 3} more" if len(authors) > 3 else ""))
            )
        if metadata.year:
            facts.append(str(metadata.year))
        facts.append(f"{metadata.page_count} pages")
        return "\n".join(
            [
                f"# {title}",
                "",
                " · ".join(facts),
                "",
                "> **PaperMind brief:** the paper in two pages. Every statement below comes from "
                "the paper and was checked against it; `[p. N]` marks where to read more. The full "
                "report has the method, setup, every result and all the evidence.",
            ]
        )

    def _in_short(self, summary: dict) -> str:
        body = self._prose(self._fresh(clip(_stated(summary.get("concise_summary")), sentences=5, max_chars=900)))
        if not body:
            return ""
        return f"## In short\n\n{body}"

    def _the_problem(self, summary: dict) -> str:
        problem = self._fresh(clip(_stated(summary.get("research_problem")), sentences=2, max_chars=400))
        motivation = self._fresh(clip(_stated(summary.get("motivation")), sentences=2, max_chars=400))
        if not (problem or motivation):
            return ""
        lines = ["## The problem", ""]
        if problem:
            lines.append(self._prose(problem, subject="research problem"))
        if motivation:
            lines.append("")
            lines.append(f"**Why it matters.** {self._prose(motivation, subject='motivation')}")
        return "\n".join(lines)

    def _what_it_does(self, summary: dict, agent_results: dict[str, AgentResult]) -> str:
        title = "## How the review was done" if self._kind == "review" else "## What the paper does"
        lines = [title, ""]
        contribution = self._fresh(clip(_stated(summary.get("main_contribution")), sentences=3, max_chars=550))
        method = self._fresh(clip(_stated(summary.get("proposed_method")), sentences=3, max_chars=550))
        if contribution:
            lines.append(self._prose(contribution, subject="contribution"))
            lines.append("")
        if method:
            lines.append(f"**The approach.** {self._prose(method, subject='method')}")
            lines.append("")

        steps = self._payload(agent_results, AgentName.METHODOLOGY).get("pipeline_steps") or []
        steps = sorted(steps, key=lambda s: s.get("order", 0))[:5]
        if steps:
            lines.append("**How it works, step by step:**")
            lines.append("")
            for number, step in enumerate(steps, start=1):
                evidence = [Evidence(**e) for e in step.get("evidence", [])]
                description = clip(step.get("description"), sentences=1, max_chars=220)
                self._said.add(_key(description))  # so no takeaway repeats a step
                lines.append(
                    f"{number}. **{_esc(step.get('name'))}** — {self._prose(description)} "
                    f"{format_citations(evidence, limit=1)}".rstrip()
                )
        return "\n".join(lines).rstrip() if len(lines) > 2 else ""

    def _what_it_found(self, summary: dict, agent_results: dict[str, AgentResult]) -> str:
        lines = ["## What it found", ""]
        headline = self._fresh(clip(_stated(summary.get("main_results")), sentences=3, max_chars=550))
        if headline:
            lines.append(self._prose(headline, subject="results"))
            lines.append("")

        reported = self._payload(agent_results, AgentName.EXPERIMENTS).get("reported_results") or []
        rows = []
        for item in reported:
            evidence = [Evidence(**e) for e in item.get("evidence", [])]
            kind, _ = self._evidence_attribution(evidence)
            if kind == "cited" or not any(e.is_verified for e in evidence):
                continue  # the brief shows only this paper's own, grounded numbers
            rows.append((item, evidence))
        # The paper's best results first, then in the order it reports them.
        rows.sort(key=lambda pair: not pair[0].get("is_best"))
        if rows:
            shown = rows[:6]
            # A column no row fills is noise in a short table.
            with_dataset = any(item.get("dataset") for item, _ in shown)
            lines.append("| System |" + (" Dataset |" if with_dataset else "") + " Metric | Value | Where |")
            lines.append("| --- |" + (" --- |" if with_dataset else "") + " --- | --- | --- |")
            for item, evidence in shown:
                best = " ★" if item.get("is_best") else ""
                dataset = f" {_esc(item.get('dataset')) or '—'} |" if with_dataset else ""
                lines.append(
                    f"| {_esc(item.get('system')) or '—'} |{dataset} "
                    f"{_esc(item.get('metric'))} | **{_esc(item.get('value'))}**{best} "
                    f"| {format_citations(evidence, limit=1)} |"
                )
            if any(item.get("is_best") for item, _ in shown):
                lines.append("")
                lines.append("_★ the paper's best result._")
            if len(rows) > 6:
                lines.append("")
                lines.append(f"_{len(rows) - 6} more results are in the full report._")
        return "\n".join(lines).rstrip() if len(lines) > 2 else ""

    def _takeaways(self, findings: list[Finding]) -> str:
        grounded = [
            f
            for f in findings
            if f.status is FindingStatus.SUPPORTED and f.attribution != "cited" and f.has_evidence
        ]

        def rank(f: Finding) -> tuple[int, float]:
            category = (f.category or "").lower()
            order = _TAKEAWAY_ORDER.index(category) if category in _TAKEAWAY_ORDER else len(_TAKEAWAY_ORDER)
            return order, -f.confidence

        picked: list[Finding] = []
        seen_agents: dict[str, int] = {}
        for finding in sorted(grounded, key=rank):
            if self._already_said(finding.claim) or not _readable(finding.claim):
                continue
            # Spread the takeaways across agents rather than five from one.
            if seen_agents.get(finding.agent or "", 0) >= 2:
                continue
            seen_agents[finding.agent or ""] = seen_agents.get(finding.agent or "", 0) + 1
            picked.append(finding)
            if len(picked) == 5:
                break
        if not picked:
            return ""
        lines = ["## Key takeaways", ""]
        lines.extend(
            f"- {_esc(clip(f.claim, sentences=2, max_chars=260))} {format_citations(f.evidence, limit=1)}".rstrip()
            for f in picked
        )
        return "\n".join(lines)

    def _trust(
        self,
        summary: dict,
        agent_results: dict[str, AgentResult],
        findings: list[Finding],
        verification_results: list[VerificationResult],
    ) -> str:
        lines = ["## How far to trust it", ""]
        if verification_results:
            upheld = sum(
                1
                for r in verification_results
                if r.status in (FindingStatus.SUPPORTED, FindingStatus.PARTIALLY_SUPPORTED)
            )
            lines.append(
                f"- **Independent check:** {upheld} of {len(verification_results)} high-impact "
                "claims were upheld when re-checked against the paper."
            )

        experiments = self._payload(agent_results, AgentName.EXPERIMENTS)
        verdict = experiments.get("conclusions_supported")
        if verdict and verdict != FindingStatus.UNKNOWN.value:
            label = _STATUS_LABEL.get(FindingStatus(verdict), verdict) if verdict in {
                s.value for s in FindingStatus
            } else verdict
            reason = clip(experiments.get("conclusions_assessment"), sentences=1, max_chars=260)
            lines.append(
                f"- **Do the experiments back the conclusions?** {label}."
                + (f" {self._prose(reason)}" if reason else "")
            )

        limitations = clip(_stated(summary.get("limitations")), sentences=2, max_chars=380)
        if limitations:
            lines.append(
                f"- **Limitations the authors state:** {self._prose(limitations, subject='limitations')}"
            )

        weak = [f for f in findings if f.status in _WEAK]
        if weak:
            lines.append(
                f"- **Not confirmed:** {len(weak)} statement{'' if len(weak) == 1 else 's'} produced "
                "during the analysis could not be grounded in the paper and "
                f"{'is' if len(weak) == 1 else 'are'} left out of this brief. The full report "
                "lists them, so you can see what not to rely on."
            )
        lines.append(
            "- **Scope:** this reads only the PDF. It cannot tell whether the authors are right "
            "about the world, only what the paper says and where."
        )
        return "\n".join(lines)

    def _more(self) -> str:
        return (
            "---\n\n"
            "_Need more? The full report covers the method in detail, the mathematics, the "
            "datasets and setup, every reported result and ablation, a reproducibility "
            "checklist, and the evidence behind every statement._"
        )
