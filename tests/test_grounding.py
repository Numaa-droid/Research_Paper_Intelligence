"""Agents read Markdown answers; citations resolve to the index, not to the model."""

from __future__ import annotations

import re

from paperpilot.agents.experiments import ExperimentAgent
from paperpilot.agents.methodology import MethodologyAgent
from paperpilot.agents.summary import SummaryAgent
from paperpilot.agents.verification import VerificationAgent
from paperpilot.domain.enums import CitationQuality, FindingStatus
from paperpilot.domain.models import ClaimCandidate
from paperpilot.orchestration.planner import create_plan, read_planner_response
from paperpilot.providers.llm import LLMProvider, LLMResponse
from paperpilot.response import ResponseDocument


class ScriptedProvider(LLMProvider):
    """Builds its Markdown answer from the request, like a model reading the prompt."""

    name = "scripted"

    def __init__(self, write) -> None:
        self.write = write

    def generate(self, request):
        return LLMResponse(text=self.write(request), model="x", provider=self.name)


def _sentence(source: dict) -> str:
    """A verbatim sentence from a retrieved source."""
    text = " ".join(source["text"].split())
    return next((s for s in text.split(". ") if len(s) > 40), text[:120]).rstrip(".")


def test_agent_results_keep_the_markdown_and_its_parsed_document(llm, agent_context):
    result = SummaryAgent(llm).run(agent_context)

    assert result.response_markdown.startswith("## Research Problem")
    document = ResponseDocument.model_validate(result.response_document)
    assert [s.key for s in document.sections][:2] == ["research problem", "motivation"]
    assert result.payload["research_problem"]


def test_a_cited_label_supplies_page_chunk_and_section_from_the_index(agent_context):
    captured = {}

    def write(request):
        source = request.evidence[2]
        captured.update(source)
        return (
            "## Findings\n\n### Finding 1\n- Claim: A grounded claim.\n- Status: supported\n\n"
            f"> [{source['source_id']}] {_sentence(source)}\n"
        )

    result = SummaryAgent(ScriptedProvider(write)).run(agent_context)
    evidence = result.findings[0].evidence[0]
    chunk = agent_context.index.get(captured["chunk_id"])

    assert captured["source_id"] == "S3"
    assert (evidence.page, evidence.chunk_id) == (chunk.page, chunk.chunk_id)
    assert evidence.section == chunk.citation_section
    assert evidence.citation_quality in (CitationQuality.EXACT, CitationQuality.NORMALIZED)
    assert result.findings[0].status is FindingStatus.SUPPORTED


def test_a_wrong_label_is_recovered_by_locating_the_quote(agent_context):
    captured = {}

    def write(request):
        source = request.evidence[0]
        captured.update(source)
        return (
            "## Findings\n\n### Finding 1\n- Claim: Mislabelled claim.\n- Status: supported\n\n"
            f"> [S99] {_sentence(source)}\n"
        )

    result = SummaryAgent(ScriptedProvider(write)).run(agent_context)
    evidence = result.findings[0].evidence[0]

    assert evidence.chunk_id == captured["chunk_id"]
    assert evidence.is_verified
    assert any("S99" in note for note in evidence.validation_notes)


def test_a_loosely_formatted_answer_is_still_read(agent_context):
    """Bold pseudo-headings, trailing citations, prose statuses and word confidences."""

    def write(request):
        source = request.evidence[0]
        return (
            "Here is my analysis.\n\n"
            "**Research Problem**\n\nThe paper studies long documents.\n\n"
            "**Findings**\n\n"
            "#### 1. The method handles long inputs\n"
            "- **Status:** Partially supported.\n"
            "- **Confidence:** high\n\n"
            f'> "{_sentence(source)}" [{source["source_id"]}]\n'
        )

    result = SummaryAgent(ScriptedProvider(write)).run(agent_context)
    finding = result.findings[0]

    assert result.payload["research_problem"] == "The paper studies long documents."
    assert finding.claim == "The method handles long inputs"
    assert finding.evidence and finding.evidence[0].is_verified
    assert finding.status is FindingStatus.PARTIALLY_SUPPORTED


def test_findings_written_as_a_table_are_read(agent_context):
    def write(request):
        source = request.evidence[1]
        return (
            "## Findings\n\n"
            "| Claim | Status | Confidence | Source | Quote |\n"
            "| --- | --- | --- | --- | --- |\n"
            f"| Tabular claim | supported | 0.7 | [{source['source_id']}] | "
            f"{_sentence(source).replace('|', chr(92) + '|')} |\n"
        )

    result = SummaryAgent(ScriptedProvider(write)).run(agent_context)

    assert [f.claim for f in result.findings] == ["Tabular claim"]
    assert result.findings[0].evidence[0].is_verified


def test_an_answer_in_json_is_not_trusted_as_structure(agent_context):
    provider = ScriptedProvider(lambda r: '{"findings": [{"claim": "json claim"}]}')
    result = SummaryAgent(provider).run(agent_context)

    assert result.status == "ok"
    assert result.findings == []
    assert result.response_markdown.startswith("{")


def test_methodology_steps_come_from_step_subsections(agent_context):
    def write(request):
        source = request.evidence[0]
        return (
            "## Pipeline Steps\n\n"
            "### Step 2: Encode\n- Inferred: no\n\nChunks are encoded.\n\n"
            f"> [{source['source_id']}] {_sentence(source)}\n\n"
            "### Step 1: Split\n- Inferred: yes\n\nThe document is split.\n\n"
            "## Pseudocode\n```text\n1. split\n2. encode\n```\n"
        )

    result = MethodologyAgent(ScriptedProvider(write)).run(agent_context)
    steps = result.payload["pipeline_steps"]

    assert [(s["order"], s["name"], s["is_inferred"]) for s in steps] == [
        (1, "Split", True),
        (2, "Encode", False),
    ]
    assert steps[1]["evidence"][0]["citation_quality"] in ("exact", "normalized")
    assert result.payload["pseudocode"] == "1. split\n2. encode"


def test_a_reported_value_without_a_quote_is_grounded_in_its_cited_source_line(agent_context):
    captured = {}

    def write(request):
        decimal = re.compile(r"^\d+\.\d+$")
        table = next(
            e
            for e in request.evidence
            if e["element_type"] == "table" and any(decimal.match(t) for t in e["text"].split())
        )
        value = next(t for t in table["text"].split() if decimal.match(t))
        captured.update(value=value, chunk_id=table["chunk_id"])
        return (
            "## Reported Results\n\n| System | Metric | Value | Source | Quote |\n"
            "| --- | --- | --- | --- | --- |\n"
            f"| X | score | {value} | [{table['source_id']}] |  |\n"
        )

    result = ExperimentAgent(ScriptedProvider(write)).run(agent_context)
    reported = result.payload["reported_results"][0]

    assert reported["value"] == captured["value"]
    assert reported["evidence"][0]["chunk_id"] == captured["chunk_id"]
    assert captured["value"] in reported["evidence"][0]["quote"]
    assert result.payload["unverified_numbers"] == []


def test_verifier_reads_its_verdict_from_markdown(agent_context):
    def write(request):
        source = request.evidence[0]
        return (
            "## Verdict\n- Status: supported\n- Confidence: 0.9\n\n"
            "## Explanation\nThe passage states it.\n\n"
            f"## Supporting Evidence\n> [{source['source_id']}] {_sentence(source)}\n\n"
            "## Contradicting Evidence\nNone.\n"
        )

    candidate = ClaimCandidate(claim="routing long documents", source_agent="summary", impact=0.8)
    result = VerificationAgent(ScriptedProvider(write)).verify(candidate, agent_context)

    assert result.status is FindingStatus.SUPPORTED
    assert result.explanation == "The passage states it."
    assert result.supporting_evidence and not result.contradicting_evidence
    assert ResponseDocument.model_validate(result.response_document).sections


def test_planner_decisions_are_read_from_its_markdown_table():
    decisions = read_planner_response(
        "## Agents\n| Agent | Run | Reason |\n| --- | --- | --- |\n"
        "| Experiments agent | No | no results section |\n| math | yes | equations |\n\n"
        "## Priority Sections\n- Method\n"
    )
    assert decisions["run"] == {"experiments": False, "math": True}
    assert decisions["reasons"]["experiments"] == "no results section"
    assert decisions["priority_sections"] == ["Method"]


def test_the_planning_model_cannot_switch_agents_off(index, settings):
    """A model that sees only section names must not skip agents the index supports."""
    heuristic = create_plan(index, settings, use_model=False)
    veto = ScriptedProvider(
        lambda r: "## Agents\n| Agent | Run | Reason |\n| --- | --- | --- |\n"
        "| methodology | no | no method section |\n| experiments | no | no results |\n"
        "| math | no | not needed |\n"
    )
    plan = create_plan(index, settings, veto)

    assert (plan.run_methodology, plan.run_experiments, plan.run_math) == (
        heuristic.run_methodology,
        heuristic.run_experiments,
        heuristic.run_math,
    )
    assert plan.verification_threshold == settings.verification_threshold
    assert "suggested skipping" in plan.rationale["run_methodology"]


def test_an_unstructured_planner_answer_leaves_the_heuristic_plan_intact(index, settings):
    heuristic = create_plan(index, settings, use_model=False)
    plan = create_plan(index, settings, ScriptedProvider(lambda r: "I think you should run it all."))

    assert (plan.run_experiments, plan.run_math, plan.max_retries) == (
        heuristic.run_experiments,
        heuristic.run_math,
        heuristic.max_retries,
    )


# --- structure-anchored context ---------------------------------------------


def test_agents_always_read_the_sections_their_task_is_about(agent_context, llm):
    from paperpilot.agents.math_agent import MathAgent
    from paperpilot.domain.enums import ElementType

    experiments = ExperimentAgent(llm).retrieve(agent_context)
    anchored = [r for r in experiments if r.matched_by == ["structure"]]
    sections = {r.chunk.citation_section for r in anchored}

    assert {"4.1 Setup", "4.2 Main Results", "4.3 Ablations"} <= sections
    assert any(r.chunk.element_type.value == "table" for r in anchored)
    assert experiments[: len(anchored)] == anchored, "anchors come first so budgets keep them"

    equations = agent_context.index.by_element_type(ElementType.EQUATION)
    math_ids = {r.chunk_id for r in MathAgent(llm).retrieve(agent_context)}
    assert {c.chunk_id for c in equations} <= math_ids


def test_the_bibliography_never_fills_an_agent_context(agent_context, llm):
    results = ExperimentAgent(llm).retrieve(agent_context)
    assert not any((r.chunk.section or "").lower().startswith("reference") for r in results)


# --- equations come from the index ------------------------------------------


def test_a_cited_equation_takes_its_number_page_and_text_from_the_index(agent_context):
    from paperpilot.agents.math_agent import MathAgent

    captured = {}

    def write(request):
        source = next(e for e in request.evidence if e["element_type"] == "equation")
        captured.update(source)
        return (
            f"## Equations\n\n### Equation 7\n- Source: [{source['source_id']}]\n"
            "- Representation: unavailable\n- Purpose: routing objective\n\n"
            "```text\n(unavailable)\n```\n"
        )

    result = MathAgent(ScriptedProvider(write), min_equations=1).run(agent_context)
    chunk = agent_context.index.get(captured["chunk_id"])
    cited = next(e for e in result.payload["equations"] if e["purpose"] == "routing objective")

    assert cited["source_text"] == chunk.text
    assert cited["page"] == chunk.page
    assert cited["representation"] == "extracted_text"
    assert cited["equation_id"].startswith(("Equation (", "Unnumbered equation"))
    assert cited["evidence"] and cited["evidence"][0]["citation_quality"] == "exact"


def test_equations_the_model_skipped_are_listed_verbatim(agent_context):
    from paperpilot.agents.math_agent import MathAgent
    from paperpilot.domain.enums import ElementType

    result = MathAgent(ScriptedProvider(lambda r: "## Equations\nNot stated.\n"), min_equations=1).run(
        agent_context
    )
    detected = agent_context.index.by_element_type(ElementType.EQUATION)

    assert len(result.payload["equations"]) == len(detected)
    assert {e["source_text"] for e in result.payload["equations"]} == {c.text for c in detected}
    assert sum("not explained" in q for q in result.open_questions) == len(detected)


# --- experiments and methodology readers ------------------------------------


def test_a_reported_result_without_a_number_is_left_out(agent_context):
    def write(request):
        return (
            "## Reported Results\n\n| System | Metric | Value | Source | Quote |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| SRT-Large | F1 | Not numeric, hypothesized highest | [S1] |  |\n"
        )

    result = ExperimentAgent(ScriptedProvider(write)).run(agent_context)

    assert result.payload["reported_results"] == []
    # A pipeline note, not a question the paper leaves open.
    assert any("SRT-Large F1" in d for d in result.payload["diagnostics"])
    assert not any("SRT-Large F1" in q for q in result.open_questions)


def test_a_step_description_written_as_a_bullet_is_kept(agent_context):
    def write(request):
        return (
            "## Pipeline Steps\n\n### Step 1: Ingest\n- Inferred: no\n"
            "- Documents are parsed page by page.\n"
        )

    steps = MethodologyAgent(ScriptedProvider(write)).run(agent_context).payload["pipeline_steps"]
    assert (steps[0]["name"], steps[0]["description"], steps[0]["is_inferred"]) == (
        "Ingest",
        "Documents are parsed page by page.",
        False,
    )
