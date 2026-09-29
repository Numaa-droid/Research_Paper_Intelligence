"""Analysis planning.

The Orchestrator does not blindly run every agent. It inspects what the index
actually contains — sections, tables, figures, equations, page count — and
decides which specialists have something to work on.

Which agents run, the verification threshold and the retry budget are decided
deterministically from the index and the settings. A model may reorder the
priority sections (only to sections that exist), and its run/skip opinions are
recorded in the rationale, but it can never switch an agent on or off: it sees
only section names and counts, and a garbled outline — a PDF whose headings did
not extract — would otherwise let it skip the methodology of a paper it has not
read. Structure is a fact about the document, not an opinion.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..discourse.profile import profile_paper
from ..domain.enums import ElementType
from ..domain.models import AnalysisPlan, PaperMetadata
from ..errors import ProviderError
from ..ingestion.chunker import element_counts
from ..observability import info, warning
from ..providers.llm import LLMProvider, LLMRequest
from ..response import normalize_key, parse_markdown
from ..response.reader import DocumentReader, parse_bool
from ..retrieval.index import PaperIndex
from .prompt_bridge import planner_prompt

_METHOD_HINTS = ("method", "approach", "model", "architecture", "framework", "algorithm", "system")
_EXPERIMENT_HINTS = ("experiment", "result", "evaluation", "empirical", "study", "benchmark")


def paper_structure(index: PaperIndex, settings: Settings) -> dict[str, Any]:
    """The structural facts the planner is allowed to reason over."""
    metadata = index.metadata
    counts = element_counts(index.chunks)
    return {
        "title": metadata.title,
        "page_count": metadata.page_count,
        "chunk_count": metadata.chunk_count,
        "ocr_page_count": metadata.ocr_page_count,
        "sections": index.sections(),
        # Elements, not chunks: a table split into caption and body is one table.
        "equation_count": counts.get(ElementType.EQUATION.value, 0),
        "table_count": counts.get(ElementType.TABLE.value, 0),
        "figure_count": counts.get(ElementType.FIGURE.value, 0),
        "reference_count": len(index.by_element_type(ElementType.REFERENCE)),
        "math_min_equations": settings.math_min_equations,
    }


def heuristic_plan(index: PaperIndex, settings: Settings) -> AnalysisPlan:
    """Build the baseline plan from the index alone. Never fails."""
    structure = paper_structure(index, settings)
    sections = [s.lower() for s in structure["sections"]]

    def has(hints: tuple[str, ...]) -> bool:
        return any(hint in section for section in sections for hint in hints)

    has_method = has(_METHOD_HINTS)
    experiment_headings = [
        s for s in structure["sections"] if any(h in s.lower() for h in _EXPERIMENT_HINTS)
    ]
    has_experiments = bool(experiment_headings) or structure["table_count"] > 0
    equations = structure["equation_count"]
    run_math = equations >= settings.math_min_equations
    profile = profile_paper(index)

    experiment_signals = []
    if experiment_headings:
        experiment_signals.append(
            "experiment/result heading(s): " + ", ".join(experiment_headings[:3])
        )
    if structure["table_count"]:
        experiment_signals.append(f"{structure['table_count']} table(s) to read")

    rationale = {
        "paper_kind": f"{profile.kind}: " + "; ".join(profile.signals),
        "run_summary": "the summary agent is the grounded baseline and always runs",
        "run_methodology": (
            "a method/approach section was detected"
            if has_method
            else "no explicit method section was detected; running anyway to describe the "
            "approach from the body text"
        ),
        "run_experiments": (
            " and ".join(experiment_signals) + " detected"
            if has_experiments
            else "no experiments section or result tables were detected"
        ),
        "run_math": (
            f"{equations} equation(s) detected, at or above the threshold of "
            f"{settings.math_min_equations}"
            if run_math
            else f"only {equations} equation(s) detected, below the threshold of "
            f"{settings.math_min_equations}"
        ),
        "run_table_analysis": f"{structure['table_count']} table(s) available to analyse",
    }
    if structure["ocr_page_count"]:
        rationale["ocr"] = (
            f"{structure['ocr_page_count']} page(s) came from OCR; quote matching will be "
            "approximate on those pages"
        )

    priority = [
        section
        for section in structure["sections"]
        if not section.lower().startswith(("reference", "bibliograph", "acknowledg", "front matter"))
    ][:8]

    return AnalysisPlan(
        run_summary=True,
        run_methodology=True,
        run_experiments=has_experiments,
        run_math=run_math,
        run_table_analysis=structure["table_count"] > 0,
        priority_sections=priority,
        verification_threshold=settings.verification_threshold,
        max_retries=settings.max_retries,
        rationale=rationale,
        created_by="heuristic",
        paper_kind=profile.kind,
        paper_kind_signals=profile.signals,
    )


def refine_plan_with_model(
    base: AnalysisPlan,
    index: PaperIndex,
    settings: Settings,
    provider: LLMProvider,
) -> AnalysisPlan:
    """Let the model adjust the heuristic plan, within hard structural limits."""
    structure = paper_structure(index, settings)
    system, user = planner_prompt(structure)
    request = LLMRequest(
        agent="planner",
        task="planner",
        system=system,
        user=user,
        prompt_version="v2",
        context={"structure": structure},
    )
    try:
        text = provider.generate(request).text
    except ProviderError as exc:
        warning("planner.model_failed", error=str(exc)[:200])
        return base
    decisions = read_planner_response(text)

    plan = base.model_copy(deep=True)
    plan.created_by = "hybrid"

    runs, reasons = decisions["run"], decisions["reasons"]
    for agent in ("methodology", "experiments", "math", "table_analysis"):
        key = f"run_{agent}"
        suggestion = runs.get(agent)
        if suggestion is None or suggestion == getattr(plan, key):
            continue
        kept = "run" if getattr(plan, key) else "skip"
        why = f": {reasons[agent]}" if reasons.get(agent) else ""
        plan.rationale[key] = (
            f"{plan.rationale.get(key, 'decided from the paper structure')} "
            f"(the planning model suggested {'running' if suggestion else 'skipping'} it{why}; "
            f"kept '{kept}' because agent selection follows the index, not the model)"
        )

    known = {s.lower(): s for s in structure["sections"]}
    resolved = [known[s.lower()] for s in decisions["priority_sections"] if s.lower() in known]
    if resolved:
        plan.priority_sections = resolved[:8]

    return plan


def read_planner_response(text: str) -> dict[str, Any]:
    """Planner decisions from its Markdown answer; absent values are ``None``/empty."""
    reader = DocumentReader(parse_markdown(text))
    run: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for table in reader.tables(reader.section("agents", "agent plan")):
        for record in reader.records(
            table, {"agent": ("name",), "run": ("enabled", "decision"), "reason": ("why", "rationale")}
        ):
            agent_cell = record["agent"]
            if agent_cell is None:
                continue
            agent = normalize_key(agent_cell.value).replace(" agent", "").replace(" ", "_")
            decision = parse_bool(record["run"].value) if record["run"] is not None else None
            if agent and decision is not None:
                run[agent] = decision
            if agent and record["reason"] is not None and record["reason"].value:
                reasons[agent] = record["reason"].value

    return {
        "run": run,
        "reasons": reasons,
        "priority_sections": reader.items(reader.section("priority sections")),
    }


def create_plan(
    index: PaperIndex,
    settings: Settings,
    provider: LLMProvider | None = None,
    *,
    use_model: bool = True,
) -> AnalysisPlan:
    """Create the analysis plan for one paper."""
    plan = heuristic_plan(index, settings)
    if provider is not None and use_model:
        plan = refine_plan_with_model(plan, index, settings, provider)
    info(
        "planner.plan_created",
        created_by=plan.created_by,
        agents=plan.enabled_agents(),
        priority_sections=plan.priority_sections[:5],
        verification_threshold=plan.verification_threshold,
        max_retries=plan.max_retries,
    )
    return plan


def plan_for_metadata(metadata: PaperMetadata, settings: Settings) -> AnalysisPlan:
    """Plan from metadata alone; used where the index is not available."""
    return AnalysisPlan(
        run_summary=True,
        run_methodology=True,
        run_experiments=metadata.table_count > 0,
        run_math=metadata.equation_count >= settings.math_min_equations,
        run_table_analysis=metadata.table_count > 0,
        priority_sections=metadata.sections[:8],
        verification_threshold=settings.verification_threshold,
        max_retries=settings.max_retries,
        created_by="heuristic",
    )
