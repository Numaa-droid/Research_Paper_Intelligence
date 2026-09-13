"""Analysis planning.

The Orchestrator does not blindly run every agent. It inspects what the index
actually contains — sections, tables, figures, equations, page count — and
decides which specialists have something to work on.

The plan is produced by a deterministic heuristic first. A model may then refine
it, but only within limits: it can turn an agent *off* when the heuristic was
generous, and it can reorder priority sections, but it cannot turn an agent *on*
for content that does not exist in the index. Structure is a fact about the
document, not an opinion.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..domain.enums import ElementType
from ..domain.models import AnalysisPlan, PaperMetadata
from ..errors import ProviderError
from ..observability import info, warning
from ..providers.llm import LLMProvider, LLMRequest
from ..retrieval.index import PaperIndex
from .prompt_bridge import planner_prompt

_METHOD_HINTS = ("method", "approach", "model", "architecture", "framework", "algorithm", "system")
_EXPERIMENT_HINTS = ("experiment", "result", "evaluation", "empirical", "study", "benchmark")


def paper_structure(index: PaperIndex, settings: Settings) -> dict[str, Any]:
    """The structural facts the planner is allowed to reason over."""
    metadata = index.metadata
    return {
        "title": metadata.title,
        "page_count": metadata.page_count,
        "chunk_count": metadata.chunk_count,
        "ocr_page_count": metadata.ocr_page_count,
        "sections": index.sections(),
        "equation_count": len(index.by_element_type(ElementType.EQUATION)),
        "table_count": len(index.by_element_type(ElementType.TABLE)),
        "figure_count": len(index.by_element_type(ElementType.FIGURE)),
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
    has_experiments = has(_EXPERIMENT_HINTS) or structure["table_count"] > 0
    equations = structure["equation_count"]
    run_math = equations >= settings.math_min_equations

    rationale = {
        "run_summary": "the summary agent is the grounded baseline and always runs",
        "run_methodology": (
            "a method/approach section was detected"
            if has_method
            else "no explicit method section was detected; running anyway to describe the "
            "approach from the body text"
        ),
        "run_experiments": (
            f"{structure['table_count']} table(s) and an experiments/results section were detected"
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
        prompt_version="v1",
        context={"structure": structure},
    )
    try:
        payload = provider.generate_json(request).content
    except ProviderError as exc:
        warning("planner.model_failed", error=str(exc)[:200])
        return base

    plan = base.model_copy(deep=True)
    plan.created_by = "hybrid"

    # The model may disable an agent, but may not enable one for absent content.
    if payload.get("run_experiments") is False:
        plan.run_experiments = False
        plan.rationale["run_experiments"] = "disabled by the planning model"
    if payload.get("run_methodology") is False:
        plan.run_methodology = False
        plan.rationale["run_methodology"] = "disabled by the planning model"
    if payload.get("run_math") is False and plan.run_math:
        plan.run_math = False
        plan.rationale["run_math"] = "disabled by the planning model"
    elif payload.get("run_math") and not plan.run_math:
        plan.rationale["run_math"] = (
            f"the planning model asked for the math agent, but only "
            f"{structure['equation_count']} equation(s) exist in the index; request declined"
        )

    known = {s.lower(): s for s in structure["sections"]}
    requested = [str(s) for s in (payload.get("priority_sections") or [])]
    resolved = [known[s.lower()] for s in requested if s.lower() in known]
    if resolved:
        plan.priority_sections = resolved[:8]

    threshold = payload.get("verification_threshold")
    if isinstance(threshold, int | float) and 0.0 <= float(threshold) <= 1.0:
        plan.verification_threshold = float(threshold)

    retries = payload.get("max_retries")
    if isinstance(retries, int) and 0 <= retries <= settings.max_retries:
        plan.max_retries = retries

    for key, value in (payload.get("rationale") or {}).items():
        if isinstance(value, str) and key not in plan.rationale:
            plan.rationale[str(key)] = value

    return plan


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
