"""Specialist agents. Every one returns validated Pydantic models."""

from .base import AgentContext, BaseAgent
from .experiments import ExperimentAgent
from .math_agent import MathAgent, detect_math_content
from .methodology import MethodologyAgent
from .summary import SummaryAgent
from .verification import (
    VerificationAgent,
    apply_verification,
    extract_high_impact_claims,
)

#: Specialist agents keyed by the name the planner uses.
AGENT_REGISTRY = {
    "summary": SummaryAgent,
    "methodology": MethodologyAgent,
    "experiments": ExperimentAgent,
    "math": MathAgent,
}

__all__ = [
    "AGENT_REGISTRY",
    "AgentContext",
    "BaseAgent",
    "ExperimentAgent",
    "MathAgent",
    "MethodologyAgent",
    "SummaryAgent",
    "VerificationAgent",
    "apply_verification",
    "detect_math_content",
    "extract_high_impact_claims",
]
