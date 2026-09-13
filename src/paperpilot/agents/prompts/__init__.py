"""Versioned prompts.

Every prompt carries an explicit version string which is recorded in traces
alongside the agent name and model, so a change in output quality can be
attributed to a prompt change rather than guessed at.
"""

from .registry import PROMPTS, PromptTemplate, get_prompt

__all__ = ["PROMPTS", "PromptTemplate", "get_prompt"]
