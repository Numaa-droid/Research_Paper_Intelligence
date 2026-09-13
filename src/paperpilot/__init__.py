"""PaperPilot: a citation-grounded, multi-agent research-paper analysis system.

PaperPilot is a *domain-specific* application. It is not general intelligence and
makes no such claim: it parses one research paper, indexes it with page-aware
metadata, plans an analysis, runs specialist agents over targeted evidence,
verifies high-impact claims against the original text, and emits a Markdown
report in which every factual statement is either cited or explicitly marked as
an inference, an uncertainty, or an unsupported claim.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
