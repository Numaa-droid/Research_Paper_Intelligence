"""Typed error hierarchy.

Errors carry a stable ``code`` so the API and CLI can map failures to exit codes
and HTTP statuses without string matching.
"""

from __future__ import annotations


class PaperPilotError(Exception):
    """Base class for every error raised by PaperPilot."""

    code = "paperpilot_error"
    http_status = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigurationError(PaperPilotError):
    code = "configuration_error"


class ValidationError(PaperPilotError):
    """Deterministic validation failed (file type, size, schema, citation range)."""

    code = "validation_error"
    http_status = 400


class UnsupportedFileError(ValidationError):
    code = "unsupported_file"
    http_status = 415


class FileTooLargeError(ValidationError):
    code = "file_too_large"
    http_status = 413


class IngestionError(PaperPilotError):
    code = "ingestion_error"
    http_status = 422


class RetrievalError(PaperPilotError):
    code = "retrieval_error"


class ProviderError(PaperPilotError):
    """The model provider failed or returned output that could not be parsed."""

    code = "provider_error"
    http_status = 502


class AgentError(PaperPilotError):
    code = "agent_error"


class NotFoundError(PaperPilotError):
    code = "not_found"
    http_status = 404
