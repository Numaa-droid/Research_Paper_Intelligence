"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .. import __version__
from ..config import Settings, get_settings
from ..observability import configure_logging, info
from ..service import PaperPilotService
from .routes import register_error_handlers, router
from .tasks import TaskRunner, ThreadPoolTaskRunner

DESCRIPTION = """\
**PaperPilot** analyses a single research-paper PDF and returns a citation-grounded
Markdown report.

Every factual statement in a report is either backed by a quote that was
re-checked against the original PDF, or explicitly labelled as an inference, an
uncertainty, or an unsupported claim.

PaperPilot is a domain-specific application. It is not general intelligence, it
knows nothing about a paper beyond the PDF you submit, and it verifies claims only
against that paper's own text.

**Typical flow**

1. `POST /papers` — upload a PDF, receive a `document_id`.
2. `POST /papers/{document_id}/analyze` — start an analysis, receive an `analysis_id`.
3. `GET /analyses/{analysis_id}` — poll for progress.
4. `GET /analyses/{analysis_id}/report` — fetch the Markdown report.
"""


def create_app(
    settings: Settings | None = None,
    *,
    service: PaperPilotService | None = None,
    task_runner: TaskRunner | None = None,
) -> FastAPI:
    """Build the application. Dependencies are injectable so tests stay hermetic."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        info(
            "api.startup",
            version=__version__,
            llm_provider=settings.llm_provider,
            embedding_provider=settings.embedding_provider,
        )
        yield
        app.state.task_runner.shutdown()
        info("api.shutdown")

    app = FastAPI(
        title="PaperPilot",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        contact={"name": "PaperPilot"},
    )
    app.state.settings = settings
    app.state.service = service or PaperPilotService(settings)
    app.state.task_runner = task_runner or ThreadPoolTaskRunner(max_workers=2)

    app.include_router(router)
    register_error_handlers(app)
    return app


#: Module-level app for ``uvicorn paperpilot.api.app:app``.
app = create_app()
