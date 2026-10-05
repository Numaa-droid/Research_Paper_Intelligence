"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .. import __version__
from ..config import Settings, get_settings
from ..observability import configure_logging, info, warning
from ..service import PaperPilotService
from .guard import RateLimiter, SessionMiddleware, UploadSizeMiddleware
from .routes import register_error_handlers, router
from .tasks import TaskRunner, ThreadPoolTaskRunner

DESCRIPTION = """\
**PaperMind** analyses a single research-paper PDF and returns a citation-grounded
Markdown report.

Every factual statement in a report is either backed by a quote that was
re-checked against the original PDF, or explicitly labelled as an inference, an
uncertainty, or an unsupported claim.

PaperMind is a domain-specific application. It is not general intelligence, it
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
    configure_logging(settings.log_level, settings.log_format, http=settings.log_http)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        info(
            "api.startup",
            version=__version__,
            llm_provider=settings.llm_provider,
            llm_model=None if settings.llm_provider == "mock" else settings.llm_model,
            embedding_provider=settings.embedding_provider,
            invite_required=bool(settings.invite_code),
        )
        if not settings.invite_code:
            warning("api.open_signup", setting="PAPERPILOT_INVITE_CODE")
        yield
        app.state.task_runner.shutdown()
        info("api.shutdown")

    app = FastAPI(
        title="PaperMind",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        contact={"name": "PaperMind"},
    )
    app.state.settings = settings
    app.state.service = service or PaperPilotService(settings)
    app.state.task_runner = task_runner or ThreadPoolTaskRunner(max_workers=2)
    app.state.rate_limiter = (
        RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)
        if settings.rate_limit_requests
        else None
    )
    #: Sign-in and sign-up attempts per client address per 15 minutes.
    app.state.auth_limiter = RateLimiter(20, 900)

    app.include_router(router)
    register_error_handlers(app)
    # The last middleware added runs first: the session is checked before any
    # of the body is read.
    app.add_middleware(UploadSizeMiddleware, max_bytes=settings.max_upload_bytes)
    app.add_middleware(SessionMiddleware, accounts=app.state.service.accounts)
    return app


#: Module-level app for ``uvicorn paperpilot.api.app:app``.
app = create_app()
