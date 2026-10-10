"""HTTP routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse

from .. import __version__
from ..accounts import MIN_PASSWORD, SESSION_DAYS
from ..domain.enums import AnalysisStatus
from ..errors import (
    AuthenticationError,
    NotFoundError,
    PaperPilotError,
    RateLimitError,
    UnsupportedFileError,
)
from ..observability import info
from ..response import SCHEMA_VERSION, response_document_json_schema
from ..service import PaperPilotService
from ..storage.repository import User
from .guard import SESSION_COOKIE, auth_rate_limited, rate_limited, session_token
from .schemas import (
    AnalysisResponse,
    AnalysisSummary,
    AnalyzeRequest,
    AuthConfigResponse,
    AuthResponse,
    ChangePasswordRequest,
    DocumentResponse,
    HealthResponse,
    LoginRequest,
    SignUpRequest,
    UploadResponse,
    UserResponse,
)

router = APIRouter()

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def get_service(request: Request) -> PaperPilotService:
    return request.app.state.service


def get_runner(request: Request):
    return request.app.state.task_runner


def current_user(request: Request) -> User:
    """The signed-in account; ``SessionMiddleware`` already refused anonymous calls."""
    user = getattr(request.state, "user", None)
    if user is None:
        raise AuthenticationError("please sign in")
    return user


def _user_response(user: User) -> UserResponse:
    return UserResponse(user_id=user.user_id, email=user.email, name=user.name, created_at=user.created_at)


def _signed_in(request: Request, response: Response, user: User, token: str) -> AuthResponse:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_DAYS * 86400,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )
    return AuthResponse(user=_user_response(user), token=token)


# --- accounts ----------------------------------------------------------------


@router.get("/auth/config", response_model=AuthConfigResponse, tags=["auth"])
def auth_config(service: PaperPilotService = Depends(get_service)) -> AuthConfigResponse:
    """What the sign-up form must ask for."""
    return AuthConfigResponse(
        invite_required=service.accounts.invite_required, min_password_length=MIN_PASSWORD
    )


@router.post(
    "/auth/signup",
    response_model=AuthResponse,
    status_code=201,
    tags=["auth"],
    dependencies=[Depends(auth_rate_limited)],
)
def sign_up(
    payload: SignUpRequest,
    request: Request,
    response: Response,
    service: PaperPilotService = Depends(get_service),
) -> AuthResponse:
    """Create an account (with the invite code, when one is configured) and sign in."""
    user, token = service.accounts.sign_up(
        name=payload.name, email=payload.email, password=payload.password, invite_code=payload.invite_code
    )
    return _signed_in(request, response, user, token)


@router.post(
    "/auth/login", response_model=AuthResponse, tags=["auth"], dependencies=[Depends(auth_rate_limited)]
)
def log_in(
    payload: LoginRequest,
    request: Request,
    response: Response,
    service: PaperPilotService = Depends(get_service),
) -> AuthResponse:
    user, token = service.accounts.log_in(email=payload.email, password=payload.password)
    return _signed_in(request, response, user, token)


@router.post("/auth/logout", status_code=204, tags=["auth"])
def log_out(request: Request, service: PaperPilotService = Depends(get_service)) -> Response:
    token = session_token(request.scope)
    if token:
        service.accounts.log_out(token)
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@router.get("/auth/me", response_model=UserResponse, tags=["auth"])
def me(user: User = Depends(current_user)) -> UserResponse:
    return _user_response(user)


@router.post(
    "/auth/password", status_code=204, tags=["auth"], dependencies=[Depends(auth_rate_limited)]
)
def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
) -> Response:
    """Change the password; every other session of the account is signed out."""
    service.accounts.change_password(
        user,
        current=payload.current_password,
        new=payload.new_password,
        token=session_token(request.scope) or "",
    )
    return Response(status_code=204)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: PaperPilotService = Depends(get_service)) -> HealthResponse:
    """Liveness plus the configuration facts a caller needs to interpret results."""
    from ..ingestion.ocr import build_ocr_engine

    settings = service.settings
    ocr = build_ocr_engine(**settings.ocr_engine_options())

    warnings: list[str] = []
    if not settings.invite_code:
        warnings.append(
            "No invite code is configured: anyone who can reach this server can create an "
            "account and spend the model budget. Set PAPERPILOT_INVITE_CODE before exposing it."
        )
    if settings.llm_provider == "mock":
        warnings.append(
            "The deterministic mock provider is active: the pipeline runs end to end, "
            "but reports are not a real analysis. Set PAPERPILOT_LLM_PROVIDER."
        )
    if settings.ocr_enabled and not ocr.available:
        warnings.append(
            f"OCR is enabled but the PaddleOCR-VL server at {settings.ocr_base_url} is not "
            "reachable; pages that fail the extraction check will be indexed with whatever "
            "native text exists."
        )
    if settings.embedding_provider == "hashing":
        warnings.append(
            "The offline hashing embedder is active; dense retrieval quality is limited."
        )

    return HealthResponse(
        version=__version__,
        llm_provider=settings.llm_provider,
        llm_model=service.llm.model,
        embedding_provider=settings.embedding_provider,
        ocr_available=ocr.available,
        warnings=warnings,
    )


@router.post(
    "/papers",
    response_model=UploadResponse,
    status_code=201,
    tags=["papers"],
    dependencies=[Depends(rate_limited)],
)
async def upload_paper(
    file: UploadFile = File(..., description="A research-paper PDF"),
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
) -> UploadResponse:
    """Accept one PDF and return its document identifier."""
    if file.content_type and file.content_type not in (
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    ):
        raise UnsupportedFileError(
            f"expected a PDF upload, got content type '{file.content_type}'"
        )

    metadata = service.register_upload(
        file.file, filename=file.filename or "upload.pdf", owner_id=user.user_id
    )
    info("api.upload", document_id=metadata.document_id, bytes=metadata.file_size_bytes)
    return UploadResponse(
        document_id=metadata.document_id,
        filename=metadata.filename,
        file_size_bytes=metadata.file_size_bytes,
    )


@router.get("/papers", response_model=list[DocumentResponse], tags=["papers"])
def list_papers(
    limit: int = 50,
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
) -> list[DocumentResponse]:
    return [
        DocumentResponse.from_metadata(m)
        for m in service.list_documents(limit=limit, owner_id=user.user_id)
    ]


@router.get("/papers/{document_id}", response_model=DocumentResponse, tags=["papers"])
def get_paper(
    document_id: str,
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
) -> DocumentResponse:
    return DocumentResponse.from_metadata(service.get_document(document_id, owner_id=user.user_id))


@router.post(
    "/papers/{document_id}/analyze",
    response_model=AnalysisResponse,
    status_code=202,
    tags=["analyses"],
    dependencies=[Depends(rate_limited)],
)
def analyze_paper(
    document_id: str,
    payload: AnalyzeRequest | None = None,
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
    runner=Depends(get_runner),
) -> AnalysisResponse:
    """Start an analysis.

    Returns ``202`` with an analysis id immediately; poll
    ``GET /analyses/{analysis_id}`` for progress. Pass ``{"wait": true}`` to block
    until the run finishes (small papers only).
    """
    record = service.create_analysis(document_id, owner_id=user.user_id)
    options = payload or AnalyzeRequest()

    if options.wait:
        completed = service.run_analysis(record.analysis_id)
        return AnalysisResponse.from_record(completed)

    runner.submit(record.analysis_id, lambda: service.run_analysis(record.analysis_id))
    # Re-read: a synchronous runner (tests, single-worker deployments) has already
    # finished by now, and returning the stale record would report "pending".
    return AnalysisResponse.from_record(service.get_analysis(record.analysis_id))


@router.get("/analyses", response_model=list[AnalysisSummary], tags=["analyses"])
def list_analyses(
    limit: int = 50,
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
) -> list[AnalysisSummary]:
    """The account's analyses, newest first, with headline numbers for each."""
    titles = {
        m.document_id: m.title or m.filename
        for m in service.list_documents(limit=1000, owner_id=user.user_id)
    }
    return [
        AnalysisSummary.from_record(r, titles.get(r.document_id))
        for r in service.list_analyses(owner_id=user.user_id, limit=limit)
    ]


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse, tags=["analyses"])
def get_analysis(
    analysis_id: str,
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
) -> AnalysisResponse:
    """Poll an analysis. Safe to call as often as a client likes."""
    return AnalysisResponse.from_record(service.get_analysis(analysis_id, owner_id=user.user_id))


@router.get("/analyses/{analysis_id}/report", tags=["analyses"])
def get_report(
    analysis_id: str,
    format: str = "markdown",
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
):
    """Return the final Markdown report."""
    record = service.get_analysis(analysis_id, owner_id=user.user_id)
    if record.status is not AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "report_not_ready",
                "message": f"analysis is '{record.status.value}', not completed",
                "details": {"status": record.status.value},
            },
        )
    markdown = service.get_report(analysis_id, owner_id=user.user_id)
    if format == "json":
        return record.model_dump(mode="json")
    if format == "pdf":
        return _pdf_response(service, record, markdown, kind="report")
    return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8")


@router.get("/analyses/{analysis_id}/brief", tags=["analyses"])
def get_brief(
    analysis_id: str,
    format: str = "markdown",
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
):
    """The two-page brief: what to read before the full report.

    ``format=pdf`` returns it as a PDF. Analyses made before briefs existed have
    none (``404``); their full report is unaffected.
    """
    record = service.get_analysis(analysis_id, owner_id=user.user_id)
    if record.status is not AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "report_not_ready",
                "message": f"analysis is '{record.status.value}', not completed",
                "details": {"status": record.status.value},
            },
        )
    if not record.brief_markdown:
        raise NotFoundError(
            f"analysis '{analysis_id}' has no brief; it was made before briefs existed. "
            "Run the paper again to get one."
        )
    if format == "pdf":
        return _pdf_response(service, record, record.brief_markdown, kind="brief")
    return PlainTextResponse(record.brief_markdown, media_type="text/markdown; charset=utf-8")


def _pdf_response(service: PaperPilotService, record, markdown: str, *, kind: str) -> Response:
    from ..reporting.pdf import render_report_pdf, report_filename

    metadata = service.documents.get(record.document_id)
    title = (metadata.title or metadata.filename) if metadata else None
    filename = report_filename(record.analysis_id, title)
    if kind == "brief":
        filename = filename.removesuffix(".pdf") + "-brief.pdf"
    return Response(
        render_report_pdf(markdown, title=title, kind=kind),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/analyses/{analysis_id}/findings", tags=["analyses"])
def get_findings(
    analysis_id: str,
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
) -> dict:
    """Structured findings and verification verdicts, for programmatic use."""
    record = service.get_analysis(analysis_id, owner_id=user.user_id)
    return {
        "analysis_id": record.analysis_id,
        "status": record.status.value,
        "findings": [f.model_dump(mode="json") for f in record.findings],
        "verification_results": [
            v.model_dump(mode="json", exclude=_RESPONSE_FIELDS)
            for v in record.verification_results
        ],
        "stats": record.stats(),
    }


_RESPONSE_FIELDS = {"response_markdown", "response_document"}


@router.get("/analyses/{analysis_id}/responses", tags=["analyses"])
def get_model_responses(
    analysis_id: str,
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
) -> dict:
    """Each model answer as written (Markdown) and as its parsed response document."""
    record = service.get_analysis(analysis_id, owner_id=user.user_id)
    return {
        "analysis_id": record.analysis_id,
        "schema_version": SCHEMA_VERSION,
        "agents": {
            name: {
                "status": result.status,
                "attempt": result.attempt,
                "response_markdown": result.response_markdown,
                "response_document": result.response_document,
            }
            for name, result in record.agent_results.items()
        },
        "verification": [
            {
                "claim": v.claim,
                "status": v.status.value,
                "response_markdown": v.response_markdown,
                "response_document": v.response_document,
            }
            for v in record.verification_results
        ],
    }


# --- traces ------------------------------------------------------------------


@router.get("/traces", tags=["traces"])
def list_traces(
    limit: int = 50,
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
) -> list[dict]:
    """The account's run traces, newest first, with headline numbers for each."""
    return service.list_traces(owner_id=user.user_id, limit=min(max(limit, 1), 200))


@router.get("/traces/{trace_id}", tags=["traces"])
def get_trace(
    trace_id: str,
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
) -> dict:
    """One run: every span with its inputs, outputs, timing, tokens and events.

    A running trace is returned as it stands, so polling shows it grow.
    """
    return service.get_trace(trace_id, owner_id=user.user_id)


@router.get("/analyses/{analysis_id}/trace", tags=["traces"])
def get_analysis_trace(
    analysis_id: str,
    user: User = Depends(current_user),
    service: PaperPilotService = Depends(get_service),
) -> dict:
    """The most recent trace of an analysis."""
    return service.trace_for_analysis(analysis_id, owner_id=user.user_id)


@router.get("/tracing", response_class=HTMLResponse, include_in_schema=False)
def tracing_page() -> HTMLResponse:
    """Trace viewer: how the agents worked through a run, step by step."""
    page = _WEB_DIR / "traces.html"
    if not page.exists():
        return HTMLResponse("<h1>Traces</h1><p>UI asset missing. API is at /traces.</p>")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@router.get("/schemas/response-document", tags=["system"])
def get_response_document_schema() -> dict:
    """JSON Schema of the document every model answer is parsed into."""
    return response_document_json_schema()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Minimal UI: upload, watch progress, read the report."""
    page = _WEB_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>PaperMind</h1><p>UI asset missing. API is at /docs.</p>")
    return HTMLResponse(page.read_text(encoding="utf-8"))


def register_error_handlers(app) -> None:
    """Map the typed error hierarchy onto HTTP responses."""
    from fastapi.responses import JSONResponse

    @app.exception_handler(PaperPilotError)
    async def handle(request: Request, exc: PaperPilotError):  # noqa: ARG001
        headers = {"Retry-After": str(exc.retry_after)} if isinstance(exc, RateLimitError) else None
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict(), headers=headers)
