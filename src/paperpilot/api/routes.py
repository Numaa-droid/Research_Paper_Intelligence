"""HTTP routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse

from .. import __version__
from ..domain.enums import AnalysisStatus
from ..errors import PaperPilotError, UnsupportedFileError
from ..observability import info
from ..response import SCHEMA_VERSION, response_document_json_schema
from ..service import PaperPilotService
from .schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    DocumentResponse,
    HealthResponse,
    UploadResponse,
)

router = APIRouter()

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def get_service(request: Request) -> PaperPilotService:
    return request.app.state.service


def get_runner(request: Request):
    return request.app.state.task_runner


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: PaperPilotService = Depends(get_service)) -> HealthResponse:
    """Liveness plus the configuration facts a caller needs to interpret results."""
    from ..ingestion.ocr import build_ocr_engine

    settings = service.settings
    ocr = build_ocr_engine(**settings.ocr_engine_options())

    warnings: list[str] = []
    if settings.llm_provider == "mock":
        warnings.append(
            "The deterministic mock provider is active: the pipeline runs end to end, "
            "but reports are not a real analysis. Set PAPERPILOT_LLM_PROVIDER."
        )
    if settings.ocr_enabled and not ocr.available:
        unavailable = (
            f"the PaddleOCR-VL server at {settings.ocr_base_url} is not reachable"
            if settings.ocr_engine == "paddleocr_vl"
            else "Tesseract is unavailable"
        )
        warnings.append(
            f"OCR is enabled but {unavailable}; scanned pages will be indexed "
            "with whatever native text exists."
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


@router.post("/papers", response_model=UploadResponse, status_code=201, tags=["papers"])
async def upload_paper(
    file: UploadFile = File(..., description="A research-paper PDF"),
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

    metadata = service.register_upload(file.file, filename=file.filename or "upload.pdf")
    info("api.upload", document_id=metadata.document_id, bytes=metadata.file_size_bytes)
    return UploadResponse(
        document_id=metadata.document_id,
        filename=metadata.filename,
        file_size_bytes=metadata.file_size_bytes,
    )


@router.get("/papers", response_model=list[DocumentResponse], tags=["papers"])
def list_papers(
    limit: int = 50, service: PaperPilotService = Depends(get_service)
) -> list[DocumentResponse]:
    return [DocumentResponse.from_metadata(m) for m in service.list_documents(limit=limit)]


@router.get("/papers/{document_id}", response_model=DocumentResponse, tags=["papers"])
def get_paper(
    document_id: str, service: PaperPilotService = Depends(get_service)
) -> DocumentResponse:
    return DocumentResponse.from_metadata(service.get_document(document_id))


@router.post(
    "/papers/{document_id}/analyze",
    response_model=AnalysisResponse,
    status_code=202,
    tags=["analyses"],
)
def analyze_paper(
    document_id: str,
    payload: AnalyzeRequest | None = None,
    service: PaperPilotService = Depends(get_service),
    runner=Depends(get_runner),
) -> AnalysisResponse:
    """Start an analysis.

    Returns ``202`` with an analysis id immediately; poll
    ``GET /analyses/{analysis_id}`` for progress. Pass ``{"wait": true}`` to block
    until the run finishes (small papers only).
    """
    record = service.create_analysis(document_id)
    options = payload or AnalyzeRequest()

    if options.wait:
        completed = service.run_analysis(record.analysis_id)
        return AnalysisResponse.from_record(completed)

    runner.submit(record.analysis_id, lambda: service.run_analysis(record.analysis_id))
    # Re-read: a synchronous runner (tests, single-worker deployments) has already
    # finished by now, and returning the stale record would report "pending".
    return AnalysisResponse.from_record(service.get_analysis(record.analysis_id))


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse, tags=["analyses"])
def get_analysis(
    analysis_id: str, service: PaperPilotService = Depends(get_service)
) -> AnalysisResponse:
    """Poll an analysis. Safe to call as often as a client likes."""
    return AnalysisResponse.from_record(service.get_analysis(analysis_id))


@router.get("/analyses/{analysis_id}/report", tags=["analyses"])
def get_report(
    analysis_id: str,
    format: str = "markdown",
    service: PaperPilotService = Depends(get_service),
):
    """Return the final Markdown report."""
    record = service.get_analysis(analysis_id)
    if record.status is not AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "report_not_ready",
                "message": f"analysis is '{record.status.value}', not completed",
                "details": {"status": record.status.value},
            },
        )
    markdown = service.get_report(analysis_id)
    if format == "json":
        return record.model_dump(mode="json")
    return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8")


@router.get("/analyses/{analysis_id}/findings", tags=["analyses"])
def get_findings(analysis_id: str, service: PaperPilotService = Depends(get_service)) -> dict:
    """Structured findings and verification verdicts, for programmatic use."""
    record = service.get_analysis(analysis_id)
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
    analysis_id: str, service: PaperPilotService = Depends(get_service)
) -> dict:
    """Each model answer as written (Markdown) and as its parsed response document."""
    record = service.get_analysis(analysis_id)
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


@router.get("/schemas/response-document", tags=["system"])
def get_response_document_schema() -> dict:
    """JSON Schema of the document every model answer is parsed into."""
    return response_document_json_schema()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Minimal UI: upload, watch progress, read the report."""
    page = _WEB_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>PaperPilot</h1><p>UI asset missing. API is at /docs.</p>")
    return HTMLResponse(page.read_text(encoding="utf-8"))


def register_error_handlers(app) -> None:
    """Map the typed error hierarchy onto HTTP responses."""
    from fastapi.responses import JSONResponse

    @app.exception_handler(PaperPilotError)
    async def handle(request: Request, exc: PaperPilotError):  # noqa: ARG001
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())
