# PaperPilot API image.
#
# Tesseract is installed because OCR is a fallback the ingestion layer expects to
# be able to reach; without it, scanned pages are indexed with whatever native
# text exists and the /health endpoint says so.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer: copy only the manifest first so edits to source do not
# invalidate the (slow) install layer.
COPY pyproject.toml README.md ./
COPY src ./src
# Optional extras to bake in, e.g. "ocr,embeddings" when the runtime config
# selects PAPERPILOT_EMBEDDING_PROVIDER=sentence_transformers. Torch comes from
# the CPU wheel index so the image does not pull multi-GB CUDA libraries.
ARG PAPERPILOT_EXTRAS=ocr
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[${PAPERPILOT_EXTRAS}]"

COPY fixtures ./fixtures
COPY scripts ./scripts

# Run as a non-root user; the data directory is the only writable path needed.
RUN useradd --create-home --uid 10001 paperpilot \
    && mkdir -p /data \
    && chown -R paperpilot:paperpilot /data /app
USER paperpilot

ENV PAPERPILOT_DATA_DIR=/data \
    PAPERPILOT_DATABASE_URL=sqlite:////data/paperpilot.sqlite \
    PAPERPILOT_LOG_FORMAT=json \
    HF_HOME=/data/hf-cache

EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "paperpilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
