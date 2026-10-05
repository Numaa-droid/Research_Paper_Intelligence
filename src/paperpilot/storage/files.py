"""Local filesystem storage.

Uploads are stored under a document-id-derived name so a hostile filename can
never escape the upload directory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO

from ..errors import FileTooLargeError, NotFoundError
from ..observability import debug
from .repository import FileStorage

_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_COPY_CHUNK = 1024 * 1024


def safe_filename(name: str, *, fallback: str = "upload.pdf") -> str:
    """Strip a user-supplied filename down to something safe to write."""
    cleaned = _SAFE.sub("_", Path(name or "").name).strip("._")
    return cleaned[:120] or fallback


class LocalFileStorage(FileStorage):
    """Stores uploads and reports on the local disk."""

    def __init__(self, *, upload_dir: Path, report_dir: Path) -> None:
        self.upload_dir = Path(upload_dir)
        self.report_dir = Path(report_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self, stream: BinaryIO, *, filename: str, document_id: str, max_bytes: int | None = None
    ) -> Path:
        """Copy ``stream`` to disk, stopping as soon as it passes ``max_bytes``."""
        target = self.upload_dir / f"{document_id}__{safe_filename(filename)}"
        written = 0
        with target.open("wb") as handle:
            while chunk := stream.read(_COPY_CHUNK):
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise FileTooLargeError(
                        f"upload exceeds the {max_bytes / (1024 * 1024):g} MB limit",
                        details={"max_bytes": max_bytes},
                    )
                handle.write(chunk)
        debug("storage.upload_saved", document_id=document_id, bytes=target.stat().st_size)
        return target

    def save_bytes(self, data: bytes, *, filename: str, document_id: str) -> Path:
        target = self.upload_dir / f"{document_id}__{safe_filename(filename)}"
        target.write_bytes(data)
        return target

    def path_for(self, document_id: str) -> Path | None:
        matches = sorted(self.upload_dir.glob(f"{document_id}__*"))
        return matches[0] if matches else None

    def require_path(self, document_id: str) -> Path:
        path = self.path_for(document_id)
        if path is None:
            raise NotFoundError(f"no stored file for document '{document_id}'")
        return path

    def save_report(self, analysis_id: str, markdown: str) -> Path:
        target = self.report_dir / f"{safe_filename(analysis_id, fallback='report')}.md"
        target.write_text(markdown, encoding="utf-8")
        debug("storage.report_saved", analysis_id=analysis_id, bytes=len(markdown))
        return target

    def read_report(self, analysis_id: str) -> str | None:
        target = self.report_dir / f"{safe_filename(analysis_id, fallback='report')}.md"
        return target.read_text(encoding="utf-8") if target.exists() else None
