"""HTTP endpoints for uploading documents.

Kept thin: read the upload, enforce transport limits, call the service.
Domain errors are mapped to status codes in `core/error_handlers.py`.
"""

from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.core.dependencies import IngestionServiceDep, SettingsDep
from app.documents.schemas import UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"description": "Not a readable PDF (or password-protected)"},
        409: {"description": "Same file content already uploaded"},
        413: {"description": "File larger than MAX_UPLOAD_MB"},
        422: {"description": "PDF has no extractable text"},
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF file to index.")],
    service: IngestionServiceDep,
    settings: SettingsDep,
) -> UploadResponse:
    # Read at most limit + 1 bytes, so an oversized upload is never fully loaded into memory.
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds {settings.max_upload_mb} MB.",
        )

    # Parsing and embedding are CPU-bound; run them off the event loop
    # so other requests are not blocked during a long upload.
    document = await run_in_threadpool(service.ingest_pdf, data, _safe_filename(file.filename))
    return UploadResponse.model_validate(document)


def _safe_filename(raw: str | None) -> str:
    """Keep only the base name; clients may send full or Windows-style paths."""
    return PurePosixPath((raw or "").replace("\\", "/")).name or "untitled.pdf"
