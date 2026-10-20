"""HTTP endpoints for managing documents.

Kept thin: read the upload, enforce transport limits, call the service.
Domain errors are mapped to status codes in `core/error_handlers.py`.
"""

from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.core.dependencies import DocumentServiceDep, SettingsDep
from app.documents.schemas import DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])

_NOT_FOUND = {404: {"description": "Unknown document_id"}}


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
    service: DocumentServiceDep,
    settings: SettingsDep,
) -> DocumentResponse:
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
    return DocumentResponse.model_validate(document)


@router.get("")
def list_documents(service: DocumentServiceDep) -> list[DocumentResponse]:
    return [DocumentResponse.model_validate(d) for d in service.list_documents()]


@router.get("/{document_id}", responses=_NOT_FOUND)
def get_document(document_id: str, service: DocumentServiceDep) -> DocumentResponse:
    return DocumentResponse.model_validate(service.get_document(document_id))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, responses=_NOT_FOUND)
def delete_document(document_id: str, service: DocumentServiceDep) -> Response:
    service.delete_document(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _safe_filename(raw: str | None) -> str:
    """Keep only the base name; clients may send full or Windows-style paths."""
    return PurePosixPath((raw or "").replace("\\", "/")).name or "untitled.pdf"
