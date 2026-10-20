"""Map domain errors to HTTP responses in one place.

The domain raises plain Python exceptions and knows nothing about HTTP;
routers stay free of try/except blocks.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.answers.exceptions import LLMUnavailableError
from app.documents.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmptyDocumentError,
    PDFParseError,
)

_STATUS_CODES: dict[type[Exception], int] = {
    PDFParseError: 400,
    DocumentNotFoundError: 404,
    DuplicateDocumentError: 409,
    EmptyDocumentError: 422,
    LLMUnavailableError: 503,
}


def register_error_handlers(app: FastAPI) -> None:
    for exc_type, status_code in _STATUS_CODES.items():
        app.add_exception_handler(exc_type, _make_handler(status_code))


def _make_handler(status_code: int):
    async def handler(_: Request, exc: Exception) -> JSONResponse:
        # Same shape as FastAPI's own HTTPException responses.
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return handler
