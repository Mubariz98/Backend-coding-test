"""Serves the single-page demo UI. It only talks to the public API, like any other client."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(include_in_schema=False)

_INDEX = Path(__file__).parent / "static" / "index.html"


@router.get("/")
def index() -> FileResponse:
    return FileResponse(_INDEX)
