"""Application entry point: `uvicorn app.main:app --reload`.

Only assembles the app: startup wiring, error handlers and routers.
No endpoint logic lives here.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import Settings
from app.core.container import build_container
from app.core.error_handlers import register_error_handlers
from app.documents.router import router as documents_router
from app.health.router import router as health_router
from app.search.router import router as search_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    """App factory; tests pass their own Settings (e.g. a temp DB path)."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load the model and open the DB once at startup, not on the first request.
        app.state.container = build_container(settings)
        yield

    app = FastAPI(
        title="PDF RAG API",
        description="Upload PDFs and search them with natural language queries.",
        version="0.2.0",
        lifespan=lifespan,
    )
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    return app


app = create_app()
