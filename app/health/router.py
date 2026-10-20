"""Liveness endpoint; also reports which backends are active."""

from fastapi import APIRouter

from app.core.dependencies import SettingsDep

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: SettingsDep) -> dict[str, str]:
    return {
        "status": "ok",
        "vector_db": settings.vector_db,
        "embedding_model": settings.embedding_model,
    }
