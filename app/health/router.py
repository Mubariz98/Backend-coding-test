"""Liveness endpoint; also reports which backends are active and whether the LLM is reachable."""

from fastapi import APIRouter

from app.core.dependencies import LLMDep, SettingsDep

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: SettingsDep, llm: LLMDep) -> dict[str, str]:
    return {
        "status": "ok",
        "vector_db": settings.vector_db,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        # Search works without the LLM; only /ask needs it.
        "llm_status": "ok" if llm.is_available() else "unavailable",
    }
