"""Shared fixtures. Every test gets its own temporary databases."""

import dataclasses
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.answers.service import AnswerService
from app.core.config import Settings
from app.main import create_app

BACKENDS = ["qdrant", "chroma"]
MODEL_CACHE = str(Path(__file__).resolve().parents[1] / "data" / "models")


@pytest.fixture
def make_settings(tmp_path):
    """Settings isolated from the developer's `.env` and real data."""

    def make(**overrides) -> Settings:
        values = {
            "qdrant_path": str(tmp_path / "qdrant"),
            "chroma_path": str(tmp_path / "chroma"),
            "sqlite_path": str(tmp_path / "documents.db"),
            "model_cache_dir": MODEL_CACHE,
            "llm_base_url": "http://127.0.0.1:9",  # nothing listens here: LLM is "offline" by default
            **overrides,
        }
        return Settings(_env_file=None, **values)

    return make


@pytest.fixture(params=BACKENDS)
def make_client(request, make_settings):
    """Build a TestClient per backend, so every API test runs on Qdrant AND Chroma.

    Pass `llm=FakeLLM()` to replace the real LLM (the embedder and DB stay real).
    """
    clients = []

    def make(llm=None, **overrides) -> TestClient:
        client = TestClient(create_app(make_settings(vector_db=request.param, **overrides)))
        client.__enter__()  # runs the lifespan (startup wiring)
        if llm is not None:
            container = client.app.state.container
            s = container.settings
            client.app.state.container = dataclasses.replace(
                container,
                llm=llm,
                answers=AnswerService(container.search, llm, s.answer_top_k, s.answer_min_score),
            )
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
