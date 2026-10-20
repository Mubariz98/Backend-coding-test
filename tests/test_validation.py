"""Pydantic validation of settings and API request models."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.search.schemas import QueryRequest


def test_default_settings_are_valid():
    assert Settings(_env_file=None).chunk_overlap < Settings(_env_file=None).chunk_size


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 0},
        {"max_upload_mb": 0},
        {"vector_db": "mongodb"},
    ],
)
def test_invalid_settings_rejected(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_query_is_stripped():
    assert QueryRequest(query="  pipes  ").query == "pipes"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 2001},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": 51},
        {"query": "ok", "min_score": 1.5},
    ],
)
def test_invalid_queries_rejected(payload):
    with pytest.raises(ValidationError):
        QueryRequest(**payload)
