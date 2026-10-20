"""Domain objects for the answers module."""

from collections.abc import Iterator
from dataclasses import dataclass

from app.shared.models import SearchHit


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[SearchHit]


@dataclass(frozen=True)
class StreamingAnswer:
    """Sources are known before generation starts; the text arrives piece by piece."""

    sources: list[SearchHit]
    tokens: Iterator[str]
