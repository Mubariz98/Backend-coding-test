"""Domain objects shared by the `documents` and `search` modules.

Plain dataclasses on purpose: the domain has no dependency on FastAPI,
Pydantic or ChromaDB. API schemas are defined separately in each module.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A piece of text from a PDF page, ready to be embedded."""

    id: str
    document_id: str
    filename: str
    page: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    """A stored chunk returned by a similarity search."""

    chunk: Chunk
    score: float  # cosine similarity; higher is more relevant
