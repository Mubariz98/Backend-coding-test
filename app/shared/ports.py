"""Interfaces (ports) the domain depends on.

Services depend on these Protocols rather than concrete libraries, so the
embedding model and the vector DB can each be swapped (or faked in tests)
without touching business logic. Any class with matching methods qualifies;
no inheritance needed.
"""

from typing import Protocol

from app.shared.models import Chunk, SearchHit

Vector = list[float]


class Embedder(Protocol):
    """Turns text into vectors. Documents and queries must use the same model."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class VectorStore(Protocol):
    """Stores chunks with their vectors and finds the nearest ones."""

    def add(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        """Insert or overwrite chunks by ID (idempotent)."""
        ...

    def search(
        self, query_vector: Vector, top_k: int, document_id: str | None = None
    ) -> list[SearchHit]:
        """Return up to `top_k` hits, most similar first, optionally within one document."""
        ...

    def has_document(self, document_id: str) -> bool: ...
