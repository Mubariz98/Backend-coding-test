"""API contracts (Pydantic) for the search module."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.shared.models import SearchHit

MAX_TOP_K = 50

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class QueryRequest(BaseModel):
    query: QueryText = Field(
        description="Natural language question (any language).",
        examples=["How do I fix a leaking faucet?"],
    )
    top_k: int | None = Field(None, ge=1, le=MAX_TOP_K, description="Max results; server default if omitted.")
    document_id: str | None = Field(None, description="Only search within this document.")
    min_score: float | None = Field(
        None, ge=-1.0, le=1.0, description="Drop results below this cosine similarity."
    )


class SearchResult(BaseModel):
    text: str
    document_id: str
    filename: str
    page: int
    score: float = Field(description="Cosine similarity; higher is more relevant.")

    @classmethod
    def from_hit(cls, hit: SearchHit) -> "SearchResult":
        """Map a domain SearchHit to its API representation."""
        c = hit.chunk
        return cls(
            text=c.text,
            document_id=c.document_id,
            filename=c.filename,
            page=c.page,
            score=hit.score,
        )


class QueryResponse(BaseModel):
    query: str
    results: list[SearchResult]
