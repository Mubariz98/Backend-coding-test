"""API contracts (Pydantic) for the answers module."""

from pydantic import BaseModel, Field

from app.search.schemas import QueryText, SearchResult


class AskRequest(BaseModel):
    question: QueryText = Field(
        description="Question in any language; the answer uses the same language.",
        examples=["Welche Garantie bietet Volkswagen?"],
    )
    document_id: str | None = Field(None, description="Only use this document as context.")


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResult] = Field(description="Passages given to the model; [n] in the answer = sources[n-1].")
