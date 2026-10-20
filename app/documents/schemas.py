"""API contracts (Pydantic) for the documents module."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    # Lets us build this directly from the Document dataclass.
    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(description="Content hash; identical files get the same ID.")
    filename: str
    pages: int
    chunks: int
    size_bytes: int
    uploaded_at: datetime
