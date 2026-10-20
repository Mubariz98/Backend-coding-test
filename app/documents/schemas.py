"""API contracts (Pydantic) for the documents module."""

from pydantic import BaseModel, ConfigDict, Field


class UploadResponse(BaseModel):
    # Lets us build this directly from the IngestedDocument dataclass.
    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(description="Content hash; identical files get the same ID.")
    filename: str
    pages: int
    chunks: int
