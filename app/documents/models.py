"""Domain objects specific to the documents module."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Document:
    """A PDF that has been parsed, chunked and stored."""

    document_id: str
    filename: str
    pages: int
    chunks: int
    size_bytes: int
    uploaded_at: datetime
