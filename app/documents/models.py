"""Domain objects specific to the documents module."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestedDocument:
    """Summary of a PDF that has been parsed, chunked and stored."""

    document_id: str
    filename: str
    pages: int
    chunks: int
