"""Interface for storing document metadata (filename, pages, upload time).

Kept separate from the VectorStore: the vector DB holds chunks for search,
this holds one row per document for listing, duplicate checks and deletion.
"""

from typing import Protocol

from app.documents.models import Document


class DocumentRepository(Protocol):
    def add(self, document: Document) -> None: ...

    def get(self, document_id: str) -> Document | None: ...

    def list(self) -> list[Document]:
        """All documents, newest first."""
        ...

    def delete(self, document_id: str) -> None: ...
