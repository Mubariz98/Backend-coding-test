"""Use cases for documents: ingest (parse -> clean -> chunk -> embed -> store), list, get, delete."""

import hashlib
import logging
import time
from datetime import datetime, timezone

from app.documents.chunker import chunk_pages
from app.documents.cleaner import remove_repeated_lines
from app.documents.exceptions import DocumentNotFoundError, DuplicateDocumentError, EmptyDocumentError
from app.documents.models import Document
from app.documents.pdf_parser import extract_pages
from app.documents.ports import DocumentRepository
from app.shared.ports import Embedder, VectorStore

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        repository: DocumentRepository,
        store: VectorStore,
        embedder: Embedder,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self._repository = repository
        self._store = store
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest_pdf(self, data: bytes, filename: str) -> Document:
        started = time.perf_counter()

        # Content hash as ID: the same file always gets the same ID, regardless
        # of filename. Checked first so duplicates cost no parsing or embedding.
        document_id = hashlib.sha256(data).hexdigest()[:32]
        if self._repository.get(document_id):
            raise DuplicateDocumentError(document_id)

        pages = remove_repeated_lines(extract_pages(data))
        chunks = chunk_pages(pages, document_id, filename, self._chunk_size, self._chunk_overlap)
        if not chunks:
            raise EmptyDocumentError("No extractable text found in PDF (scanned images need OCR).")

        vectors = self._embedder.embed_documents([c.text for c in chunks])
        # Chunks first, metadata last: if we crash in between, a re-upload simply
        # upserts the same deterministic chunk IDs again (no duplicates).
        self._store.add(chunks, vectors)
        document = Document(
            document_id=document_id,
            filename=filename,
            pages=len(pages),
            chunks=len(chunks),
            size_bytes=len(data),
            uploaded_at=datetime.now(timezone.utc),
        )
        self._repository.add(document)

        logger.info(
            "Ingested %s: %d pages, %d chunks in %.1fs",
            filename, len(pages), len(chunks), time.perf_counter() - started,
        )
        return document

    def list_documents(self) -> list[Document]:
        return self._repository.list()

    def get_document(self, document_id: str) -> Document:
        document = self._repository.get(document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)
        return document

    def delete_document(self, document_id: str) -> None:
        self.get_document(document_id)  # raises if unknown
        self._store.delete_document(document_id)
        self._repository.delete(document_id)
        logger.info("Deleted document %s", document_id)
