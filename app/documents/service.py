"""Use case: ingest a PDF (parse -> clean -> chunk -> embed -> store)."""

import hashlib
import logging
import time

from app.documents.chunker import chunk_pages
from app.documents.cleaner import remove_repeated_lines
from app.documents.exceptions import DuplicateDocumentError, EmptyDocumentError
from app.documents.models import IngestedDocument
from app.documents.pdf_parser import extract_pages
from app.shared.ports import Embedder, VectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, store: VectorStore, embedder: Embedder, chunk_size: int, chunk_overlap: int):
        self._store = store
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest_pdf(self, data: bytes, filename: str) -> IngestedDocument:
        started = time.perf_counter()

        # Content hash as ID: the same file always gets the same ID, regardless
        # of filename. Checked first so duplicates cost no parsing or embedding.
        document_id = hashlib.sha256(data).hexdigest()[:32]
        if self._store.has_document(document_id):
            raise DuplicateDocumentError(document_id)

        pages = remove_repeated_lines(extract_pages(data))
        chunks = chunk_pages(pages, document_id, filename, self._chunk_size, self._chunk_overlap)
        if not chunks:
            raise EmptyDocumentError("No extractable text found in PDF (scanned images need OCR).")

        vectors = self._embedder.embed_documents([c.text for c in chunks])
        # Chunk IDs are deterministic and the store upserts, so a concurrent
        # duplicate upload overwrites identical data instead of duplicating it.
        self._store.add(chunks, vectors)

        logger.info(
            "Ingested %s: %d pages, %d chunks in %.1fs",
            filename, len(pages), len(chunks), time.perf_counter() - started,
        )
        return IngestedDocument(
            document_id=document_id,
            filename=filename,
            pages=len(pages),
            chunks=len(chunks),
        )
