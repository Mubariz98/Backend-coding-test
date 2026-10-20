"""Domain errors for document ingestion. The router maps these to HTTP codes."""


class IngestionError(Exception):
    """Base class for errors raised while ingesting a document."""


class PDFParseError(IngestionError):
    """The file could not be read as a PDF."""


class EmptyDocumentError(IngestionError):
    """The PDF has no extractable text (e.g. a scanned image without OCR)."""


class DuplicateDocumentError(IngestionError):
    """A file with identical content has already been ingested."""

    def __init__(self, document_id: str):
        super().__init__(f"Document already uploaded (document_id={document_id}).")
        self.document_id = document_id
