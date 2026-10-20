"""Domain errors for the documents module. Mapped to HTTP codes in core/error_handlers.py."""


class DocumentError(Exception):
    """Base class for errors raised by the documents module."""


class PDFParseError(DocumentError):
    """The file could not be read as a PDF."""


class EmptyDocumentError(DocumentError):
    """The PDF has no extractable text (e.g. a scanned image without OCR)."""


class DuplicateDocumentError(DocumentError):
    """A file with identical content has already been ingested."""

    def __init__(self, document_id: str):
        super().__init__(f"Document already uploaded (document_id={document_id}).")
        self.document_id = document_id


class DocumentNotFoundError(DocumentError):
    def __init__(self, document_id: str):
        super().__init__(f"Document not found (document_id={document_id}).")
        self.document_id = document_id
