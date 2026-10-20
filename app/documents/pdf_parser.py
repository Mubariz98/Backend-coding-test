"""Extract text from PDF files."""

import logging
from io import BytesIO

from pypdf import PageObject, PasswordType, PdfReader

from app.documents.exceptions import PDFParseError

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"


def extract_pages(data: bytes) -> list[str]:
    """Return the text of each page (index 0 = page 1)."""
    # Check content, not the filename. The spec allows the header anywhere in the first 1 KB.
    if _PDF_MAGIC not in data[:1024]:
        raise PDFParseError("File is not a PDF.")

    try:
        reader = PdfReader(BytesIO(data))
        # Many PDFs are encrypted with an empty user password (permissions only);
        # those open fine. A real user password cannot be supplied.
        if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
            raise PDFParseError("Password-protected PDFs are not supported.")
        return [_page_text(page, page_no) for page_no, page in enumerate(reader.pages, start=1)]
    except PDFParseError:
        raise
    except Exception as exc:
        # pypdf raises many exception types for malformed files; expose one domain error.
        raise PDFParseError(f"Could not read PDF: {exc}") from exc


def _page_text(page: PageObject, page_no: int) -> str:
    """One broken page should not fail the whole document."""
    try:
        return page.extract_text() or ""
    except Exception:
        logger.warning("Skipping unreadable page %d", page_no, exc_info=True)
        return ""
