import pytest

from app.documents.exceptions import PDFParseError
from app.documents.pdf_parser import extract_pages
from tests.pdf_factory import make_password_protected_pdf, make_pdf


def test_extracts_text_per_page():
    pages = extract_pages(make_pdf("Page one", "Page two"))
    assert [p.strip() for p in pages] == ["Page one", "Page two"]


@pytest.mark.parametrize("data", [b"", b"hello", b"PK\x03\x04 zip file"], ids=["empty", "text", "zip"])
def test_rejects_non_pdf_content(data):
    with pytest.raises(PDFParseError, match="not a PDF"):
        extract_pages(data)


def test_rejects_corrupt_pdf():
    with pytest.raises(PDFParseError):
        extract_pages(b"%PDF-1.4\n garbage without any objects")


def test_rejects_password_protected_pdf():
    with pytest.raises(PDFParseError, match="Password-protected"):
        extract_pages(make_password_protected_pdf())
