"""Test helpers: build tiny PDFs in memory (ASCII text only)."""

from io import BytesIO

from pypdf import PdfWriter


def make_pdf(*pages: str) -> bytes:
    """Build a PDF with one page per argument; newlines in a page become separate lines."""
    pages = pages or ("",)
    page_ids = [4 + 2 * i for i in range(len(pages))]  # each page: page obj + content obj
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [%s] /Count %d >>"
        % (b" ".join(b"%d 0 R" % p for p in page_ids), len(pages)),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_id, text in zip(page_ids, pages):
        lines = " 0 -14 Td ".join(f"({line}) Tj" for line in text.split("\n"))
        stream = f"BT /F1 12 Tf 72 720 Td {lines} ET".encode("latin-1")
        objects[page_id] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents %d 0 R /Resources << /Font << /F1 3 0 R >> >> >>" % (page_id + 1)
        )
        objects[page_id + 1] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj_id in sorted(objects):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (obj_id, objects[obj_id])
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref)
    return bytes(out)


def make_password_protected_pdf(password: str = "secret") -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt(password, algorithm="AES-256")
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()
