from app.documents.cleaner import remove_repeated_lines


def test_removes_header_that_differs_only_by_page_number():
    pages = [f"ACME Corp   intern {n}\nUnique content {chr(65 + n)}" for n in range(1, 5)]
    cleaned = remove_repeated_lines(pages)
    assert all("ACME" not in p for p in cleaned)
    assert cleaned[0] == "Unique content B"


def test_keeps_lines_on_few_pages():
    pages = ["Chapter 1\nalpha", "Chapter 1\nbeta", "Other\ngamma", "Other\ndelta", "Third\neps"]
    assert remove_repeated_lines(pages) == pages


def test_short_documents_untouched():
    pages = ["Header\nA", "Header\nB"]
    assert remove_repeated_lines(pages) == pages
