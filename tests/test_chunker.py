import pytest

from app.documents.chunker import chunk_pages, split_text


def test_chunks_respect_size_and_never_cut_words():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = split_text(text, chunk_size=16, overlap=0)
    assert all(len(c) <= 16 for c in chunks)
    assert " ".join(chunks).split() == text.split()  # every word kept, in order


def test_consecutive_chunks_overlap():
    chunks = split_text("a b c d e f", chunk_size=5, overlap=2)
    assert chunks == ["a b c", "c d e", "e f"]


def test_word_longer_than_chunk_becomes_own_chunk():
    assert split_text("tiny " + "x" * 50 + " end", chunk_size=10, overlap=3) == ["tiny", "x" * 50, "end"]


def test_whitespace_is_normalised():
    assert split_text("  hello \n\n  world\t ", chunk_size=100, overlap=10) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n ", chunk_size=100, overlap=10) == []


def test_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=2, overlap=2)


def test_chunk_pages_keeps_page_numbers_and_stable_ids():
    chunks = chunk_pages(["first page", "", "third page"], "doc1", "a.pdf", 100, 10)
    assert [c.page for c in chunks] == [1, 3]  # empty page is skipped
    assert [c.id for c in chunks] == ["doc1:1:0", "doc1:3:0"]
