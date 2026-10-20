"""Split page text into overlapping chunks suitable for embedding."""

from app.shared.models import Chunk


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack whole words into chunks of at most `chunk_size` characters.

    Consecutive chunks share up to `overlap` characters of trailing words, so a
    sentence crossing a boundary is retrievable from either side. Words are
    never cut; a single word longer than `chunk_size` becomes its own chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()  # also normalises PDF whitespace
    chunks: list[str] = []
    start = 0
    while start < len(words):
        # Grow the chunk word by word (always take at least one word).
        end, length = start + 1, len(words[start])
        while end < len(words) and length + 1 + len(words[end]) <= chunk_size:
            length += 1 + len(words[end])
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break

        # Step back over trailing words that fit in the overlap, but always move forward.
        next_start, tail = end, 0
        while next_start - 1 > start and tail + 1 + len(words[next_start - 1]) <= overlap:
            next_start -= 1
            tail += 1 + len(words[next_start])
        start = next_start
    return chunks


def chunk_pages(
    pages: list[str],
    document_id: str,
    filename: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Chunk each page separately so every chunk keeps its page number."""
    return [
        Chunk(
            id=f"{document_id}:{page_no}:{i}",
            document_id=document_id,
            filename=filename,
            page=page_no,
            text=piece,
        )
        for page_no, page_text in enumerate(pages, start=1)
        for i, piece in enumerate(split_text(page_text, chunk_size, overlap))
    ]
