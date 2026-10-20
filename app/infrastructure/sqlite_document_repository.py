"""SQLite implementation of the `DocumentRepository` port (stdlib only, one file on disk)."""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from app.documents.models import Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename    TEXT    NOT NULL,
    pages       INTEGER NOT NULL,
    chunks      INTEGER NOT NULL,
    size_bytes  INTEGER NOT NULL,
    uploaded_at TEXT    NOT NULL
)
"""


class SqliteDocumentRepository:
    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # A short-lived connection per call is simple and safe across FastAPI's worker threads.
        return sqlite3.connect(self._path)

    def add(self, document: Document) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (
                    document.document_id,
                    document.filename,
                    document.pages,
                    document.chunks,
                    document.size_bytes,
                    document.uploaded_at.isoformat(),
                ),
            )

    def get(self, document_id: str) -> Document | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return _to_document(row) if row else None

    def list(self) -> list[Document]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [_to_document(r) for r in rows]

    def delete(self, document_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))


def _to_document(row: tuple) -> Document:
    document_id, filename, pages, chunks, size_bytes, uploaded_at = row
    return Document(document_id, filename, pages, chunks, size_bytes, datetime.fromisoformat(uploaded_at))
