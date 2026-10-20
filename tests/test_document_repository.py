from datetime import datetime, timedelta, timezone

from app.documents.models import Document
from app.infrastructure.sqlite_document_repository import SqliteDocumentRepository


def doc(doc_id: str, minutes_ago: int = 0) -> Document:
    when = datetime(2026, 1, 1, tzinfo=timezone.utc) - timedelta(minutes=minutes_ago)
    return Document(doc_id, f"{doc_id}.pdf", pages=3, chunks=7, size_bytes=1234, uploaded_at=when)


def test_add_get_round_trip(tmp_path):
    repo = SqliteDocumentRepository(str(tmp_path / "nested" / "docs.db"))  # creates folders
    repo.add(doc("a"))
    assert repo.get("a") == doc("a")
    assert repo.get("missing") is None


def test_list_newest_first(tmp_path):
    repo = SqliteDocumentRepository(str(tmp_path / "docs.db"))
    repo.add(doc("old", minutes_ago=10))
    repo.add(doc("new", minutes_ago=1))
    assert [d.document_id for d in repo.list()] == ["new", "old"]


def test_delete_and_persistence(tmp_path):
    path = str(tmp_path / "docs.db")
    SqliteDocumentRepository(path).add(doc("a"))
    SqliteDocumentRepository(path).add(doc("b"))

    reopened = SqliteDocumentRepository(path)
    reopened.delete("a")
    reopened.delete("missing")  # no-op

    assert [d.document_id for d in SqliteDocumentRepository(path).list()] == ["b"]
