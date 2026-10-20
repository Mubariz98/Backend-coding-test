"""Contract tests: every VectorStore adapter must pass the same checks.

This is what makes the DB swappable. A new adapter (pgvector, Weaviate, ...)
is added to the fixture and must pass these unchanged.
"""

import pytest

from app.infrastructure.chroma_store import ChromaVectorStore
from app.infrastructure.qdrant_store import QdrantVectorStore
from app.shared.models import Chunk


@pytest.fixture(params=["qdrant", "chroma"])
def store(request, tmp_path):
    if request.param == "qdrant":
        return QdrantVectorStore(str(tmp_path), "test", vector_size=3)
    return ChromaVectorStore(str(tmp_path), "test")


def chunk(doc: str, i: int, text: str = "") -> Chunk:
    return Chunk(id=f"{doc}:1:{i}", document_id=doc, filename=f"{doc}.pdf", page=1, text=text or f"{doc}-{i}")


def test_search_on_empty_store_returns_nothing(store):
    assert store.search([1.0, 0.0, 0.0], top_k=5) == []


def test_nearest_first_with_similarity_scores(store):
    store.add(
        [chunk("a", 0, "x-axis"), chunk("a", 1, "y-axis"), chunk("a", 2, "diagonal")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0], top_k=3)

    assert [h.chunk.text for h in hits] == ["x-axis", "diagonal", "y-axis"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert hits[2].score == pytest.approx(0.0, abs=1e-3)
    assert hits[0].chunk == chunk("a", 0, "x-axis")  # metadata round-trips intact


def test_top_k_larger_than_store(store):
    store.add([chunk("a", 0)], [[1.0, 0.0, 0.0]])
    assert len(store.search([1.0, 0.0, 0.0], top_k=10)) == 1


def test_filter_by_document(store):
    store.add([chunk("a", 0), chunk("b", 0)], [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]])
    hits = store.search([1.0, 0.0, 0.0], top_k=5, document_id="b")
    assert [h.chunk.document_id for h in hits] == ["b"]


def test_has_document(store):
    store.add([chunk("a", 0)], [[1.0, 0.0, 0.0]])
    assert store.has_document("a")
    assert not store.has_document("b")


def test_add_is_idempotent(store):
    for _ in range(2):
        store.add([chunk("a", 0)], [[1.0, 0.0, 0.0]])
    assert len(store.search([1.0, 0.0, 0.0], top_k=10)) == 1


def test_add_nothing_is_noop(store):
    store.add([], [])
    assert store.search([1.0, 0.0, 0.0], top_k=5) == []


def test_data_persists_across_instances(request, store, tmp_path):
    store.add([chunk("a", 0)], [[1.0, 0.0, 0.0]])
    if isinstance(store, QdrantVectorStore):
        store._client.close()  # embedded Qdrant allows one client per folder
        reopened = QdrantVectorStore(str(tmp_path), "test", vector_size=3)
    else:
        reopened = ChromaVectorStore(str(tmp_path), "test")
    assert reopened.has_document("a")
