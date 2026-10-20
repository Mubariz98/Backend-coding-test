"""Use-case tests with in-memory fakes: no database, no embedding model, instant."""

import pytest

from app.documents.exceptions import DuplicateDocumentError, EmptyDocumentError, PDFParseError
from app.documents.service import IngestionService
from app.search.service import SearchService
from app.shared.models import Chunk, SearchHit
from tests.pdf_factory import make_pdf


class FakeEmbedder:
    """Satisfies the Embedder port with constant vectors."""

    dimension = 2

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeVectorStore:
    """Satisfies the VectorStore port; records what it was given."""

    def __init__(self, scores=None):
        self.chunks: list[Chunk] = []
        self.scores = scores  # optional fixed scores returned by search
        self.last_search: dict = {}

    def add(self, chunks, vectors):
        assert len(chunks) == len(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector, top_k, document_id=None):
        self.last_search = {"top_k": top_k, "document_id": document_id}
        scores = self.scores or [1.0] * len(self.chunks)
        return [SearchHit(chunk=c, score=s) for c, s in zip(self.chunks, scores)][:top_k]

    def has_document(self, document_id):
        return any(c.document_id == document_id for c in self.chunks)


def ingestion(store, chunk_size=100, overlap=10):
    return IngestionService(store, FakeEmbedder(), chunk_size, overlap)


def test_ingest_stores_chunks():
    store = FakeVectorStore()
    doc = ingestion(store).ingest_pdf(make_pdf("Hello plumbing world"), "a.pdf")
    assert (doc.pages, doc.chunks, len(store.chunks)) == (1, 1, 1)
    assert store.chunks[0].filename == "a.pdf"


def test_same_content_gets_same_id():
    a = ingestion(FakeVectorStore()).ingest_pdf(make_pdf("Same"), "a.pdf")
    b = ingestion(FakeVectorStore()).ingest_pdf(make_pdf("Same"), "b.pdf")
    assert a.document_id == b.document_id


def test_duplicate_rejected_even_with_new_name():
    store = FakeVectorStore()
    service = ingestion(store)
    service.ingest_pdf(make_pdf("Same content"), "a.pdf")
    with pytest.raises(DuplicateDocumentError):
        service.ingest_pdf(make_pdf("Same content"), "renamed.pdf")
    assert len(store.chunks) == 1  # nothing stored twice


def test_repeated_headers_are_not_stored():
    topics = ["apples", "bananas", "cherries", "dates"]
    pdf = make_pdf(*[f"ACME Manual page {n}\nAll about {t}" for n, t in enumerate(topics, 1)])
    store = FakeVectorStore()
    ingestion(store).ingest_pdf(pdf, "manual.pdf")
    assert [c.text for c in store.chunks] == [f"All about {t}" for t in topics]


def test_ingest_rejects_garbage():
    with pytest.raises(PDFParseError):
        ingestion(FakeVectorStore()).ingest_pdf(b"nope", "x.pdf")


def test_ingest_rejects_pdf_without_text():
    with pytest.raises(EmptyDocumentError):
        ingestion(FakeVectorStore()).ingest_pdf(make_pdf(""), "x.pdf")


def test_search_uses_default_top_k_and_passes_filter():
    store = FakeVectorStore()
    SearchService(store, FakeEmbedder(), default_top_k=7).search("anything", document_id="doc1")
    assert store.last_search == {"top_k": 7, "document_id": "doc1"}


def test_search_drops_hits_below_min_score():
    store = FakeVectorStore(scores=[0.9, 0.5, 0.1])
    store.chunks = [Chunk(str(i), "d", "f.pdf", 1, f"t{i}") for i in range(3)]
    hits = SearchService(store, FakeEmbedder(), default_top_k=5).search("q", min_score=0.5)
    assert [h.score for h in hits] == [0.9, 0.5]
