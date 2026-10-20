"""Use-case tests with in-memory fakes: no database, no model, no LLM. Instant."""

import pytest

from app.answers.exceptions import LLMUnavailableError
from app.answers.service import NO_CONTEXT_ANSWER, AnswerService
from app.documents.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmptyDocumentError,
    PDFParseError,
)
from app.documents.service import DocumentService
from app.search.service import SearchService
from app.shared.models import Chunk
from tests.fakes import FakeEmbedder, FakeLLM, FakeRepository, FakeVectorStore
from tests.pdf_factory import make_pdf


def documents(store=None, repo=None, chunk_size=100, overlap=10):
    return DocumentService(repo or FakeRepository(), store or FakeVectorStore(), FakeEmbedder(), chunk_size, overlap)


# --- documents ------------------------------------------------------------

def test_ingest_stores_chunks_and_metadata():
    store, repo = FakeVectorStore(), FakeRepository()
    pdf = make_pdf("Hello plumbing world")
    doc = documents(store, repo).ingest_pdf(pdf, "a.pdf")

    assert (doc.pages, doc.chunks, doc.size_bytes) == (1, 1, len(pdf))
    assert store.chunks[0].filename == "a.pdf"
    assert repo.get(doc.document_id) == doc


def test_same_content_gets_same_id():
    a = documents().ingest_pdf(make_pdf("Same"), "a.pdf")
    b = documents().ingest_pdf(make_pdf("Same"), "b.pdf")
    assert a.document_id == b.document_id


def test_duplicate_rejected_even_with_new_name():
    store = FakeVectorStore()
    service = documents(store)
    service.ingest_pdf(make_pdf("Same content"), "a.pdf")
    with pytest.raises(DuplicateDocumentError):
        service.ingest_pdf(make_pdf("Same content"), "renamed.pdf")
    assert len(store.chunks) == 1  # nothing stored twice


def test_repeated_headers_are_not_stored():
    topics = ["apples", "bananas", "cherries", "dates"]
    pdf = make_pdf(*[f"ACME Manual page {n}\nAll about {t}" for n, t in enumerate(topics, 1)])
    store = FakeVectorStore()
    documents(store).ingest_pdf(pdf, "manual.pdf")
    assert [c.text for c in store.chunks] == [f"All about {t}" for t in topics]


def test_ingest_rejects_garbage():
    with pytest.raises(PDFParseError):
        documents().ingest_pdf(b"nope", "x.pdf")


def test_ingest_rejects_pdf_without_text():
    repo = FakeRepository()
    with pytest.raises(EmptyDocumentError):
        documents(repo=repo).ingest_pdf(make_pdf(""), "x.pdf")
    assert repo.list() == []  # nothing half-registered


def test_delete_removes_chunks_and_metadata_and_allows_reupload():
    store, repo = FakeVectorStore(), FakeRepository()
    service = documents(store, repo)
    pdf = make_pdf("Delete me")
    doc = service.ingest_pdf(pdf, "a.pdf")

    service.delete_document(doc.document_id)

    assert store.chunks == [] and repo.list() == []
    assert service.ingest_pdf(pdf, "a.pdf").document_id == doc.document_id


def test_get_and_delete_unknown_document():
    with pytest.raises(DocumentNotFoundError):
        documents().get_document("missing")
    with pytest.raises(DocumentNotFoundError):
        documents().delete_document("missing")


# --- search ---------------------------------------------------------------

def test_search_uses_default_top_k_and_passes_filter():
    store = FakeVectorStore()
    SearchService(store, FakeEmbedder(), default_top_k=7).search("anything", document_id="doc1")
    assert store.last_search == {"top_k": 7, "document_id": "doc1"}


def test_search_drops_hits_below_min_score():
    store = FakeVectorStore(scores=[0.9, 0.5, 0.1])
    store.chunks = [Chunk(str(i), "d", "f.pdf", 1, f"t{i}") for i in range(3)]
    hits = SearchService(store, FakeEmbedder(), default_top_k=5).search("q", min_score=0.5)
    assert [h.score for h in hits] == [0.9, 0.5]


# --- answers --------------------------------------------------------------

def answers(scores, llm=None, top_k=4, min_score=0.4):
    store = FakeVectorStore(scores=scores)
    store.chunks = [Chunk(str(i), "d", "f.pdf", i + 1, f"passage {i}") for i in range(len(scores))]
    llm = llm or FakeLLM()
    return AnswerService(SearchService(store, FakeEmbedder(), 5), llm, top_k, min_score), llm


def test_answer_uses_only_relevant_passages_with_citations():
    service, llm = answers(scores=[0.9, 0.8, 0.2])
    answer = service.answer("question?")

    assert [h.chunk.page for h in answer.sources] == [1, 2]  # 0.2 < min_score dropped
    assert "[1] (f.pdf, page 1)" in llm.prompts[0] and "[2] (f.pdf, page 2)" in llm.prompts[0]
    assert "Question: question?" in llm.prompts[0]


def test_answer_without_relevant_passages_skips_llm():
    service, llm = answers(scores=[0.1, 0.05])
    answer = service.answer("unrelated?")
    assert (answer.text, answer.sources, llm.prompts) == (NO_CONTEXT_ANSWER, [], [])


def test_stream_answer_yields_tokens():
    service, _ = answers(scores=[0.9])
    streamed = service.stream_answer("q")
    assert len(streamed.sources) == 1
    assert "".join(streamed.tokens).strip() == FakeLLM.ANSWER


def test_stream_answer_fails_fast_when_llm_down():
    service, _ = answers(scores=[0.9], llm=FakeLLM(available=False))
    with pytest.raises(LLMUnavailableError):
        service.stream_answer("q")
