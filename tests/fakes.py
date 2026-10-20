"""In-memory fakes for every port. They make use-case tests instant and deterministic."""

from app.answers.exceptions import LLMUnavailableError
from app.shared.models import SearchHit


class FakeEmbedder:
    dimension = 2

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self, scores=None):
        self.chunks = []
        self.scores = scores  # optional fixed scores returned by search
        self.last_search: dict = {}

    def add(self, chunks, vectors):
        assert len(chunks) == len(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector, top_k, document_id=None):
        self.last_search = {"top_k": top_k, "document_id": document_id}
        scores = self.scores or [1.0] * len(self.chunks)
        hits = [SearchHit(chunk=c, score=s) for c, s in zip(self.chunks, scores)]
        return [h for h in hits if document_id in (None, h.chunk.document_id)][:top_k]

    def delete_document(self, document_id):
        self.chunks = [c for c in self.chunks if c.document_id != document_id]


class FakeRepository:
    def __init__(self):
        self.documents = {}

    def add(self, document):
        self.documents[document.document_id] = document

    def get(self, document_id):
        return self.documents.get(document_id)

    def list(self):
        return sorted(self.documents.values(), key=lambda d: d.uploaded_at, reverse=True)

    def delete(self, document_id):
        self.documents.pop(document_id, None)


class FakeLLM:
    """Returns a fixed answer and records the prompts it was given."""

    model = "fake-llm"
    ANSWER = "Two years [1]."

    def __init__(self, available=True):
        self.available = available
        self.prompts: list[str] = []

    def generate(self, system, prompt):
        if not self.available:
            raise LLMUnavailableError("fake LLM is down")
        self.prompts.append(prompt)
        return self.ANSWER

    def stream(self, system, prompt):
        for word in self.generate(system, prompt).split(" "):
            yield word + " "

    def is_available(self):
        return self.available
