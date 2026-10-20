"""Use case: semantic search over ingested documents."""

from app.shared.models import SearchHit
from app.shared.ports import Embedder, VectorStore


class SearchService:
    def __init__(self, store: VectorStore, embedder: Embedder, default_top_k: int):
        self._store = store
        self._embedder = embedder
        self._default_top_k = default_top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        min_score: float | None = None,
    ) -> list[SearchHit]:
        # The query must be embedded with the same model as the documents.
        query_vector = self._embedder.embed_query(query)
        hits = self._store.search(query_vector, top_k or self._default_top_k, document_id)
        # Filtering here (not in the store) keeps the behaviour identical across backends.
        if min_score is not None:
            hits = [h for h in hits if h.score >= min_score]
        return hits
