"""Composition root: the only place that knows which concrete classes are used.

Everything else depends on the ports in `app/shared/ports.py`. Swapping the
vector DB or the embedding model means changing this file (or just `.env`).
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import Settings
from app.documents.service import IngestionService
from app.infrastructure.fastembed_embedder import FastEmbedEmbedder
from app.search.service import SearchService
from app.shared.ports import Embedder, VectorStore


@dataclass(frozen=True)
class Container:
    settings: Settings
    ingestion: IngestionService
    search: SearchService


def build_container(settings: Settings) -> Container:
    embedder = _load_embedder(settings.embedding_model, settings.model_cache_dir)
    store = _build_vector_store(settings, embedder.dimension)
    return Container(
        settings=settings,
        ingestion=IngestionService(store, embedder, settings.chunk_size, settings.chunk_overlap),
        search=SearchService(store, embedder, settings.default_top_k),
    )


@lru_cache
def _load_embedder(model_name: str, cache_dir: str) -> Embedder:
    """Loading a model is slow and it never changes, so share one per model."""
    return FastEmbedEmbedder(model_name, cache_dir)


def _build_vector_store(settings: Settings, vector_size: int) -> VectorStore:
    collection = _collection_name(settings)
    # Imports are local so only the selected backend's library gets loaded.
    if settings.vector_db == "qdrant":
        from app.infrastructure.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(settings.qdrant_path, collection, vector_size)

    from app.infrastructure.chroma_store import ChromaVectorStore

    return ChromaVectorStore(settings.chroma_path, collection)


def _collection_name(settings: Settings) -> str:
    """One collection per embedding model.

    Vectors from different models are not comparable, so changing the model
    automatically starts a fresh collection instead of silently mixing them.
    """
    model_slug = re.sub(r"[^a-z0-9]+", "-", settings.embedding_model.split("/")[-1].lower())
    return f"{settings.collection_name}-{model_slug.strip('-')}"
