"""Composition root: the only place that knows which concrete classes are used.

Everything else depends on ports (Protocols). Swapping the vector DB, the
embedding model or the LLM means changing this file, or often just `.env`.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from app.answers.ports import LLM
from app.answers.service import AnswerService
from app.core.config import Settings
from app.documents.service import DocumentService
from app.infrastructure.fastembed_embedder import FastEmbedEmbedder
from app.infrastructure.ollama_llm import OllamaLLM
from app.infrastructure.sqlite_document_repository import SqliteDocumentRepository
from app.search.service import SearchService
from app.shared.ports import Embedder, VectorStore


@dataclass(frozen=True)
class Container:
    settings: Settings
    documents: DocumentService
    search: SearchService
    answers: AnswerService
    llm: LLM


def build_container(settings: Settings) -> Container:
    embedder = _load_embedder(settings.embedding_model, settings.model_cache_dir)
    store = _build_vector_store(settings, embedder.dimension)
    llm = OllamaLLM(
        settings.llm_base_url,
        settings.llm_model,
        settings.llm_timeout_s,
        settings.llm_context_tokens,
        settings.llm_max_tokens,
    )
    search = SearchService(store, embedder, settings.default_top_k)
    return Container(
        settings=settings,
        documents=DocumentService(
            SqliteDocumentRepository(settings.sqlite_path),
            store,
            embedder,
            settings.chunk_size,
            settings.chunk_overlap,
        ),
        search=search,
        answers=AnswerService(search, llm, settings.answer_top_k, settings.answer_min_score),
        llm=llm,
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

        return QdrantVectorStore(collection, vector_size, path=settings.qdrant_path, url=settings.qdrant_url)

    from app.infrastructure.chroma_store import ChromaVectorStore

    return ChromaVectorStore(settings.chroma_path, collection)


def _collection_name(settings: Settings) -> str:
    """One collection per embedding model.

    Vectors from different models are not comparable, so changing the model
    automatically starts a fresh collection instead of silently mixing them.
    """
    model_slug = re.sub(r"[^a-z0-9]+", "-", settings.embedding_model.split("/")[-1].lower())
    return f"{settings.collection_name}-{model_slug.strip('-')}"
