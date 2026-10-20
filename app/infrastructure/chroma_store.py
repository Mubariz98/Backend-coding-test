"""ChromaDB implementation of the `VectorStore` port.

Vectors are computed by the Embedder and passed in, so Chroma is used purely
as storage + nearest-neighbour index (no built-in embedding function).
"""

import chromadb

from app.shared.models import Chunk, SearchHit
from app.shared.ports import Vector


class ChromaVectorStore:
    def __init__(self, path: str, collection_name: str):
        self._client = chromadb.PersistentClient(
            path=path, settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {"document_id": c.document_id, "filename": c.filename, "page": c.page}
                for c in chunks
            ],
        )

    def search(
        self, query_vector: Vector, top_k: int, document_id: str | None = None
    ) -> list[SearchHit]:
        res = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"document_id": document_id} if document_id else None,
        )
        # Chroma returns one list per query; we only send one query.
        return [
            SearchHit(
                chunk=Chunk(
                    id=chunk_id,
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    page=meta["page"],
                    text=text,
                ),
                score=round(1 - distance, 4),  # cosine distance -> similarity
            )
            for chunk_id, text, meta, distance in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})
