"""Qdrant implementation of the `VectorStore` port.

Runs in embedded (local) mode: data is persisted to `path`, no server needed.
To use a Qdrant server instead, construct the client with `QdrantClient(url=...)`.
"""

import uuid

from qdrant_client import QdrantClient, models

from app.shared.models import Chunk, SearchHit
from app.shared.ports import Vector


class QdrantVectorStore:
    def __init__(self, path: str, collection_name: str, vector_size: int):
        self._client = QdrantClient(path=path)
        self._collection = collection_name
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def add(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        if not chunks:
            return
        self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=_point_id(chunk.id),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "filename": chunk.filename,
                        "page": chunk.page,
                        "text": chunk.text,
                    },
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

    def search(
        self, query_vector: Vector, top_k: int, document_id: str | None = None
    ) -> list[SearchHit]:
        res = self._client.query_points(
            self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=_document_filter(document_id),
            with_payload=True,
        )
        return [
            SearchHit(
                chunk=Chunk(
                    id=p.payload["chunk_id"],
                    document_id=p.payload["document_id"],
                    filename=p.payload["filename"],
                    page=p.payload["page"],
                    text=p.payload["text"],
                ),
                score=round(p.score, 4),  # Qdrant already returns cosine similarity
            )
            for p in res.points
        ]

    def has_document(self, document_id: str) -> bool:
        res = self._client.count(
            self._collection, count_filter=_document_filter(document_id), exact=True
        )
        return res.count > 0


def _point_id(chunk_id: str) -> str:
    """Qdrant IDs must be ints or UUIDs; derive a stable UUID from our string ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _document_filter(document_id: str | None) -> models.Filter | None:
    if document_id is None:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    )
