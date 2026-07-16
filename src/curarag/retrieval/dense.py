from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from curarag.config import Settings, get_settings
from curarag.embeddings import embed_query
from curarag.models import Chunk, RetrievedHit

_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class DenseIndex:
    def __init__(self, client: QdrantClient | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = client or QdrantClient(url=self.settings.qdrant_url)
        self.collection = self.settings.qdrant_collection

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dim, distance=Distance.COSINE
                ),
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(self.collection, points=points)

    def query(self, query: str, top_k: int | None = None) -> list[RetrievedHit]:
        top_k = top_k or self.settings.dense_top_k
        vector = embed_query(query)
        result = self.client.query_points(self.collection, query=vector, limit=top_k, with_payload=True)
        return [
            RetrievedHit(chunk=Chunk(**point.payload), dense_score=float(point.score))
            for point in result.points
        ]

    def scroll_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            chunks.extend(Chunk(**r.payload) for r in records)
            if offset is None:
                break
        return chunks

    def count(self) -> int:
        return self.client.count(self.collection).count
