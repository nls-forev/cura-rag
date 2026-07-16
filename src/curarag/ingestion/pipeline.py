from __future__ import annotations

from pydantic import BaseModel

from curarag.config import ChunkingStrategy, Settings, get_settings
from curarag.embeddings import embed_texts
from curarag.ingestion.chunking import chunk_documents
from curarag.ingestion.dedupe import dedupe_chunks
from curarag.models import Document
from curarag.retrieval.dense import DenseIndex


class IngestReport(BaseModel):
    documents: int
    chunks_before_dedupe: int
    chunks_indexed: int
    strategy: str


def ingest_documents(
    docs: list[Document],
    dense: DenseIndex | None = None,
    strategy: ChunkingStrategy | None = None,
    settings: Settings | None = None,
    recreate: bool = False,
) -> IngestReport:
    settings = settings or get_settings()
    strategy = strategy or settings.chunking_strategy
    dense = dense or DenseIndex(settings=settings)

    chunks = chunk_documents(docs, strategy, settings)
    before = len(chunks)
    if not chunks:
        dense.ensure_collection(recreate=recreate)
        return IngestReport(documents=len(docs), chunks_before_dedupe=0, chunks_indexed=0, strategy=strategy.value)

    vectors = embed_texts([c.text for c in chunks])
    chunks, vectors = dedupe_chunks(chunks, vectors, settings.dedupe_threshold)

    dense.ensure_collection(recreate=recreate)
    dense.upsert(chunks, vectors)

    return IngestReport(
        documents=len(docs),
        chunks_before_dedupe=before,
        chunks_indexed=len(chunks),
        strategy=strategy.value,
    )
