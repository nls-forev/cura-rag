from __future__ import annotations

import re

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from curarag.config import ChunkingStrategy, Settings, get_settings
from curarag.models import Chunk, Document


def _mk_chunk(doc: Document, section: str, text: str, index: int, strategy: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}::{strategy}::{index}",
        doc_id=doc.doc_id,
        text=text.strip(),
        source=doc.source,
        title=doc.title,
        section=section,
        url=doc.url,
        chunk_index=index,
        strategy=strategy,
    )


def _fixed(doc: Document, settings: Settings) -> list[Chunk]:
    full = "\n\n".join(f"{s.heading}\n{s.text}" for s in doc.sections)
    size, overlap = settings.chunk_size, settings.chunk_overlap
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(full), step)):
        piece = full[start : start + size]
        if piece.strip():
            chunks.append(_mk_chunk(doc, "", piece, i, "fixed"))
        if start + size >= len(full):
            break
    return chunks


def _recursive(doc: Document, settings: Settings) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    index = 0
    for section in doc.sections:
        for piece in splitter.split_text(section.text):
            if piece.strip():
                chunks.append(_mk_chunk(doc, section.heading, piece, index, "recursive"))
                index += 1
    return chunks


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _semantic(doc: Document, settings: Settings) -> list[Chunk]:
    """Group adjacent sentences, cutting where embedding similarity drops below a
    percentile threshold of the section's own boundary gradient."""
    from curarag.embeddings import embed_texts

    chunks: list[Chunk] = []
    index = 0
    for section in doc.sections:
        sentences = _split_sentences(section.text)
        if len(sentences) <= 1:
            if section.text.strip():
                chunks.append(_mk_chunk(doc, section.heading, section.text, index, "semantic"))
                index += 1
            continue

        vecs = np.asarray(embed_texts(sentences))
        sims = [
            float(np.dot(vecs[i], vecs[i + 1]) /
                  (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i + 1]) + 1e-8))
            for i in range(len(sentences) - 1)
        ]
        cut_threshold = float(np.percentile(sims, 25)) if sims else 0.0

        buffer = [sentences[0]]
        for i, sim in enumerate(sims):
            over_budget = sum(len(s) for s in buffer) > settings.chunk_size
            if sim < cut_threshold or over_budget:
                chunks.append(_mk_chunk(doc, section.heading, " ".join(buffer), index, "semantic"))
                index += 1
                buffer = []
            buffer.append(sentences[i + 1])
        if buffer:
            chunks.append(_mk_chunk(doc, section.heading, " ".join(buffer), index, "semantic"))
            index += 1
    return chunks


_STRATEGIES = {
    ChunkingStrategy.fixed: _fixed,
    ChunkingStrategy.recursive: _recursive,
    ChunkingStrategy.semantic: _semantic,
}


def chunk_document(
    doc: Document,
    strategy: ChunkingStrategy | None = None,
    settings: Settings | None = None,
) -> list[Chunk]:
    settings = settings or get_settings()
    strategy = strategy or settings.chunking_strategy
    return _STRATEGIES[strategy](doc, settings)


def chunk_documents(
    docs: list[Document],
    strategy: ChunkingStrategy | None = None,
    settings: Settings | None = None,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, strategy, settings))
    return out
