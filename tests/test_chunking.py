from __future__ import annotations

from curarag.config import ChunkingStrategy, get_settings
from curarag.ingestion.chunking import chunk_document
from curarag.ingestion.dedupe import dedupe_chunks
from curarag.models import Chunk, Document, Section


def _doc() -> Document:
    return Document(
        doc_id="guideline:test",
        source="guideline",
        title="Test Drug",
        url="http://example.test",
        sections=[
            Section(heading="Dosage", text="Take 500 mg every 8 hours. Do not exceed 3 grams."),
            Section(heading="Warnings", text="May cause drowsiness. Avoid alcohol while taking it."),
        ],
    )


def test_recursive_preserves_section_metadata():
    chunks = chunk_document(_doc(), ChunkingStrategy.recursive, get_settings())
    assert chunks
    assert all(c.strategy == "recursive" for c in chunks)
    assert {c.section for c in chunks} == {"Dosage", "Warnings"}
    assert all(c.doc_id == "guideline:test" for c in chunks)


def test_chunk_ids_are_unique_and_stable():
    chunks = chunk_document(_doc(), ChunkingStrategy.recursive, get_settings())
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    again = chunk_document(_doc(), ChunkingStrategy.recursive, get_settings())
    assert ids == [c.chunk_id for c in again]


def test_fixed_respects_size():
    settings = get_settings().model_copy(update={"chunk_size": 40, "chunk_overlap": 5})
    chunks = chunk_document(_doc(), ChunkingStrategy.fixed, settings)
    assert chunks
    assert all(len(c.text) <= 40 for c in chunks)


def test_dedupe_drops_near_duplicates():
    chunks = [
        Chunk(chunk_id="a", doc_id="d", text="alpha", source="s", title="t"),
        Chunk(chunk_id="b", doc_id="d", text="alpha copy", source="s", title="t"),
        Chunk(chunk_id="c", doc_id="d", text="different", source="s", title="t"),
    ]
    vectors = [[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]]
    kept, kept_vecs = dedupe_chunks(chunks, vectors, threshold=0.95)
    assert [c.chunk_id for c in kept] == ["a", "c"]
    assert len(kept_vecs) == 2
