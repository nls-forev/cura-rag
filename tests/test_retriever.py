from __future__ import annotations

from curarag.config import get_settings
from curarag.models import Chunk, RetrievedHit
from curarag.retrieval import retriever as retriever_mod
from curarag.retrieval.retriever import HybridRetriever


def _hit(cid: str, dense=None, sparse=None) -> RetrievedHit:
    chunk = Chunk(chunk_id=cid, doc_id="d", text=f"text {cid}", source="s", title="t")
    return RetrievedHit(chunk=chunk, dense_score=dense, sparse_score=sparse)


class FakeDense:
    def __init__(self, hits):
        self._hits = hits

    def query(self, query, top_k):
        return self._hits

    def scroll_chunks(self):
        return []


def test_retrieve_orchestrates_fusion_and_rerank(monkeypatch):
    dense_hits = [_hit("a", dense=0.9), _hit("b", dense=0.8)]
    fake = FakeDense(dense_hits)

    def fake_rerank(query, hits, top_k):
        for i, h in enumerate(hits):
            h.rerank_score = float(len(hits) - i)
        return hits[:top_k]

    monkeypatch.setattr(retriever_mod, "rerank", fake_rerank)

    r = HybridRetriever(dense=fake, settings=get_settings().model_copy(update={"rerank_top_k": 1}))
    out = r.retrieve("dosage question")
    assert len(out) == 1
    assert out[0].chunk.chunk_id == "a"


def test_retrieval_strength_zero_without_hits():
    fake = FakeDense([])
    r = HybridRetriever(dense=fake, settings=get_settings())
    assert r.retrieval_strength([]) == 0.0


def test_retrieval_strength_monotonic_in_score():
    fake = FakeDense([])
    r = HybridRetriever(dense=fake, settings=get_settings())
    low = _hit("a")
    low.rerank_score = -2.0
    high = _hit("b")
    high.rerank_score = 4.0
    assert r.retrieval_strength([high]) > r.retrieval_strength([low])
