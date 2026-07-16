from __future__ import annotations

from curarag.models import Chunk, RetrievedHit
from curarag.retrieval.fusion import reciprocal_rank_fusion


def _hit(cid: str, dense=None, sparse=None) -> RetrievedHit:
    chunk = Chunk(chunk_id=cid, doc_id="d", text=cid, source="s", title="t")
    return RetrievedHit(chunk=chunk, dense_score=dense, sparse_score=sparse)


def test_rrf_rewards_agreement_across_lists():
    dense = [_hit("a", dense=0.9), _hit("b", dense=0.8), _hit("c", dense=0.7)]
    sparse = [_hit("b", sparse=5.0), _hit("a", sparse=4.0), _hit("d", sparse=3.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [h.chunk.chunk_id for h in fused]
    # a and b appear in both lists near the top, so they must outrank c and d.
    assert set(ids[:2]) == {"a", "b"}
    assert ids[-1] in {"c", "d"}


def test_rrf_merges_scores_from_both_lists():
    dense = [_hit("a", dense=0.9)]
    sparse = [_hit("a", sparse=4.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert len(fused) == 1
    hit = fused[0]
    assert hit.dense_score == 0.9
    assert hit.sparse_score == 4.0
    assert hit.rrf_score == 2 * (1 / 61)


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
