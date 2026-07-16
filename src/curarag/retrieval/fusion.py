from __future__ import annotations

from collections import defaultdict

from curarag.models import RetrievedHit


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedHit]],
    k: int = 60,
) -> list[RetrievedHit]:
    """Merge ranked hit lists by Reciprocal Rank Fusion. A chunk appearing in
    several lists accumulates score; carried scores from each list are preserved
    on the merged hit."""
    fused: dict[str, float] = defaultdict(float)
    merged: dict[str, RetrievedHit] = {}

    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk.chunk_id
            fused[cid] += 1.0 / (k + rank)
            if cid not in merged:
                merged[cid] = hit.model_copy(deep=True)
            else:
                existing = merged[cid]
                existing.dense_score = existing.dense_score or hit.dense_score
                existing.sparse_score = existing.sparse_score or hit.sparse_score

    ordered = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    out: list[RetrievedHit] = []
    for cid in ordered:
        hit = merged[cid]
        hit.rrf_score = fused[cid]
        out.append(hit)
    return out
