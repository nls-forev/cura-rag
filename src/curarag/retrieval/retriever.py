from __future__ import annotations

import math

from curarag.config import Settings, get_settings
from curarag.models import RetrievedHit
from curarag.retrieval.dense import DenseIndex
from curarag.retrieval.fusion import reciprocal_rank_fusion
from curarag.retrieval.rerank import rerank
from curarag.retrieval.sparse import SparseIndex


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HybridRetriever:
    """Dense (Qdrant) + sparse (BM25) -> RRF -> cross-encoder rerank -> top-k.

    The sparse index is rebuilt in memory from the chunks stored in Qdrant so
    both indexes always cover the identical chunk set."""

    def __init__(self, dense: DenseIndex | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.dense = dense or DenseIndex(settings=self.settings)
        self.sparse = SparseIndex(self.dense.scroll_chunks())

    def refresh_sparse(self) -> None:
        self.sparse = SparseIndex(self.dense.scroll_chunks())

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedHit]:
        top_k = top_k or self.settings.rerank_top_k
        dense_hits = self.dense.query(query, self.settings.dense_top_k)
        sparse_hits = self.sparse.search(query, self.settings.sparse_top_k)
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=self.settings.rrf_k)
        candidates = fused[: max(self.settings.dense_top_k, self.settings.sparse_top_k)]
        return rerank(query, candidates, top_k)

    def retrieval_strength(self, hits: list[RetrievedHit]) -> float:
        """Map the top reranker logit to a 0-1 confidence. The cross-encoder is
        the sharpest signal for whether any retrieved passage is on-topic."""
        if not hits:
            return 0.0
        top = max(h.rerank_score for h in hits if h.rerank_score is not None)
        return _sigmoid(top)
