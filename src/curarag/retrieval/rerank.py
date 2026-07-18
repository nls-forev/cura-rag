from __future__ import annotations

from functools import lru_cache

from curarag.config import get_settings
from curarag.models import RetrievedHit


@lru_cache
def get_reranker():
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    return CrossEncoder(settings.reranker_model, device=settings.model_device)


def rerank(query: str, hits: list[RetrievedHit], top_k: int) -> list[RetrievedHit]:
    if not hits:
        return []
    scores = get_reranker().predict([(query, hit.chunk.text) for hit in hits])
    for hit, score in zip(hits, scores, strict=True):
        hit.rerank_score = float(score)
    return sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)[:top_k]
