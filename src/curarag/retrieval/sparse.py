from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from curarag.models import Chunk, RetrievedHit

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class SparseIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, top_k: int = 30) -> list[RetrievedHit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedHit(chunk=self.chunks[i], sparse_score=float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]
