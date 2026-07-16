from __future__ import annotations

import numpy as np

from curarag.config import get_settings
from curarag.models import Chunk


def dedupe_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    threshold: float | None = None,
) -> tuple[list[Chunk], list[list[float]]]:
    """Drop near-duplicate chunks whose cosine similarity to an already-kept
    chunk exceeds the threshold. Returns the surviving chunks and their vectors
    aligned by index."""
    if not chunks:
        return [], []
    threshold = threshold if threshold is not None else get_settings().dedupe_threshold

    mat = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    unit = mat / norms

    kept_idx: list[int] = []
    kept_unit: list[np.ndarray] = []
    for i in range(len(chunks)):
        if kept_unit:
            sims = np.asarray(kept_unit) @ unit[i]
            if float(sims.max()) > threshold:
                continue
        kept_idx.append(i)
        kept_unit.append(unit[i])

    return [chunks[i] for i in kept_idx], [embeddings[i] for i in kept_idx]
