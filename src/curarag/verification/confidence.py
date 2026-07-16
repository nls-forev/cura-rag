from __future__ import annotations

from curarag.models import Citation, Confidence, RetrievedHit


def compute_confidence(
    retrieval_strength: float,
    citations: list[Citation],
    answer: str,
    hits: list[RetrievedHit],
) -> Confidence:
    """Composite of three signals:
    - retrieval: how on-topic the best retrieved passage is (reranker-derived).
    - citation_coverage: fraction of used citations the judge confirmed supported.
    - completeness: whether the answer actually cited evidence at all.
    """
    if citations:
        verified = sum(1 for c in citations if c.supported is not False)
        citation_coverage = verified / len(citations)
    else:
        citation_coverage = 0.0

    completeness = 1.0 if citations else 0.0

    composite = 0.5 * retrieval_strength + 0.35 * citation_coverage + 0.15 * completeness
    return Confidence(
        retrieval=round(retrieval_strength, 4),
        citation_coverage=round(citation_coverage, 4),
        completeness=round(completeness, 4),
        composite=round(composite, 4),
    )
