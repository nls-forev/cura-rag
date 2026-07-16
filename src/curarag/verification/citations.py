from __future__ import annotations

import json
import re

from curarag.generation.llm import LLMClient, LLMError
from curarag.models import Citation, RetrievedHit

_MARKER = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"[^.!?\n]*[.!?\n]")

_JUDGE_SYSTEM = """You verify clinical citations. For each claim you are given the
exact source passage it cites. Decide whether the passage directly supports the
claim. Reply ONLY with a JSON array; one object per claim:
[{"index": <int>, "supported": <true|false>, "reason": "<short>"}]
Be strict: if the passage does not state the claim, supported is false."""


def parse_marker_claims(answer: str) -> list[tuple[str, int]]:
    """Return (claim_sentence, marker) pairs for every citation marker used."""
    pairs: list[tuple[str, int]] = []
    for sentence in _SENTENCE.findall(answer):
        markers = _MARKER.findall(sentence)
        clean = _MARKER.sub("", sentence)
        clean = re.sub(r"\s+([.!?,;:])", r"\1", clean)
        clean = re.sub(r"\s{2,}", " ", clean).strip()
        for m in markers:
            pairs.append((clean, int(m)))
    return pairs


def build_citations(answer: str, hits: list[RetrievedHit]) -> list[Citation]:
    """Map every distinct marker in the answer to its chunk metadata."""
    citations: list[Citation] = []
    seen: set[int] = set()
    for marker in (int(m) for m in _MARKER.findall(answer)):
        if marker in seen or marker < 1 or marker > len(hits):
            continue
        seen.add(marker)
        c = hits[marker - 1].chunk
        citations.append(
            Citation(
                marker=marker,
                chunk_id=c.chunk_id,
                source=c.source,
                title=c.title,
                section=c.section,
                url=c.url,
                quote=c.text[:280],
            )
        )
    return citations


def verify_citations(
    answer: str,
    hits: list[RetrievedHit],
    citations: list[Citation],
    llm: LLMClient | None = None,
) -> list[Citation]:
    """LLM-as-judge: mark each citation supported/unsupported by its passage."""
    pairs = parse_marker_claims(answer)
    valid = [(claim, m) for claim, m in pairs if 1 <= m <= len(hits)]
    if not valid:
        return citations

    payload = [
        {"index": i, "claim": claim, "cited_passage": hits[m - 1].chunk.text}
        for i, (claim, m) in enumerate(valid)
    ]
    llm = llm or LLMClient()
    try:
        raw = llm.complete(_JUDGE_SYSTEM, json.dumps(payload), temperature=0.0)
        verdicts = _parse_verdicts(raw)
    except (LLMError, json.JSONDecodeError):
        return citations

    # A marker is supported only if all claims citing it are supported.
    marker_ok: dict[int, bool] = {}
    marker_reason: dict[int, str] = {}
    for i, (_, marker) in enumerate(valid):
        v = verdicts.get(i, {"supported": True, "reason": ""})
        prev = marker_ok.get(marker, True)
        marker_ok[marker] = prev and bool(v["supported"])
        if not v["supported"]:
            marker_reason[marker] = v.get("reason", "")

    for cit in citations:
        cit.supported = marker_ok.get(cit.marker, True)
        cit.judge_reason = marker_reason.get(cit.marker, "")
    return citations


def _parse_verdicts(raw: str) -> dict[int, dict]:
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("no json array", raw, 0)
    data = json.loads(raw[start : end + 1])
    return {int(item["index"]): item for item in data}
