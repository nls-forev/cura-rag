from __future__ import annotations

import json

from curarag.models import Chunk, RetrievedHit
from curarag.verification.citations import (
    build_citations,
    parse_marker_claims,
    verify_citations,
)


def _hits() -> list[RetrievedHit]:
    return [
        RetrievedHit(chunk=Chunk(chunk_id="c1", doc_id="d1", text="Max dose is 4 g per day.", source="guideline", title="Acetaminophen", section="Dosage")),
        RetrievedHit(chunk=Chunk(chunk_id="c2", doc_id="d2", text="Antidote is N-acetylcysteine.", source="guideline", title="Acetaminophen", section="Warnings")),
    ]


class FakeLLM:
    def __init__(self, verdicts):
        self._verdicts = verdicts

    def complete(self, system, user, temperature=None):
        return json.dumps(self._verdicts)


def test_parse_marker_claims_extracts_pairs():
    answer = "The max dose is 4 g [1]. The antidote is NAC [2]."
    pairs = parse_marker_claims(answer)
    assert any(claim.startswith("The max dose is 4 g") and m == 1 for claim, m in pairs)
    assert any("antidote" in claim.lower() and m == 2 for claim, m in pairs)


def test_build_citations_maps_markers_to_chunks():
    answer = "Max dose 4 g [1]. Antidote NAC [2]."
    cits = build_citations(answer, _hits())
    assert [c.marker for c in cits] == [1, 2]
    assert cits[0].chunk_id == "c1"
    assert cits[1].source == "guideline"


def test_build_citations_ignores_out_of_range_markers():
    answer = "Some claim [5]."
    assert build_citations(answer, _hits()) == []


def test_verify_flags_unsupported_citation():
    answer = "Max dose 4 g [1]. Antidote is aspirin [2]."
    cits = build_citations(answer, _hits())
    fake = FakeLLM([
        {"index": 0, "supported": True, "reason": "matches"},
        {"index": 1, "supported": False, "reason": "passage says NAC not aspirin"},
    ])
    verified = verify_citations(answer, _hits(), cits, llm=fake)
    by_marker = {c.marker: c for c in verified}
    assert by_marker[1].supported is True
    assert by_marker[2].supported is False
    assert "aspirin" in by_marker[2].judge_reason
