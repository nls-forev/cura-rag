from __future__ import annotations

import json
import re

from curarag.generation.llm import LLMClient, LLMError
from curarag.models import Answer

_CORRECTNESS_SYSTEM = """You grade a candidate clinical answer against a reference
answer. Reply ONLY with JSON: {"score": <0.0-1.0>, "reason": "<short>"}.
1.0 = fully correct and complete, 0.5 = partially correct, 0.0 = wrong or missing.
Judge clinical equivalence, not wording."""

_FAITHFULNESS_SYSTEM = """You check whether every claim in an answer is supported by
the provided context passages. Reply ONLY with JSON:
{"score": <0.0-1.0>, "reason": "<short>"}. 1.0 = all claims grounded in context,
0.0 = key claims not found in context (hallucinated)."""


def abstain_correct(answer: Answer, case: dict) -> bool:
    return bool(answer.abstained) == bool(case["must_abstain"])


def context_relevance(answer: Answer, case: dict) -> float | None:
    """Recall of the expected source documents among retrieved chunks."""
    expected = set(case.get("relevant_sources") or [])
    if not expected:
        return None
    retrieved = {h.chunk.doc_id for h in answer.retrieved_chunks}
    return len(expected & retrieved) / len(expected)


def citation_accuracy(answer: Answer) -> float | None:
    """Fraction of emitted citations the judge confirmed as supported."""
    if not answer.citations:
        return None
    supported = sum(1 for c in answer.citations if c.supported is not False)
    return supported / len(answer.citations)


def key_fact_recall(answer: Answer, case: dict) -> float | None:
    facts = case.get("key_facts") or []
    if not facts:
        return None
    text = answer.answer.lower()
    hits = sum(1 for f in facts if _normalize(f) in _normalize(text))
    return hits / len(facts)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def answer_correctness(answer: Answer, case: dict, llm: LLMClient | None) -> float | None:
    if case["must_abstain"] or llm is None:
        return None
    user = json.dumps(
        {"reference": case["expected_answer"], "candidate": answer.answer}
    )
    return _judge_score(_CORRECTNESS_SYSTEM, user, llm)


def faithfulness(answer: Answer, llm: LLMClient | None) -> float | None:
    if answer.abstained or llm is None:
        return None
    context = "\n\n".join(h.chunk.text for h in answer.retrieved_chunks)
    user = json.dumps({"context": context, "answer": answer.answer})
    return _judge_score(_FAITHFULNESS_SYSTEM, user, llm)


def _judge_score(system: str, user: str, llm: LLMClient) -> float | None:
    try:
        raw = llm.complete(system, user, temperature=0.0)
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start : end + 1])
        return max(0.0, min(1.0, float(data["score"])))
    except (LLMError, json.JSONDecodeError, KeyError, ValueError):
        return None
