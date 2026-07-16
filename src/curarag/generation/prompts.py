from __future__ import annotations

from curarag.models import RetrievedHit

ABSTAIN_TEXT = (
    "I don't have sufficient evidence in the indexed sources to answer this safely."
)

SYSTEM_PROMPT = f"""You are CuraRAG, a clinical drug-information assistant.

Rules you must never break:
- Answer ONLY using the numbered context blocks provided. Do not use outside knowledge.
- Every clinical claim (dose, indication, contraindication, interaction, warning) must
  carry an inline citation like [1] or [2] pointing at the context block it came from.
- If the context does not contain enough information to answer safely, respond with
  EXACTLY this and nothing else: "{ABSTAIN_TEXT}"
- Never guess or infer a dose, contraindication, or interaction that is not stated.
- Be concise and clinical. Do not add disclaimers beyond what the context states.

Format:
- Plain prose or short bullets, each factual sentence ending with its citation marker(s).
"""


def build_context(hits: list[RetrievedHit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        header = f"[{i}] source={c.source} | title={c.title} | section={c.section or 'n/a'}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
