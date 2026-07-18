"""Gradio front end for CuraRAG. Kept separate from the FastAPI service: this
module only wires a UI around the same Answerer, so the API and the Space share
one code path and neither imports the other."""

from __future__ import annotations

from functools import lru_cache

from curarag.generation.answerer import Answerer
from curarag.generation.llm import LLMError
from curarag.models import Answer

INTRO = (
    "# CuraRAG\n"
    "Answers **only** from verified drug labels and clinical guidelines. Cites the "
    "exact source passage for every claim, verifies those citations, and abstains "
    "when the evidence isn't there.\n\n"
    "_Not medical advice. A portfolio demo over a small public corpus._"
)

EXAMPLES = [
    "What is the maximum daily dose of acetaminophen for adults?",
    "What is the target INR for a patient on warfarin for atrial fibrillation?",
    "Is it safe for a patient taking warfarin to also take ibuprofen?",
    "What is the correct starting dose of insulin glargine?",
]


@lru_cache
def _answerer() -> Answerer:
    return Answerer()


def answer_to_markdown(ans: Answer) -> tuple[str, str]:
    """Render an Answer as (answer_markdown, evidence_markdown)."""
    if ans.abstained:
        head = f"> **Abstained.** {ans.answer}"
    else:
        head = ans.answer

    conf = ans.confidence
    if conf:
        head += (
            f"\n\n---\n**Confidence {conf.composite:.2f}** "
            f"(retrieval {conf.retrieval:.2f} · citations verified "
            f"{conf.citation_coverage:.2f})"
        )

    if not ans.citations:
        return head, "_No citations._"

    lines = ["### Citations"]
    for c in ans.citations:
        flag = "unsupported" if c.supported is False else "verified"
        lines.append(f"**[{c.marker}]** _{flag}_ — {c.source} · {c.title} · {c.section or 'n/a'}")
        if c.quote:
            lines.append(f"> {c.quote}")
    return head, "\n\n".join(lines)


def ask(question: str) -> tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "Enter a question.", ""
    try:
        ans = _answerer().ask(question)
    except LLMError as exc:
        return f"**LLM unavailable.** {exc}", ""
    return answer_to_markdown(ans)


def build_demo():
    import gradio as gr

    with gr.Blocks(title="CuraRAG") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            question = gr.Textbox(
                label="Clinical question",
                placeholder="What is the max daily dose of acetaminophen?",
                scale=4,
                autofocus=True,
            )
            submit = gr.Button("Ask", variant="primary", scale=1)
        gr.Examples(EXAMPLES, inputs=question)
        answer_md = gr.Markdown()
        evidence_md = gr.Markdown()

        submit.click(ask, inputs=question, outputs=[answer_md, evidence_md])
        question.submit(ask, inputs=question, outputs=[answer_md, evidence_md])

    return demo
