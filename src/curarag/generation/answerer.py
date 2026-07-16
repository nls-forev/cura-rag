from __future__ import annotations

from curarag.config import Settings, get_settings
from curarag.generation.llm import LLMClient, LLMError, get_llm
from curarag.generation.prompts import (
    ABSTAIN_TEXT,
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
)
from curarag.models import Answer, Confidence, RetrievedHit
from curarag.retrieval.retriever import HybridRetriever
from curarag.verification.citations import build_citations, verify_citations
from curarag.verification.confidence import compute_confidence


class Answerer:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever(settings=self.settings)
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def ask(self, question: str, verify: bool = True) -> Answer:
        hits = self.retriever.retrieve(question)
        strength = self.retriever.retrieval_strength(hits)

        if strength < self.settings.retrieval_confidence_threshold or not hits:
            return self._abstain(question, hits, strength)

        context = build_context(hits)
        try:
            text = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, context)).strip()
        except LLMError as exc:
            return Answer(
                question=question,
                answer=f"Generation failed: {exc}",
                abstained=True,
                retrieved_chunks=hits,
            )

        if ABSTAIN_TEXT[:40] in text or not text:
            return self._abstain(question, hits, strength, model_declined=True)

        citations = build_citations(text, hits)
        if verify and citations:
            citations = verify_citations(text, hits, citations, self.llm)

        confidence = compute_confidence(strength, citations, text, hits)
        return Answer(
            question=question,
            answer=text,
            abstained=False,
            citations=citations,
            confidence=confidence,
            retrieved_chunks=hits,
        )

    def _abstain(
        self,
        question: str,
        hits: list[RetrievedHit],
        strength: float,
        model_declined: bool = False,
    ) -> Answer:
        searched = sorted({h.chunk.source for h in hits}) or ["indexed corpus"]
        titles = sorted({h.chunk.title for h in hits})[:5]
        reason = (
            "The model could not ground an answer in the retrieved passages."
            if model_declined
            else "Retrieval confidence fell below the safety threshold."
        )
        msg = (
            f"Insufficient evidence to answer safely. {reason} "
            f"Searched sources: {', '.join(searched)}. "
            + (f"Closest indexed documents: {', '.join(titles)}. " if titles else "")
            + "Consult the official drug label or clinical guideline directly."
        )
        return Answer(
            question=question,
            answer=msg,
            abstained=True,
            citations=[],
            confidence=Confidence(
                retrieval=round(strength, 4),
                citation_coverage=0.0,
                completeness=0.0,
                composite=round(0.5 * strength, 4),
            ),
            retrieved_chunks=hits,
        )
