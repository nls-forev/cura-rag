from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A normalized source document before chunking."""

    doc_id: str
    source: str
    title: str
    url: str = ""
    sections: list[Section] = Field(default_factory=list)


class Section(BaseModel):
    heading: str
    text: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    source: str
    title: str
    section: str = ""
    url: str = ""
    chunk_index: int = 0
    strategy: str = ""


class RetrievedHit(BaseModel):
    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else (self.rrf_score or 0.0)


class Citation(BaseModel):
    marker: int
    chunk_id: str
    source: str
    title: str
    section: str = ""
    url: str = ""
    quote: str = ""
    supported: bool | None = None
    judge_reason: str = ""


class Confidence(BaseModel):
    retrieval: float
    citation_coverage: float
    completeness: float
    composite: float


class Answer(BaseModel):
    question: str
    answer: str
    abstained: bool = False
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence | None = None
    retrieved_chunks: list[RetrievedHit] = Field(default_factory=list)


Document.model_rebuild()
