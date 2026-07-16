from __future__ import annotations

from pydantic import BaseModel, Field

from curarag.models import Answer


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    verify: bool = True


class AskResponse(Answer):
    pass


class IngestDrugRequest(BaseModel):
    drug_names: list[str] = Field(default_factory=list)
    recreate: bool = False


class IngestResponse(BaseModel):
    documents: int
    chunks_indexed: int
    strategy: str


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    title: str
    url: str = ""
    chunks: int


class DocumentsResponse(BaseModel):
    total_chunks: int
    documents: list[DocumentInfo]


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    collection: str
