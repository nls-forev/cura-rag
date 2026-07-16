from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, HTTPException

from curarag.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    IngestDrugRequest,
    IngestResponse,
)
from curarag.generation.answerer import Answerer
from curarag.generation.llm import LLMError
from curarag.ingestion.loaders import load_openfda_labels
from curarag.ingestion.pipeline import ingest_documents
from curarag.retrieval.dense import DenseIndex

router = APIRouter()


def _dense() -> DenseIndex:
    return DenseIndex()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    dense = _dense()
    try:
        count = dense.count() if dense.client.collection_exists(dense.collection) else 0
        status = "ok"
    except Exception:  # noqa: BLE001 - health must not raise
        count, status = 0, "degraded"
    return HealthResponse(status=status, indexed_chunks=count, collection=dense.collection)


@router.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        answer = Answerer().ask(req.question, verify=req.verify)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AskResponse(**answer.model_dump())


@router.post("/v1/ingest", response_model=IngestResponse)
def ingest(req: IngestDrugRequest) -> IngestResponse:
    if not req.drug_names:
        raise HTTPException(status_code=400, detail="drug_names must not be empty")
    docs = load_openfda_labels(drug_names=req.drug_names)
    if not docs:
        raise HTTPException(status_code=404, detail="No labels found for the given drug names")
    report = ingest_documents(docs, recreate=req.recreate)
    return IngestResponse(
        documents=report.documents,
        chunks_indexed=report.chunks_indexed,
        strategy=report.strategy,
    )


@router.get("/v1/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    dense = _dense()
    if not dense.client.collection_exists(dense.collection):
        return DocumentsResponse(total_chunks=0, documents=[])
    chunks = dense.scroll_chunks()
    grouped: dict[str, list] = defaultdict(list)
    for c in chunks:
        grouped[c.doc_id].append(c)
    infos = [
        DocumentInfo(
            doc_id=doc_id,
            source=items[0].source,
            title=items[0].title,
            url=items[0].url,
            chunks=len(items),
        )
        for doc_id, items in sorted(grouped.items())
    ]
    return DocumentsResponse(total_chunks=len(chunks), documents=infos)
