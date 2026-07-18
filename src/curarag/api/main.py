from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from curarag.api.routes import router
from curarag.config import get_settings


def _maybe_seed() -> None:
    settings = get_settings()
    if not settings.seed_on_startup:
        return
    from curarag.retrieval.dense import DenseIndex
    from curarag.seed import run_seed

    dense = DenseIndex(settings=settings)
    already = dense.client.collection_exists(dense.collection) and dense.count() > 0
    if already:
        return
    try:
        run_seed(recreate=True, include_openfda=settings.seed_include_openfda)
    except Exception as exc:  # noqa: BLE001 - a failed seed must not block boot
        print(f"Startup seed failed: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _maybe_seed()
    yield


app = FastAPI(
    title="CuraRAG",
    description=(
        "A clinical RAG system that answers only from verified drug labels and "
        "guidelines, cites the exact source passage for every claim, verifies those "
        "citations, and abstains when the evidence isn't there."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "CuraRAG", "docs": "/docs"}
