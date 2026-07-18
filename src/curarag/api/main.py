from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from curarag.api.routes import router


@asynccontextmanager
async def lifespan(_: FastAPI):
    from curarag.seed import ensure_seeded

    ensure_seeded()
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
