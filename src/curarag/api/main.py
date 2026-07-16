from __future__ import annotations

from fastapi import FastAPI

from curarag.api.routes import router

app = FastAPI(
    title="CuraRAG",
    description=(
        "A clinical RAG system that answers only from verified drug labels and "
        "guidelines, cites the exact source passage for every claim, verifies those "
        "citations, and abstains when the evidence isn't there."
    ),
    version="0.1.0",
)
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "CuraRAG", "docs": "/docs"}
