from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from curarag.api import routes
from curarag.api.main import app
from curarag.models import Answer, Citation, Confidence


class FakeDense:
    collection = "curarag_test"

    class _Client:
        def collection_exists(self, name):
            return False

    client = _Client()

    def count(self):
        return 0

    def scroll_chunks(self):
        return []


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(routes, "_dense", lambda: FakeDense())
    return TestClient(app)


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "CuraRAG"


def test_health_degrades_gracefully(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["indexed_chunks"] == 0
    assert body["collection"] == "curarag_test"


def test_documents_empty(client):
    resp = client.get("/v1/documents")
    assert resp.status_code == 200
    assert resp.json() == {"total_chunks": 0, "documents": []}


def test_ask_returns_answer_shape(client, monkeypatch):
    fake_answer = Answer(
        question="q",
        answer="Max dose is 4 g [1].",
        abstained=False,
        citations=[Citation(marker=1, chunk_id="c1", source="guideline", title="Acetaminophen")],
        confidence=Confidence(retrieval=0.8, citation_coverage=1.0, completeness=1.0, composite=0.9),
    )

    class FakeAnswerer:
        def __init__(self, *a, **k):
            pass

        def ask(self, question, verify=True):
            return fake_answer

    monkeypatch.setattr(routes, "Answerer", FakeAnswerer)
    resp = client.post("/v1/ask", json={"question": "acetaminophen max dose?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is False
    assert body["citations"][0]["marker"] == 1
    assert body["confidence"]["composite"] == 0.9


def test_ask_rejects_empty_question(client):
    resp = client.post("/v1/ask", json={"question": ""})
    assert resp.status_code == 422
