# CuraRAG

A clinical RAG system that answers only from verified drug labels and guidelines, cites the exact source passage for every claim, verifies those citations, and abstains when the evidence isn't there.

CuraRAG is built for the failure mode that matters in drug information: a confident, wrong dose or contraindication. It grounds every answer in retrieved evidence, attaches inline citations to the exact passages, uses an LLM-as-judge to check that each cited passage actually supports its claim, scores its own confidence, and refuses to answer when retrieval is weak. Hallucination is treated as a failure, not a rough edge.

## What it does

1. Ingests official public sources (openFDA drug labels, curated guideline monographs, MedlinePlus) into a normalized document model that preserves source, title, section heading, and URL on every chunk.
2. Indexes chunks into a Qdrant vector store and a BM25 sparse index over the identical chunk set.
3. Retrieves with a hybrid pipeline: dense + sparse → Reciprocal Rank Fusion → cross-encoder rerank → top-k.
4. Generates a grounded answer that may only use the numbered context blocks and must cite each claim as `[1]`, `[2]`.
5. Verifies each citation with an LLM judge and flags any that its passage does not support.
6. Scores composite confidence and abstains below a configurable retrieval threshold, returning what was searched instead of guessing.

## Architecture

```
src/curarag/
  config.py            pydantic-settings: secrets + all tunables
  models.py            Pydantic boundary types: Document, Chunk, RetrievedHit, Citation, Confidence, Answer
  embeddings.py        all-MiniLM-L6-v2 (384-dim) via langchain-huggingface
  ingestion/
    loaders.py         openFDA, MedlinePlus, PDF, and guideline-text loaders -> normalized Documents
    chunking.py        three switchable strategies (fixed | recursive | semantic)
    dedupe.py          near-duplicate drop (cosine > 0.95)
    pipeline.py        load -> chunk -> embed -> dedupe -> upsert
  retrieval/
    dense.py           Qdrant upsert + query
    sparse.py          BM25Okapi index + search
    fusion.py          Reciprocal Rank Fusion
    rerank.py          cross-encoder ms-marco-MiniLM-L-6-v2
    retriever.py       hybrid orchestration + retrieval-strength scoring
  generation/
    llm.py             provider-swappable client (deepseek | openai | anthropic), key from config
    prompts.py         grounded, citation-instructed system prompt + abstain rule
    answerer.py        retrieve -> abstain gate -> generate -> parse citations -> verify -> confidence
  verification/
    citations.py       parse markers, map to chunks, LLM-as-judge support check
    confidence.py      composite score
  api/                 FastAPI app, routes, schemas
  cli.py               seed / ingest / ask / documents
eval/                  golden set, metrics, runner, chunking comparison
```

### Key design decisions

- **Single source of truth for both indexes.** The sparse BM25 index is rebuilt in memory from the chunks stored in Qdrant on startup, so dense and sparse always cover the exact same chunk set. No drift.
- **Abstain gate before generation.** Retrieval strength is derived from the top cross-encoder logit (the sharpest on-topic signal). Below `RETRIEVAL_CONFIDENCE_THRESHOLD` (default 0.35, tuned on the golden set) the system never calls the LLM — it cannot hallucinate a dose it never generated.
- **Citation verification is independent of generation.** A separate judge call checks each claim against its cited passage. A marker is only marked supported if every claim citing it is supported.
- **Provider swappable, no hardcoded keys.** The generation layer reads the active provider's key from `pydantic-settings`. `.env` is gitignored; `.env.example` ships blank.
- **Deterministic, network-free eval.** The eval suite seeds an isolated collection from the checked-in guideline monographs only, so scores are reproducible and never touch a live index.

## Setup

Requires Docker. LLM answers need an API key for one provider; ingestion, retrieval, and the deterministic eval metrics do not.

```bash
cp .env.example .env          # set DEEPSEEK_API_KEY (or OPENAI/ANTHROPIC) for /ask
make up                       # builds images, starts Qdrant + API, seeds the demo corpus
```

`make up` runs the seed step for you. The API is then at http://localhost:8000 (docs at `/docs`).

Optional query UI:

```bash
make frontend                 # Streamlit at http://localhost:8501
```

### Local (non-Docker) development

```bash
make install                  # pip install -e .[dev,frontend]
# start Qdrant separately, e.g. docker run -p 6333:6333 qdrant/qdrant
make seed
make serve
```

## Using it

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question":"What is the maximum daily dose of acetaminophen for adults?"}' | jq
```

Response shape:

```json
{
  "answer": "The maximum recommended adult dose is 4000 mg (4 g) in 24 hours [1].",
  "abstained": false,
  "citations": [{"marker": 1, "source": "guideline", "title": "Acetaminophen (Paracetamol) Clinical Monograph", "section": "Dosage and Administration", "supported": true}],
  "confidence": {"retrieval": 0.91, "citation_coverage": 1.0, "completeness": 1.0, "composite": 0.86},
  "retrieved_chunks": [ ... ]
}
```

Endpoints: `POST /v1/ask`, `POST /v1/ingest`, `GET /v1/documents`, `GET /health`.

CLI:

```bash
curarag ask "target INR for atrial fibrillation on warfarin"
curarag ingest --drug Metformin --drug Lisinopril
curarag documents
```

## Adding a data source

1. Add a loader in `src/curarag/ingestion/loaders.py` that returns `list[Document]`, setting `source`, `title`, `url`, and `Section(heading, text)` for each section. Section headings drive structure-aware chunking and citation quality.
2. Wire it into `src/curarag/seed.py` (or call `ingest_documents(...)` directly).
3. Chunking, dedup, embedding, and indexing are handled by the pipeline — nothing else to change.

Guideline monographs use a lightweight text convention (first line title, optional `URL:` line, `## heading` sections) so a curated set stays checked in under `data/raw/guidelines/` and ingests with no network.

## Evaluation

The eval suite (`eval/`) runs a hand-written golden set of 55 cases (`eval/golden.jsonl`) covering straightforward lookups, multi-passage questions, unanswerable questions that must abstain, and ambiguous questions. Ground truth was written by hand against the corpus, not generated by an LLM.

Metrics per case:

- **answer correctness** — LLM-as-judge vs the golden answer.
- **faithfulness** — are the answer's claims grounded in the retrieved context.
- **context relevance** — recall of the expected source documents among retrieved chunks.
- **citation accuracy** — fraction of emitted citations the judge confirms are supported.
- **key fact recall** — deterministic check that golden key facts appear in the answer.
- **abstain correctness** — does the system refuse exactly the unanswerable cases.

Run it:

```bash
make eval              # single strategy -> eval/reports/eval_<strategy>.{json,md}
make eval-compare      # all three chunking strategies -> eval/reports/chunking_comparison.md
```

`make eval-compare` produces the chunking comparison table (fixed vs recursive vs semantic) as concrete evidence for which strategy retrieves and grounds best on this corpus. Both a machine-readable JSON report and a human-readable markdown summary are emitted under `eval/reports/`.

An offline sanity check over the sparse (BM25) leg alone — no models, no LLM — gives document-level recall@5 of 1.00 across the 44 answerable golden cases for both fixed (18 chunks) and recursive (36 chunks) chunking of the guideline corpus. Recall is saturated because the seed guideline corpus is small (6 monographs, 6 sections each); the discriminating signal between chunking strategies comes from the full hybrid pipeline and the faithfulness/citation-accuracy judge metrics, which `make eval-compare` reports. Run the full suite against the larger openFDA-augmented corpus for a harder benchmark.

Answer-correctness and faithfulness require an LLM key; pass `--no-llm` (or run without a key) to compute the retrieval- and citation-side metrics only. Abstain correctness, context relevance, and key-fact recall run without a judge.

## Tests

```bash
make test              # pytest: chunking, dedup, RRF, retriever wiring, citation verification, API
make lint              # ruff
```

## Non-goals

No cloud deployment, CI/CD, TLS, or autoscaling. Local Docker Compose is the finish line for this pass.
