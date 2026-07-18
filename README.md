---
title: CuraRAG
emoji: 💊
colorFrom: red
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

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

The eval suite (`eval/`) runs a hand-written golden set of 56 cases (`eval/golden.jsonl`) covering straightforward lookups, multi-passage questions, unanswerable questions that must abstain, and ambiguous questions. Ground truth was written by hand against the corpus, not generated by an LLM.

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

### Results

Full suite, 56 cases, DeepSeek (`deepseek-chat`) as generator and judge, guideline corpus:

| Metric | fixed | recursive | semantic |
| --- | --- | --- | --- |
| answer correctness | 0.936 | 0.993 | 1.000 |
| faithfulness | 1.000 | 0.932 | 0.955 |
| context relevance | 1.000 | 1.000 | 1.000 |
| citation accuracy | 0.861 | 0.881 | 0.909 |
| key fact recall | 0.977 | 1.000 | 1.000 |
| abstain correct | 0.964 | 1.000 | 1.000 |

Takeaways:

- **Recursive and semantic chunking refuse every unanswerable case (12/12)** and recall every golden key fact. Fixed chunking missed two cases (`met-b12`, `multi-inflam-choice`) where size-blind boundaries split the relevant facts, dropping retrieval strength below the abstain threshold.
- Semantic chunking edges out the others on correctness and citation accuracy; recursive (section-aware) is within noise of it and is an order of magnitude cheaper at ingest (no per-sentence embedding), which is why it is the default.
- Fixed chunking scores highest on faithfulness only because it abstained on cases the others answered — fewer answers, fewer claims to ground. Faithfulness must be read alongside abstain/correctness, not alone.
- Context relevance saturates at 1.00 on the 6-monograph guideline corpus; an offline BM25-only sanity check shows the same (doc-recall@5 = 1.00 over the 44 answerable cases). The corpus is small by design so the eval is deterministic and network-free; re-run against the openFDA-augmented index for a harder retrieval benchmark.

Answer-correctness and faithfulness require an LLM key; pass `--no-llm` (or run without a key) to compute the retrieval- and citation-side metrics only. Abstain correctness, context relevance, and key-fact recall run without a judge.

## Tests

```bash
make test              # pytest: chunking, dedup, RRF, retriever wiring, citation verification, API
make lint              # ruff
```

## Deployment (single container)

CuraRAG can run the whole system in one process: Qdrant **embedded** (in-process, on disk) instead of a separate service, seeded on first boot. This is driven entirely by env vars — no code change:

| Env var | Purpose |
| --- | --- |
| `QDRANT_PATH` | Path for embedded Qdrant. If set, the Qdrant server URL is ignored. |
| `SEED_ON_STARTUP` | `true` → seed the corpus on boot if the collection is empty. |
| `SEED_INCLUDE_OPENFDA` | `false` keeps boot fast (guideline monographs only). |
| `CURARAG_DATA_DIR` | Corpus location, for when the package is installed away from the repo. |
| `DEEPSEEK_API_KEY` | Supplied as a platform **secret**, never baked into an image. |

### Hugging Face Space (free, Gradio SDK)

The repo is a ready Gradio Space: [`app.py`](app.py) sets the single-container defaults, seeds the guideline corpus, and serves a UI over the same `Answerer` the API uses. Hugging Face installs from [`requirements.txt`](requirements.txt) (it does not read `pyproject.toml`, and it installs deps before copying the repo, so `pip install .` is not an option) — the runtime deps are listed there explicitly, minus `torch` and `gradio`, which the Space base image pins.

```bash
# create a Gradio Space at huggingface.co/new-space (SDK: Gradio), then:
git remote add hf https://huggingface.co/spaces/<user>/curarag
git push hf master        # use a Hugging Face access token as the password
# in the Space settings, add DEEPSEEK_API_KEY as a secret
```

The free CPU tier (16 GB) fits the embedding + reranker models. First cold start downloads the models and seeds the corpus (~2 minutes); the Space then sleeps after inactivity. The `Dockerfile` (bakes the models, no download at boot) remains for local or Docker-based hosts.

## Non-goals

No CI/CD, TLS, or autoscaling. The single-container Space above is a demo deploy, not a production posture (no persistent index, no horizontal scaling).
