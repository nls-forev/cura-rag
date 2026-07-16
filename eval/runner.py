from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import eval.metrics as M
from curarag.config import ChunkingStrategy, Settings, get_settings
from curarag.generation.answerer import Answerer
from curarag.generation.llm import LLMClient, LLMError
from curarag.ingestion.loaders import load_guideline_dir
from curarag.ingestion.pipeline import ingest_documents
from curarag.retrieval.dense import DenseIndex
from curarag.retrieval.retriever import HybridRetriever

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"
REPORTS_DIR = EVAL_DIR / "reports"

METRIC_KEYS = [
    "answer_correctness",
    "faithfulness",
    "context_relevance",
    "citation_accuracy",
    "key_fact_recall",
    "abstain_correct",
]


def load_golden() -> list[dict]:
    with GOLDEN_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _eval_settings(strategy: ChunkingStrategy) -> Settings:
    base = get_settings()
    # Reproducible, network-free eval: guideline corpus only, in an isolated
    # collection so a reviewer's live index is never touched.
    return base.model_copy(
        update={
            "qdrant_collection": f"{base.qdrant_collection}_eval_{strategy.value}",
            "chunking_strategy": strategy,
        }
    )


def build_answerer(strategy: ChunkingStrategy, use_llm: bool) -> Answerer:
    settings = _eval_settings(strategy)
    docs = load_guideline_dir(settings_raw_dir())
    dense = DenseIndex(settings=settings)
    ingest_documents(docs, dense=dense, strategy=strategy, settings=settings, recreate=True)
    retriever = HybridRetriever(dense=dense, settings=settings)
    llm = None
    if use_llm:
        try:
            llm = LLMClient(settings=settings)
        except LLMError:
            llm = None
    return Answerer(retriever=retriever, llm=llm, settings=settings)


def settings_raw_dir():
    from curarag.config import RAW_DIR

    return RAW_DIR / "guidelines"


def _llm_available(answerer: Answerer) -> bool:
    return answerer._llm is not None


def run_case(answerer: Answerer, case: dict, llm: LLMClient | None) -> dict:
    answer = answerer.ask(case["question"], verify=llm is not None)
    return {
        "id": case["id"],
        "type": case["type"],
        "abstained": answer.abstained,
        "confidence": answer.confidence.composite if answer.confidence else None,
        "metrics": {
            "answer_correctness": M.answer_correctness(answer, case, llm),
            "faithfulness": M.faithfulness(answer, llm),
            "context_relevance": M.context_relevance(answer, case),
            "citation_accuracy": M.citation_accuracy(answer),
            "key_fact_recall": M.key_fact_recall(answer, case),
            "abstain_correct": 1.0 if M.abstain_correct(answer, case) else 0.0,
        },
    }


def aggregate(rows: list[dict]) -> dict[str, float | None]:
    agg: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        vals = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
        agg[key] = round(sum(vals) / len(vals), 4) if vals else None
    return agg


def run_suite(strategy: ChunkingStrategy, use_llm: bool = True) -> dict:
    answerer = build_answerer(strategy, use_llm)
    llm = answerer._llm
    golden = load_golden()
    rows = [run_case(answerer, case, llm) for case in golden]
    return {
        "strategy": strategy.value,
        "llm_judge": llm is not None,
        "n_cases": len(rows),
        "aggregate": aggregate(rows),
        "cases": rows,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def to_markdown(report: dict) -> str:
    agg = report["aggregate"]
    lines = [
        f"# CuraRAG Eval Report ({report['strategy']} chunking)",
        "",
        f"- Cases: {report['n_cases']}",
        f"- LLM judge: {'on' if report['llm_judge'] else 'off (deterministic metrics only)'}",
        f"- Generated: {report['generated_at']}",
        "",
        "| Metric | Score |",
        "| --- | --- |",
    ]
    for key in METRIC_KEYS:
        lines.append(f"| {key.replace('_', ' ')} | {_fmt(agg[key])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CuraRAG golden eval suite.")
    parser.add_argument("--strategy", default=None, help="fixed | recursive | semantic")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM-judge metrics.")
    args = parser.parse_args()

    strategy = (
        ChunkingStrategy(args.strategy) if args.strategy else get_settings().chunking_strategy
    )
    report = run_suite(strategy, use_llm=not args.no_llm)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{strategy.value}"
    (REPORTS_DIR / f"{stem}.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (REPORTS_DIR / f"{stem}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
