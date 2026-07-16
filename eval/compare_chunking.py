from __future__ import annotations

import argparse
import json

from curarag.config import ChunkingStrategy
from eval.runner import METRIC_KEYS, REPORTS_DIR, _fmt, run_suite

STRATEGIES = [ChunkingStrategy.fixed, ChunkingStrategy.recursive, ChunkingStrategy.semantic]


def comparison_markdown(reports: list[dict]) -> str:
    header = "| Metric | " + " | ".join(r["strategy"] for r in reports) + " |"
    sep = "| --- | " + " | ".join("---" for _ in reports) + " |"
    lines = ["# Chunking Strategy Comparison", "", header, sep]
    for key in METRIC_KEYS:
        cells = " | ".join(_fmt(r["aggregate"][key]) for r in reports)
        lines.append(f"| {key.replace('_', ' ')} | {cells} |")
    lines += ["", f"Judge: {'on' if reports[0]['llm_judge'] else 'off'} | "
              f"Cases: {reports[0]['n_cases']}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunking strategies on the golden suite.")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    reports = []
    for strategy in STRATEGIES:
        print(f"Running suite for {strategy.value} chunking...")
        reports.append(run_suite(strategy, use_llm=not args.no_llm))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "chunking_comparison.json").write_text(json.dumps(reports, indent=2))
    md = comparison_markdown(reports)
    (REPORTS_DIR / "chunking_comparison.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
