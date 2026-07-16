from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from curarag.generation.answerer import Answerer
from curarag.ingestion.loaders import load_openfda_labels
from curarag.ingestion.pipeline import ingest_documents
from curarag.retrieval.dense import DenseIndex

app = typer.Typer(help="CuraRAG command line: seed, ingest, ask.", no_args_is_help=True)
console = Console()


@app.command()
def seed(
    strategy: str = typer.Option(None, help="Override chunking strategy for this run."),
    recreate: bool = typer.Option(True, help="Recreate the collection from scratch."),
):
    """Ingest the fixed demo corpus (drug labels + guideline PDFs)."""
    from curarag.seed import run_seed

    run_seed(strategy=strategy, recreate=recreate)


@app.command()
def ingest(
    drug: list[str] = typer.Option(..., "--drug", help="Drug brand name to fetch from openFDA."),
    recreate: bool = typer.Option(False),
):
    """Fetch and index specific drug labels by brand name."""
    docs = load_openfda_labels(drug_names=list(drug))
    report = ingest_documents(docs, recreate=recreate)
    console.print(report.model_dump())


@app.command()
def ask(question: str, no_verify: bool = typer.Option(False, "--no-verify")):
    """Ask a clinical question against the indexed corpus."""
    answer = Answerer().ask(question, verify=not no_verify)
    if answer.abstained:
        console.print(f"[yellow]ABSTAINED[/yellow]: {answer.answer}")
    else:
        console.print(answer.answer)
    if answer.confidence:
        console.print(f"[dim]confidence={answer.confidence.composite}[/dim]")
    for c in answer.citations:
        flag = "ok" if c.supported is not False else "UNSUPPORTED"
        console.print(f"  [{c.marker}] ({flag}) {c.source} — {c.title} / {c.section}")


@app.command()
def documents():
    """List indexed source documents."""
    dense = DenseIndex()
    if not dense.client.collection_exists(dense.collection):
        console.print("No collection yet. Run `curarag seed`.")
        raise typer.Exit(1)
    chunks = dense.scroll_chunks()
    table = Table("doc_id", "source", "title", "chunks")
    grouped: dict[str, list] = {}
    for c in chunks:
        grouped.setdefault(c.doc_id, []).append(c)
    for doc_id, items in sorted(grouped.items()):
        table.add_row(doc_id, items[0].source, items[0].title, str(len(items)))
    console.print(table)


if __name__ == "__main__":
    app()
