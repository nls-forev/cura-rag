from __future__ import annotations

from curarag.config import RAW_DIR, ChunkingStrategy, get_settings
from curarag.ingestion.loaders import load_guideline_dir, load_openfda_labels
from curarag.ingestion.pipeline import ingest_documents
from curarag.models import Document

# A fixed roster of common drugs so a reviewer gets a broad, queryable corpus
# from one command. openFDA is best-effort; the checked-in guideline monographs
# guarantee a usable corpus even with no network.
DEMO_DRUGS = [
    "Tylenol", "Advil", "Aleve", "Lipitor", "Zocor", "Metformin", "Glucophage",
    "Lisinopril", "Losartan", "Amlodipine", "Amoxil", "Augmentin", "Zithromax",
    "Ciprofloxacin", "Warfarin", "Coumadin", "Eliquis", "Xarelto", "Plavix",
    "Prilosec", "Nexium", "Zoloft", "Lexapro", "Prozac", "Gabapentin",
    "Lyrica", "Synthroid", "Prednisone", "Albuterol", "Ventolin", "Singulair",
    "Januvia", "Ozempic", "Humira", "Norvasc", "Cozaar", "Crestor",
    "Flexeril", "Tramadol", "Naproxen",
]


def load_seed_documents(include_openfda: bool = True) -> list[Document]:
    docs = load_guideline_dir(RAW_DIR / "guidelines")
    if include_openfda:
        fda = load_openfda_labels(drug_names=DEMO_DRUGS)
        docs.extend(fda)
    return docs


def run_seed(
    strategy: str | None = None,
    recreate: bool = True,
    include_openfda: bool = True,
) -> None:
    settings = get_settings()
    strat = ChunkingStrategy(strategy) if strategy else settings.chunking_strategy
    docs = load_seed_documents(include_openfda=include_openfda)
    print(f"Loaded {len(docs)} source documents.")
    report = ingest_documents(docs, strategy=strat, recreate=recreate, settings=settings)
    print(
        f"Indexed {report.chunks_indexed} chunks "
        f"({report.chunks_before_dedupe} before dedupe) "
        f"using '{report.strategy}' chunking from {report.documents} documents."
    )
