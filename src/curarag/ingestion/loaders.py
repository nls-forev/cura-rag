from __future__ import annotations

import re
from pathlib import Path

import httpx
from pypdf import PdfReader

from curarag.config import get_settings
from curarag.models import Document, Section

# openFDA label fields we surface, mapped to human-readable section headings.
# Order matters: it is the order sections appear on the normalized document.
OPENFDA_SECTIONS: list[tuple[str, str]] = [
    ("boxed_warning", "Boxed Warning"),
    ("indications_and_usage", "Indications and Usage"),
    ("dosage_and_administration", "Dosage and Administration"),
    ("contraindications", "Contraindications"),
    ("warnings_and_cautions", "Warnings and Precautions"),
    ("warnings", "Warnings"),
    ("drug_interactions", "Drug Interactions"),
    ("adverse_reactions", "Adverse Reactions"),
    ("use_in_specific_populations", "Use in Specific Populations"),
    ("overdosage", "Overdosage"),
]


def _clean(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _brand_or_generic(result: dict) -> str:
    openfda = result.get("openfda", {})
    for key in ("brand_name", "generic_name", "substance_name"):
        vals = openfda.get(key)
        if vals:
            return vals[0]
    return result.get("id", "Unknown drug")


def load_openfda_labels(
    drug_names: list[str] | None = None,
    limit: int = 40,
) -> list[Document]:
    """Fetch structured drug label sections from openFDA.

    If drug_names is given, one label is fetched per name; otherwise a bulk
    query pulls `limit` human-use labels.
    """
    settings = get_settings()
    docs: list[Document] = []

    with httpx.Client(timeout=settings.request_timeout) as client:
        queries: list[tuple[str, dict]]
        if drug_names:
            queries = [
                (name, {"search": f'openfda.brand_name:"{name}"', "limit": 1})
                for name in drug_names
            ]
        else:
            queries = [(None, {"search": "_exists_:indications_and_usage", "limit": limit})]

        for name, params in queries:
            try:
                resp = client.get(settings.openfda_base_url, params=params)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"openFDA fetch failed for {name or 'bulk'}: {exc}")
                continue

            for result in resp.json().get("results", []):
                doc = _openfda_result_to_doc(result)
                if doc.sections:
                    docs.append(doc)

    return docs


def _openfda_result_to_doc(result: dict) -> Document:
    title = _brand_or_generic(result)
    doc_id = f"openfda:{result.get('id', title)}"
    spl = result.get("openfda", {}).get("spl_set_id", [""])
    url = (
        f"https://labels.fda.gov/getLabel.cfm?id={spl[0]}"
        if spl and spl[0]
        else "https://open.fda.gov/apis/drug/label/"
    )

    sections: list[Section] = []
    for field, heading in OPENFDA_SECTIONS:
        raw = result.get(field)
        if not raw:
            continue
        text = _clean(" ".join(raw) if isinstance(raw, list) else str(raw))
        if text:
            sections.append(Section(heading=heading, text=text))

    return Document(doc_id=doc_id, source="openFDA", title=title, url=url, sections=sections)


def load_pdf(path: str | Path, title: str | None = None, url: str = "") -> Document:
    """Load a guideline PDF into a single-section document (page text joined)."""
    path = Path(path)
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = _clean("\n".join(pages))
    resolved_title = title or path.stem.replace("_", " ")

    return Document(
        doc_id=f"pdf:{path.stem}",
        source="guideline",
        title=resolved_title,
        url=url,
        sections=[Section(heading=resolved_title, text=text)],
    )


def load_pdf_dir(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    return [load_pdf(p) for p in sorted(directory.glob("*.pdf"))]


# Guideline text files use a lightweight convention: the first line is the title,
# an optional second line beginning "URL:" is the source URL, and "## heading"
# lines start new sections. This keeps a curated guideline set checked in and
# ingestible with no network dependency.
def load_guideline_text(path: str | Path) -> Document:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    title = lines[0].strip() if lines else path.stem
    body_start = 1
    url = ""
    if len(lines) > 1 and lines[1].startswith("URL:"):
        url = lines[1][4:].strip()
        body_start = 2

    sections: list[Section] = []
    heading = title
    buffer: list[str] = []

    def flush() -> None:
        text = _clean("\n".join(buffer))
        if text:
            sections.append(Section(heading=heading, text=text))

    for line in lines[body_start:]:
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    flush()

    return Document(
        doc_id=f"guideline:{path.stem}",
        source="guideline",
        title=title,
        url=url,
        sections=sections or [Section(heading=title, text=_clean("\n".join(lines[body_start:])))],
    )


def load_guideline_dir(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    docs = [load_pdf(p) for p in sorted(directory.glob("*.pdf"))]
    for p in sorted(directory.glob("*.txt")) + sorted(directory.glob("*.md")):
        docs.append(load_guideline_text(p))
    return docs


def load_medlineplus(topics: list[str], max_per_topic: int = 2) -> list[Document]:
    """Query the MedlinePlus web service for consumer health topic summaries."""
    settings = get_settings()
    docs: list[Document] = []

    with httpx.Client(timeout=settings.request_timeout) as client:
        for topic in topics:
            params = {"db": "healthTopics", "term": topic, "retmax": max_per_topic}
            try:
                resp = client.get(settings.medlineplus_base_url, params=params)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"MedlinePlus fetch failed for {topic}: {exc}")
                continue
            docs.extend(_parse_medlineplus_xml(resp.text, topic))

    return docs


def _parse_medlineplus_xml(xml_text: str, topic: str) -> list[Document]:
    import xml.etree.ElementTree as ET

    docs: list[Document] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return docs

    for i, doc_el in enumerate(root.iter("document")):
        url = doc_el.get("url", "")
        fields = {c.get("name"): "".join(c.itertext()) for c in doc_el.iter("content")}
        title = _clean(fields.get("title", topic))
        summary = _clean(fields.get("FullSummary", fields.get("snippet", "")))
        if not summary:
            continue
        docs.append(
            Document(
                doc_id=f"medlineplus:{topic}:{i}",
                source="MedlinePlus",
                title=title,
                url=url,
                sections=[Section(heading="Health Topic", text=summary)],
            )
        )

    return docs
