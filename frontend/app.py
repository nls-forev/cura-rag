from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("CURARAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="CuraRAG", page_icon=None, layout="centered")
st.title("CuraRAG")
st.caption(
    "Answers only from verified drug labels and guidelines. Cites every claim, "
    "verifies the citations, and abstains when the evidence isn't there."
)

with st.sidebar:
    st.subheader("Indexed sources")
    try:
        docs = httpx.get(f"{API_URL}/v1/documents", timeout=30).json()
        st.metric("Indexed chunks", docs.get("total_chunks", 0))
        for d in docs.get("documents", []):
            st.write(f"- {d['title']} ({d['source']}, {d['chunks']} chunks)")
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach API: {exc}")

question = st.text_input("Clinical question", placeholder="What is the max daily dose of acetaminophen?")

if st.button("Ask") and question:
    with st.spinner("Retrieving and verifying..."):
        try:
            resp = httpx.post(f"{API_URL}/v1/ask", json={"question": question}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    if data["abstained"]:
        st.warning(data["answer"])
    else:
        st.markdown(data["answer"])

    conf = data.get("confidence")
    if conf:
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{conf['composite']:.2f}")
        c2.metric("Retrieval", f"{conf['retrieval']:.2f}")
        c3.metric("Citations verified", f"{conf['citation_coverage']:.2f}")

    if data.get("citations"):
        st.subheader("Citations")
        for c in data["citations"]:
            flag = "unsupported" if c["supported"] is False else "verified"
            st.markdown(
                f"**[{c['marker']}]** _{flag}_ — {c['source']} / {c['title']} / "
                f"{c.get('section') or 'n/a'}"
            )
            if c.get("quote"):
                st.caption(c["quote"])
