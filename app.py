"""
app.py — Bio-Lens
==================
Frontend: clean, minimal, black background, "Search Beyond Keywords."

NCBI credentials (email / API key) are NOT UI inputs anymore — they live in
config.py, read from environment variables / a local .env file. The API key
IS still genuinely used by the real network calls in ncbi_client.py; it's
just configured once on the machine running the app instead of typed into
the browser. See config.py and .env.example for how to set it.

The only user-facing control is "Documents retrieved," in the sidebar.
Everything else (results count, hybrid fusion weight) uses a sensible fixed
default so the main screen stays uncluttered.
"""

import streamlit as st

import config
from ncbi_client import deterministic_parse, esearch, efetch_records
from encoder import load_encoder
from vector_cache import VectorCache
from background_worker import EmbeddingWorkerPool
from lexical_index import BM25Index
from dense_index import DenseIndex
from hybrid import fuse
from reranker import rerank_late_interaction

st.set_page_config(page_title="Bio-Lens", page_icon="🔬", layout="centered")

# Fixed defaults — no longer exposed as UI controls.
TOP_K = 10
SPARSE_WEIGHT = 0.5


@st.cache_resource(show_spinner=False)
def get_encoder():
    return load_encoder(prefer_real=True, quantize=True)


@st.cache_resource(show_spinner=False)
def get_cache():
    return VectorCache(path=".vector_cache.pkl")


CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    max-width: 720px;
    padding-top: 3rem;
}

.hero {
    text-align: center;
    padding: 2rem 0 2.5rem 0;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.03em;
}
.hero .tagline {
    font-size: 1.15rem;
    color: #669df6;
    font-weight: 500;
    margin: 0 0 0.6rem 0;
}
.hero .subtitle {
    font-size: 0.95rem;
    color: #9aa0a6;
    font-weight: 400;
    margin: 0;
}

div[data-testid="stTextInput"] input {
    background-color: #ffffff;
    color: #202124;
    border-radius: 28px;
    border: none;
    padding: 0.95rem 1.4rem;
    font-size: 1rem;
    box-shadow: 0 1px 8px rgba(0,0,0,0.45);
}
div[data-testid="stTextInput"] input:focus {
    outline: none;
    box-shadow: 0 0 0 2px #669df6;
}
div[data-testid="stTextInput"] label {
    display: none;
}

div[data-testid="stButton"] button {
    background-color: #4285F4;
    color: #ffffff;
    border-radius: 28px;
    border: none;
    padding: 0.7rem 2.2rem;
    font-size: 1rem;
    font-weight: 500;
    width: 100%;
    transition: background-color 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stButton"] button:hover {
    background-color: #5a95f5;
    box-shadow: 0 2px 10px rgba(66,133,244,0.4);
    color: #ffffff;
}
div[data-testid="stButton"] button:active {
    background-color: #3367d6;
}

.result-card {
    background-color: #131313;
    border: 1px solid #262626;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}
.result-card a {
    color: #8ab4f8;
    font-size: 1.1rem;
    font-weight: 500;
    text-decoration: none;
    line-height: 1.4;
}
.result-card a:hover {
    text-decoration: underline;
}
.result-meta {
    color: #9aa0a6;
    font-size: 0.82rem;
    margin: 0.35rem 0 0.7rem 0;
}
.result-abstract {
    color: #bdc1c6;
    font-size: 0.93rem;
    line-height: 1.55;
    margin: 0;
}
.result-count {
    color: #9aa0a6;
    font-size: 0.85rem;
    margin: 0.5rem 0 1.2rem 2px;
}
.empty-state {
    text-align: center;
    color: #9aa0a6;
    font-size: 0.95rem;
    padding: 2rem 0;
}

section[data-testid="stSidebar"] .sidebar-title {
    color: #ffffff;
    font-size: 1rem;
    font-weight: 500;
    margin: 0.5rem 0 1.25rem 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">Search settings</div>', unsafe_allow_html=True)
    retmax = st.slider("Documents retrieved", 10, 100, 30, step=10)

st.markdown(
    """
    <div class="hero">
      <h1>Bio-Lens</h1>
      <div class="tagline">Search Beyond Keywords</div>
      <div class="subtitle">An AI-Powered Semantic Search Platform for PubMed</div>
    </div>
    """,
    unsafe_allow_html=True,
)

encoder, _is_real_encoder = get_encoder()
cache = get_cache()

_, mid, _ = st.columns([1, 10, 1])
with mid:
    query = st.text_input(
        "Search",
        placeholder="Search PubMed by meaning, not just keywords\u2026",
        label_visibility="collapsed",
    )
    search_clicked = st.button("Search")


def _run_search(query: str):
    structured_query, _, _ = deterministic_parse(query)
    query_pooled, query_token_vecs = encoder.encode(query)

    pmids = esearch(
        structured_query, retmax=retmax, sort="relevance",
        email=config.NCBI_EMAIL, api_key=config.NCBI_API_KEY,
    )
    if not pmids:
        return []
    records = efetch_records(pmids, email=config.NCBI_EMAIL, api_key=config.NCBI_API_KEY)
    if not records:
        return []

    bm25_scores = BM25Index(records).score(query)

    worker = EmbeddingWorkerPool(encoder, cache, max_workers=4)
    pooled_map, token_map, _, _, _ = worker.ensure_embedded(records)
    worker.shutdown()
    cache.save()

    dense_index = DenseIndex(dim=encoder.dim, path_prefix=".dense_index")
    for r in records:
        if r["pmid"] in pooled_map:
            dense_index.add(r["pmid"], pooled_map[r["pmid"]])
    dense_index.save()
    dense_hits = dense_index.search(query_pooled, k=len(dense_index))
    dense_scores = {pmid: s for pmid, s in dense_hits if any(r["pmid"] == pmid for r in records)}

    fused_scores = fuse(bm25_scores, dense_scores, sparse_weight=SPARSE_WEIGHT)
    candidates = sorted(records, key=lambda r: fused_scores.get(r["pmid"], 0.0), reverse=True)

    return rerank_late_interaction(query_token_vecs, candidates, token_map)


if search_clicked and query.strip():
    with st.spinner(""):
        final_ranked = _run_search(query)

    if not final_ranked:
        st.markdown('<div class="empty-state">No results found. Try rephrasing your search.</div>', unsafe_allow_html=True)
    else:
        shown = final_ranked[:TOP_K]
        st.markdown(f'<div class="result-count">{len(shown)} results</div>', unsafe_allow_html=True)
        for rec in shown:
            meta_parts = [p for p in [rec.get("journal"), rec.get("year")] if p]
            meta_line = " · ".join(meta_parts + ([rec["authors"]] if rec.get("authors") else []))
            abstract = rec.get("abstract") or ""
            preview = (abstract[:320] + "…") if len(abstract) > 320 else abstract
            st.markdown(
                f"""
                <div class="result-card">
                    <a href="{rec['url']}" target="_blank">{rec['title']}</a>
                    <div class="result-meta">{meta_line}</div>
                    <p class="result-abstract">{preview}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
elif search_clicked:
    st.markdown('<div class="empty-state">Enter a search query to begin.</div>', unsafe_allow_html=True)