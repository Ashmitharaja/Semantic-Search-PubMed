"""
app.py — Bio-Lens with Complete PubMed Feature Set
===================================================
Includes Custom Filters, Sort Options, Display Options, and Full Page Navigation.
"""

import math
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

st.set_page_config(page_title="Bio-Lens", page_icon="🔬", layout="wide")

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
    max-width: 1080px;
    padding-top: 2rem;
}

.hero {
    text-align: center;
    padding: 1rem 0 1.5rem 0;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.2rem 0;
}
.hero .tagline {
    font-size: 1.1rem;
    color: #669df6;
    font-weight: 500;
}

.result-card {
    background-color: #131313;
    border: 1px solid #262626;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.result-card a.title {
    color: #8ab4f8;
    font-size: 1.15rem;
    font-weight: 600;
    text-decoration: none;
}
.result-card a.title:hover {
    text-decoration: underline;
}
.result-meta {
    color: #9aa0a6;
    font-size: 0.85rem;
    margin: 0.35rem 0 0.6rem 0;
}
.badge {
    background-color: #202124;
    color: #bdc1c6;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    margin-right: 6px;
    border: 1px solid #3c4043;
}
.result-abstract {
    color: #bdc1c6;
    font-size: 0.93rem;
    line-height: 1.5;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar Configuration (Custom Filters & Controls)
with st.sidebar:
    st.title("🔬 Bio-Lens Filters")
    
    st.subheader("Retrieval Controls")
    retmax = st.slider("Documents Retrieved", 10, 100, 30, step=10)
    
    st.subheader("Custom Filters")
    year_range = st.slider("Publication Year", 2000, 2026, (2015, 2026))
    
    pub_type_filter = st.multiselect(
        "Publication Types",
        ["Randomized Controlled Trial", "Clinical Trial", "Meta-Analysis", "Systematic Review", "Review", "Journal Article"],
        default=[]
    )
    
    species_filter = st.selectbox("Species / Subjects", ["All", "Humans Only", "Animals Only"])

    st.subheader("Display & Sort Options")
    sort_by = st.selectbox("Sort Results By", ["Bio-Lens AI Rank (Semantic)", "Publication Date (Newest First)"])
    display_mode = st.radio("Display View", ["Summary View", "Full Abstract", "PubMed Inspection Mode"])
    items_per_page = st.selectbox("Results Per Page", [5, 10, 20], index=1)

st.markdown(
    """
    <div class="hero">
      <h1>Bio-Lens</h1>
      <div class="tagline">Search Beyond Keywords — AI-Powered Semantic Search for PubMed</div>
    </div>
    """,
    unsafe_allow_html=True,
)

encoder, _ = get_encoder()
cache = get_cache()

query = st.text_input("Search", placeholder="Search PubMed by clinical meaning, not just keywords...", label_visibility="collapsed")
search_clicked = st.button("Search Bio-Lens", use_container_width=True)

if "page" not in st.session_state:
    st.session_state.page = 1

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

    fused_scores = fuse(bm25_scores, dense_scores, sparse_weight=0.5)
    candidates = sorted(records, key=lambda r: fused_scores.get(r["pmid"], 0.0), reverse=True)

    return rerank_late_interaction(query_token_vecs, candidates, token_map)

if (search_clicked or "last_results" in st.session_state) and query.strip():
    if search_clicked:
        st.session_state.page = 1
        with st.spinner("Executing Semantic Search & AI Re-ranking..."):
            st.session_state.last_results = _run_search(query)

    results = st.session_state.last_results

    # Apply Client-side Custom Filters
    filtered = []
    for r in results:
        try:
            r_year = int(r.get("year", 0))
        except ValueError:
            r_year = 2026
        
        if not (year_range[0] <= r_year <= year_range[1]):
            continue
            
        if pub_type_filter:
            if not any(pt in r.get("publication_types", []) for pt in pub_type_filter):
                continue

        filtered.append(r)

    # Sorting Logic
    if sort_by == "Publication Date (Newest First)":
        filtered.sort(key=lambda x: str(x.get("year", "0")), reverse=True)

    if not filtered:
        st.warning("No records matched your specific filters. Try broadening your criteria.")
    else:
        # Pagination Math
        total_items = len(filtered)
        total_pages = math.ceil(total_items / items_per_page)
        start_idx = (st.session_state.page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_items = filtered[start_idx:end_idx]

        st.caption(f"Showing {start_idx + 1}-{min(end_idx, total_items)} of {total_items} results (Page {st.session_state.page} of {total_pages})")

        # Display Results
        for rec in page_items:
            meta_line = f"{rec.get('journal', 'N/A')} · {rec.get('year', 'N/A')} · PMID: {rec['pmid']}"
            badges = "".join([f'<span class="badge">{pt}</span>' for pt in rec.get("publication_types", [])[:3]])

            st.markdown(
                f"""
                <div class="result-card">
                    <a class="title" href="{rec['url']}" target="_blank">{rec['title']}</a>
                    <div class="result-meta">{meta_line} {badges}</div>
                    <div class="result-meta"><b>Authors:</b> {rec.get('authors')}</div>
                """,
                unsafe_allow_html=True
            )

            if display_mode == "Summary View":
                abstract = rec.get("abstract", "")
                preview = (abstract[:280] + "...") if len(abstract) > 280 else abstract
                st.markdown(f'<p class="result-abstract">{preview}</p></div>', unsafe_allow_html=True)

            elif display_mode == "Full Abstract":
                st.markdown(f'<p class="result-abstract">{rec.get("abstract")}</p></div>', unsafe_allow_html=True)

            elif display_mode == "PubMed Inspection Mode":
                st.markdown(f'<p class="result-abstract">{rec.get("abstract")}</p>', unsafe_allow_html=True)
                
                with st.expander("📋 Detailed PubMed Metadata & Relations"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**MeSH Terms:**", ", ".join(rec.get("mesh_terms", [])))
                        st.write("**Grants & Funding:**", ", ".join(rec.get("grants", [])))
                    with col2:
                        st.write("**Conflict of Interest:**", rec.get("coi"))
                        st.write("**Related Resources:**", f"[Similar Articles on PubMed](https://pubmed.ncbi.nlm.nih.gov/?linkname=pubmed_pubmed&from_uid={rec['pmid']})")
                st.markdown("</div>", unsafe_allow_html=True)

        # Pagination UI Controls
        col_prev, col_center, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.session_state.page > 1:
                if st.button("← Previous"):
                    st.session_state.page -= 1
                    st.rerun()
        with col_center:
            st.markdown(f"<div style='text-align:center;'>Page {st.session_state.page} / {total_pages}</div>", unsafe_allow_html=True)
        with col_next:
            if st.session_state.page < total_pages:
                if st.button("Next →"):
                    st.session_state.page += 1
                    st.rerun()