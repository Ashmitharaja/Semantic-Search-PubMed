"""
app.py — Bio-Lens with "Search Without Filters" Action
=====================================================
"""

import math
import streamlit as st

import config
from ncbi_client import (
    deterministic_parse, 
    build_pubmed_filter_query, 
    esearch, 
    efetch_records
)
from encoder import load_encoder
from vector_cache import VectorCache
from background_worker import EmbeddingWorkerPool
from lexical_index import BM25Index
from dense_index import DenseIndex
from hybrid import fuse
from reranker import rerank_late_interaction

# 1. Page Configuration
st.set_page_config(
    page_title="Bio-Lens", 
    page_icon="🔬", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner=False)
def get_encoder():
    return load_encoder(prefer_real=True, quantize=True)

@st.cache_resource(show_spinner=False)
def get_cache():
    return VectorCache(path=".vector_cache.pkl")

# 2. Custom CSS
CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

[data-testid="stSidebar"] {
    display: block !important;
    min-width: 310px !important;
    max-width: 360px !important;
}
.block-container {
    max-width: 1100px;
    padding-top: 1rem;
}
.hero {
    text-align: center;
    padding: 0.5rem 0 1.2rem 0;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.2rem 0;
}
.hero .tagline {
    font-size: 1.05rem;
    color: #669df6;
    font-weight: 500;
}
.result-card {
    background-color: #131313;
    border: 1px solid #262626;
    border-radius: 8px;
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

# 3. Sidebar UI Controls
with st.sidebar:
    st.title("🔬 Filters & Controls")
    
    apply_filters_clicked = st.button("⚡ Apply Filters & Re-search", type="primary", use_container_width=True)
    st.markdown("---")
    
    with st.expander("⚙️ DISPLAY & SORT OPTIONS", expanded=True):
        sort_by = st.selectbox(
            "Sort by", 
            ["Best match (Bio-Lens AI Rank)", "Most recent", "Publication date", "First author", "Journal"]
        )
        display_format = st.selectbox(
            "Format", 
            ["Summary", "Abstract", "PubMed (Inspection Mode)", "PMID"]
        )
        items_per_page = st.selectbox("Per page", [10, 20, 50, 100], index=0)

    st.markdown("### 🔍 PubMed Filters")
    
    text_availability = st.multiselect("TEXT AVAILABILITY", ["Abstract", "Free full text", "Full text"], default=[])
    has_associated_data = st.checkbox("Associated data")

    st.markdown("**PUBLICATION DATE**")
    date_preset = st.radio("Date Presets", ["Any time", "1 year", "5 years", "10 years", "Custom Range"], index=0, label_visibility="collapsed")
    custom_year_range = (1990, 2026)
    if date_preset == "Custom Range":
        custom_year_range = st.slider("Select Year Range", 1990, 2026, (2015, 2026))

    article_types_list = [
        "Adaptive Clinical Trial", "Address", "Biography", "Books and Documents", 
        "Case Reports", "Clinical Study", "Clinical Trial", "Clinical Trial Protocol", 
        "Clinical Trial, Phase I", "Clinical Trial, Phase II", "Clinical Trial, Phase III", 
        "Clinical Trial, Phase IV", "Clinical Trial, Veterinary", "Collected Work", 
        "Comment", "Comparative Study", "Conference Proceedings", "Consensus Statement", 
        "Controlled Clinical Trial", "Corrected and Republished Article", "Dataset", 
        "Duplicate Publication", "Editorial", "Electronic Supplementary Materials", 
        "English Abstract", "Equivalence Trial", "Evaluation Study", "Evidence Synthesis", 
        "Expression of Concern", "Festschrift", "Guideline", "Historical Article", 
        "Interview", "Introductory Journal Article", "Lecture", "Letter", "Meta-Analysis", 
        "Multicenter Study", "Network Meta-Analysis", "News", "Observational Study", 
        "Observational Study, Veterinary", "Patient Education Handout", "Personal Narrative", 
        "Practice Guideline", "Pragmatic Clinical Trial", "Preprint", "Published Erratum", 
        "Randomized Controlled Trial", "Randomized Controlled Trial, Veterinary", 
        "Research Support, American Recovery and Reinvestment Act", "Research Support, N.I.H., Extramural", 
        "Research Support, N.I.H., Intramural", "Research Support, Non-U.S. Gov't", 
        "Research Support, U.S. Gov't, Non-P.H.S.", "Research Support, U.S. Gov't, P.H.S.", 
        "Research Support, U.S. Gov't", "Retracted Publication", "Retraction Notice", 
        "Review", "Scoping Review", "Systematic Review", "Twin Study", "Validation Study", 
        "Video-Audio Media", "Webcast"
    ]
    selected_article_types = st.multiselect("ARTICLE TYPE", article_types_list, default=[])

    language_list = ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Italian", "Russian"]
    selected_languages = st.multiselect("ARTICLE LANGUAGE", language_list, default=[])

    species_selection = st.multiselect("SPECIES", ["Humans", "Other Animals"], default=[])
    sex_selection = st.multiselect("SEX", ["Female", "Male"], default=[])

    age_groups_list = [
        "Child: birth-18 years", "Newborn: birth-1 month", "Infant: birth-23 months", 
        "Infant: 1-23 months", "Preschool Child: 2-5 years", "Child: 6-12 years", 
        "Adolescent: 13-18 years", "Adult: 19+ years", "Young Adult: 19-24 years", 
        "Adult: 19-44 years", "Middle Aged + Aged: 45+ years", "Middle Aged: 45-64 years", 
        "Aged: 65+ years", "80 and over: 80+ years"
    ]
    selected_ages = st.multiselect("AGE", age_groups_list, default=[])

    exclude_preprints = st.checkbox("Exclude preprints")
    medline_only = st.checkbox("MEDLINE")

    st.markdown("---")
    st.markdown("### 🤖 Bio-Lens AI Options")
    retmax = st.slider("Candidate Fetch Size (NCBI)", 10, 100, 30, step=10)
    sparse_weight = st.slider("BM25 vs Dense Weight", 0.0, 1.0, 0.5, step=0.1)

# 4. Main Interface Header & Action Buttons
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

col_search, col_raw = st.columns([1, 1])
with col_search:
    search_with_filters = st.button("🔍 Search with Filters", use_container_width=True, type="primary")
with col_raw:
    search_without_filters = st.button("🌐 Search Without Filters", use_container_width=True)

if "page" not in st.session_state:
    st.session_state.page = 1

def _run_search(query_str: str, ignore_filters: bool = False):
    base_q, _, _ = deterministic_parse(query_str)
    
    if ignore_filters:
        full_pubmed_query = base_q
    else:
        full_pubmed_query = build_pubmed_filter_query(
            base_q, 
            selected_ages, 
            species_selection, 
            sex_selection, 
            selected_languages,
            text_availability
        )

    query_pooled, query_token_vecs = encoder.encode(query_str)

    pmids = esearch(
        full_pubmed_query, retmax=retmax, sort="relevance",
        email=config.NCBI_EMAIL, api_key=config.NCBI_API_KEY,
    )
    if not pmids:
        return []
        
    records = efetch_records(pmids, email=config.NCBI_EMAIL, api_key=config.NCBI_API_KEY)
    if not records:
        return []

    bm25_scores = BM25Index(records).score(query_str)

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

    fused_scores = fuse(bm25_scores, dense_scores, sparse_weight=sparse_weight)
    candidates = sorted(records, key=lambda r: fused_scores.get(r["pmid"], 0.0), reverse=True)

    return rerank_late_interaction(query_token_vecs, candidates, token_map)

# 5. Search Execution Router
must_execute_search = (search_with_filters or apply_filters_clicked or search_without_filters)

if (must_execute_search or "last_results" in st.session_state) and query.strip():
    if must_execute_search:
        st.session_state.page = 1
        ignore = True if search_without_filters else False
        st.session_state.is_unfiltered_run = ignore
        
        status_msg = "Running Unfiltered PubMed Search..." if ignore else "Applying Filters & Running AI Search..."
        with st.spinner(status_msg):
            st.session_state.last_results = _run_search(query, ignore_filters=ignore)

    results = st.session_state.last_results
    is_unfiltered = st.session_state.get("is_unfiltered_run", False)

    filtered = []
    current_year = 2026
    
    for r in results:
        if is_unfiltered:
            filtered.append(r)
            continue

        try:
            r_year = int(r.get("year", 0))
        except ValueError:
            r_year = current_year
        
        # Post-retrieval Date Filter
        if date_preset == "1 year" and r_year < (current_year - 1):
            continue
        elif date_preset == "5 years" and r_year < (current_year - 5):
            continue
        elif date_preset == "10 years" and r_year < (current_year - 10):
            continue
        elif date_preset == "Custom Range":
            if not (custom_year_range[0] <= r_year <= custom_year_range[1]):
                continue

        # Post-retrieval Article Types Filter
        if selected_article_types:
            if not any(pt in r.get("publication_types", []) for pt in selected_article_types):
                continue

        # Post-retrieval Exclude Preprints Filter
        if exclude_preprints and "Preprint" in r.get("publication_types", []):
            continue

        filtered.append(r)

    # Sorting
    if sort_by == "Most recent" or sort_by == "Publication date":
        filtered.sort(key=lambda x: str(x.get("year", "0")), reverse=True)
    elif sort_by == "First author":
        filtered.sort(key=lambda x: str(x.get("authors", "")).lower())
    elif sort_by == "Journal":
        filtered.sort(key=lambda x: str(x.get("journal", "")).lower())

    # Rendering Results
    if not filtered:
        st.warning("No records matched your search parameters.")
    else:
        total_items = len(filtered)
        total_pages = math.ceil(total_items / items_per_page)
        start_idx = (st.session_state.page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_items = filtered[start_idx:end_idx]

        if is_unfiltered:
            st.info("ℹ️ Showing raw, unfiltered PubMed results.")

        st.caption(f"Showing {start_idx + 1}-{min(end_idx, total_items)} of {total_items} results (Page {st.session_state.page} of {total_pages})")

        for rec in page_items:
            if display_format == "PMID":
                st.write(f"PMID: {rec['pmid']}")
                continue

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

            if display_format == "Summary":
                abstract = rec.get("abstract", "")
                preview = (abstract[:280] + "...") if len(abstract) > 280 else abstract
                st.markdown(f'<p class="result-abstract">{preview}</p></div>', unsafe_allow_html=True)

            elif display_format == "Abstract":
                st.markdown(f'<p class="result-abstract">{rec.get("abstract")}</p></div>', unsafe_allow_html=True)

            elif display_format == "PubMed (Inspection Mode)":
                st.markdown(f'<p class="result-abstract">{rec.get("abstract")}</p>', unsafe_allow_html=True)
                
                with st.expander("📋 Detailed PubMed Metadata & Relations"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**MeSH Terms:**", ", ".join(rec.get("mesh_terms", [])))
                    with col2:
                        st.write("**Languages:**", ", ".join(rec.get("languages", [])))
                        st.write("**Related Resources:**", f"[Similar Articles on PubMed](https://pubmed.ncbi.nlm.nih.gov/?linkname=pubmed_pubmed&from_uid={rec['pmid']})")
                st.markdown("</div>", unsafe_allow_html=True)

        # Pagination Controls
        if display_format != "PMID" and total_pages > 1:
            col_prev, col_center, col_next = st.columns([1, 3, 1])
            with col_prev:
                if st.session_state.page > 1:
                    if st.button("← Previous"):
                        st.session_state.page -= 1
                        st.rerun()
            with col_center:
                st.markdown(f"<div style='text-align:center;'>Page {st.session_state.page} of {total_pages}</div>", unsafe_allow_html=True)
            with col_next:
                if st.session_state.page < total_pages:
                    if st.button("Next →"):
                        st.session_state.page += 1
                        st.rerun()