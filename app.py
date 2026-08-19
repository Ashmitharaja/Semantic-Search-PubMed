"""
PubMed Semantic Search — Streamlit demo
=========================================

Implements a practical, runnable version of the 5-stage architecture:

  STAGE 1  Intent mapping & translation
           - deterministic_parse()   -> structured Boolean/field query for NCBI ESearch
           - (optional) embed_text() -> dense query vector, used later for re-ranking

  STAGE 2  First-pass hybrid retrieval (sparse)
           - esearch()  -> calls NCBI's ESearch API (Lucene/BM25-backed inverted index)
                            returns a candidate list of PMIDs, ranked by relevance/date

  STAGE 3  Fetch + real-time embedding of candidates
           - efetch_records() -> pulls title/abstract/journal/authors for each PMID
           - embed_text()     -> generates a vector for each abstract on the fly
                                  (mirrors the "background worker fills missing vectors"
                                  idea from the blueprint, just synchronous here)

  STAGE 4  Re-ranking
           - rerank()   -> cosine similarity between query vector and each candidate
                            vector (a simplified stand-in for ColBERT-style late
                            interaction / MaxSim re-ranking)

  STAGE 5  Frontend display
           - Streamlit renders ranked cards with title, journal, authors, score,
             and a link back to the PubMed record.

Honesty note: this build uses a single dense-vector cosine re-rank (SentenceTransformer
if available, otherwise a TF-IDF cosine fallback) rather than a true token-level
late-interaction / HNSW / 4-bit-ONNX production stack. It is a faithful, working
simplification of the pipeline for demo purposes, not the full production architecture.
"""

import re
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from xml.etree import ElementTree as ET

# --------------------------------------------------------------------------------------
# Optional dense embedder: SentenceTransformer if installed, otherwise TF-IDF fallback.
# --------------------------------------------------------------------------------------
EMBEDDER_NAME = "TF-IDF cosine (fallback)"
_model = None
try:
    from sentence_transformers import SentenceTransformer

    @st.cache_resource(show_spinner=False)
    def _load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    _model = _load_model()
    EMBEDDER_NAME = "SentenceTransformer (all-MiniLM-L6-v2)"
except Exception:
    _model = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL_NAME = "pubmed-semantic-search-demo"

# --------------------------------------------------------------------------------------
# NCBI E-utilities usage policy compliance
# https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Usage_Guidelines_and_Requiremen
#   - every request should identify itself via `tool` and `email` params
#   - rate limit: 3 requests/sec without an API key, 10 requests/sec with one
# --------------------------------------------------------------------------------------
_last_request_time = {"t": 0.0}


def _throttle(has_api_key: bool):
    """Sleeps just long enough to stay under NCBI's rate limit."""
    min_interval = 1.0 / 10 if has_api_key else 1.0 / 3
    elapsed = time.time() - _last_request_time["t"]
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time["t"] = time.time()


def _ncbi_params(email: str, api_key: str, extra: dict):
    params = dict(extra)
    params["tool"] = TOOL_NAME
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    return params


STUDY_TYPE_MAP = {
    "randomized controlled trials": "Randomized Controlled Trial[pt]",
    "randomized controlled trial": "Randomized Controlled Trial[pt]",
    "clinical trials": "Clinical Trial[pt]",
    "clinical trial": "Clinical Trial[pt]",
    "rct": "Randomized Controlled Trial[pt]",
    "systematic reviews": "Systematic Review[pt]",
    "systematic review": "Systematic Review[pt]",
    "meta-analyses": "Meta-Analysis[pt]",
    "meta-analysis": "Meta-Analysis[pt]",
    "case reports": "Case Reports[pt]",
    "case report": "Case Reports[pt]",
    "observational studies": "Observational Study[pt]",
    "observational study": "Observational Study[pt]",
    "reviews": "Review[pt]",
    "review": "Review[pt]",
}

AGE_GROUP_MAP = {
    "elderly": "Aged[mesh]",
    "older adult": "Aged[mesh]",
    "older adults": "Aged[mesh]",
    "geriatric": "Aged[mesh]",
    "pediatric": "Child[mesh]",
    "paediatric": "Child[mesh]",
    "children": "Child[mesh]",
    "infant": "Infant[mesh]",
    "neonate": "Infant, Newborn[mesh]",
    "adolescent": "Adolescent[mesh]",
    "adolescents": "Adolescent[mesh]",
    "adult": "Adult[mesh]",
    "adults": "Adult[mesh]",
}


# --------------------------------------------------------------------------------------
# STAGE 1 — deterministic parser: free text -> structured NCBI query
# --------------------------------------------------------------------------------------
def deterministic_parse(query: str):
    """
    Rule-based, non-AI parser. Scans the free-text query for known study-type and
    age-group phrases and converts them into NCBI field-tagged clauses. Whatever text
    remains after stripping those phrases is treated as the core disease/topic clause
    and is searched in Title/Abstract.
    """
    q_lower = query.lower()
    clauses = []
    detected = {"study_type": None, "age_group": None}

    for phrase, tag in STUDY_TYPE_MAP.items():
        if phrase in q_lower:
            clauses.append(tag)
            detected["study_type"] = phrase
            q_lower = q_lower.replace(phrase, " ")
            break  # first match wins, mirrors ATM "stop at first match" behavior

    for phrase, tag in AGE_GROUP_MAP.items():
        if phrase in q_lower:
            clauses.append(tag)
            detected["age_group"] = phrase
            q_lower = q_lower.replace(phrase, " ")
            break

    # whatever's left is the topical / disease clause
    remainder = re.sub(r"[^a-z0-9\s\-]", " ", q_lower)
    remainder = re.sub(r"\s+", " ", remainder).strip()
    stopwords = {"find", "for", "in", "the", "a", "an", "of", "with", "and", "on"}
    remainder_terms = [
        w for w in remainder.split() if w not in stopwords and len(w) > 1
    ]
    topic_clause = None
    if remainder_terms:
        topic_clause = f'({" AND ".join(remainder_terms)})[tiab]'
        clauses.append(topic_clause)

    structured_query = " AND ".join(clauses) if clauses else query
    return structured_query, detected, topic_clause


# --------------------------------------------------------------------------------------
# STAGE 2 — sparse retrieval via NCBI ESearch (BM25/Lucene-backed inverted index)
# --------------------------------------------------------------------------------------
def esearch(term: str, retmax: int = 30, sort: str = "relevance", email: str = "", api_key: str = ""):
    params = _ncbi_params(
        email,
        api_key,
        {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax, "sort": sort},
    )
    _throttle(has_api_key=bool(api_key))
    resp = requests.get(ESEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


# --------------------------------------------------------------------------------------
# STAGE 3 — fetch candidate records (title / abstract / journal / authors)
# --------------------------------------------------------------------------------------
def efetch_records(pmids: list, email: str = "", api_key: str = ""):
    if not pmids:
        return []
    params = _ncbi_params(
        email,
        api_key,
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"},
    )
    _throttle(has_api_key=bool(api_key))
    resp = requests.get(EFETCH_URL, params=params, timeout=20)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "(no title)"

        abstract_parts = article.findall(".//Abstract/AbstractText")
        abstract = " ".join("".join(p.itertext()).strip() for p in abstract_parts) if abstract_parts else ""

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""

        year_el = article.find(".//JournalIssue/PubDate/Year")
        if year_el is None:
            year_el = article.find(".//JournalIssue/PubDate/MedlineDate")
        year = year_el.text if year_el is not None else ""

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None:
                name = last.text
                if fore is not None:
                    name = f"{fore.text} {name}"
                authors.append(name)

        records.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "authors": ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return records


# --------------------------------------------------------------------------------------
# STAGE 3b / STAGE 4 — embedding + re-ranking
# --------------------------------------------------------------------------------------
def embed_texts(texts: list):
    """Returns a dense (or TF-IDF) vector matrix for a list of texts."""
    if _model is not None:
        return _model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def rerank(query: str, records: list):
    """
    Simplified tensor/dense re-rank stage.
    True implementation note: production system would use token-level late-interaction
    (ColBERT-style MaxSim) over multi-vector representations. Here we approximate it
    with a single query/document embedding cosine similarity, which is the closest
    faithful equivalent achievable without a token-level index.
    """
    docs = [f"{r['title']}. {r['abstract']}" for r in records]
    if not docs:
        return records

    if _model is not None:
        doc_vecs = embed_texts(docs)
        query_vec = embed_texts([query])
        sims = cosine_similarity(query_vec, doc_vecs)[0]
    else:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
        all_texts = docs + [query]
        tfidf = vectorizer.fit_transform(all_texts)
        doc_vecs, query_vec = tfidf[:-1], tfidf[-1]
        sims = cosine_similarity(query_vec, doc_vecs)[0]

    for rec, score in zip(records, sims):
        rec["semantic_score"] = float(score)

    return sorted(records, key=lambda r: r["semantic_score"], reverse=True)


# --------------------------------------------------------------------------------------
# STAGE 5 — Streamlit frontend
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="PubMed Semantic Search", page_icon="🧬", layout="wide")

st.title("🧬 PubMed Semantic Search")
st.caption(
    "Free-text query → deterministic Boolean parsing + NCBI ESearch (sparse retrieval) "
    "→ candidate fetch → semantic re-rank → ranked results."
)

with st.sidebar:
    st.subheader("Pipeline settings")
    retmax = st.slider("Candidates to retrieve from ESearch (Stage 2)", 10, 100, 30, step=10)
    top_k = st.slider("Results to display after re-rank (Stage 4)", 5, 30, 10)
    sort_mode = st.selectbox("ESearch sort", ["relevance", "pub+date"], index=0)
    st.markdown("---")
    st.subheader("NCBI API compliance")
    ncbi_email = st.text_input(
        "Your email (recommended by NCBI)",
        placeholder="you@example.com",
        help="NCBI asks every E-utilities request to identify a contact email. "
        "Not required to run, but recommended.",
    )
    ncbi_api_key = st.text_input(
        "NCBI API key (optional)",
        type="password",
        help="Raises the rate limit from 3 req/sec to 10 req/sec. "
        "Get one free at https://www.ncbi.nlm.nih.gov/account/settings/",
    )
    st.caption(
        f"Rate limit in effect: **{'10' if ncbi_api_key else '3'} requests/sec** "
        f"({'with' if ncbi_api_key else 'without'} API key)."
    )
    st.markdown("---")
    st.caption(f"**Re-rank engine:** {EMBEDDER_NAME}")
    if _model is None:
        st.caption(
            "Install `sentence-transformers` for true dense embeddings; "
            "currently using a TF-IDF cosine fallback."
        )

query = st.text_input(
    "Enter your query",
    placeholder="e.g. clinical trials for lung cancer in elderly patients",
)

col_a, col_b = st.columns([1, 5])
search_clicked = col_a.button("Search", type="primary")

if search_clicked and query.strip():
    t0 = time.time()

    # Stage 1
    structured_query, detected, topic_clause = deterministic_parse(query)
    with st.expander("Stage 1 — Parsed query sent to NCBI ESearch", expanded=False):
        st.code(structured_query, language="text")
        st.json(detected)

    # Stage 2
    with st.spinner("Stage 2 — Retrieving candidates from PubMed (ESearch)..."):
        try:
            pmids = esearch(
                structured_query,
                retmax=retmax,
                sort=sort_mode,
                email=ncbi_email,
                api_key=ncbi_api_key,
            )
        except Exception as e:
            st.error(f"ESearch request failed: {e}")
            pmids = []

    if not pmids:
        st.warning("No candidates found. Try a broader query or fewer detected filters.")
    else:
        # Stage 3
        with st.spinner(f"Stage 3 — Fetching {len(pmids)} candidate records..."):
            try:
                records = efetch_records(pmids, email=ncbi_email, api_key=ncbi_api_key)
            except Exception as e:
                st.error(f"EFetch request failed: {e}")
                records = []

        # Stage 4
        with st.spinner("Stage 4 — Semantic re-ranking..."):
            ranked = rerank(query, records) if records else []

        elapsed = time.time() - t0
        st.success(f"Retrieved {len(records)} candidates, re-ranked in {elapsed:.2f}s.")

        # Stage 5
        for rec in ranked[:top_k]:
            with st.container(border=True):
                title_col, score_col = st.columns([5, 1])
                title_col.markdown(f"### [{rec['title']}]({rec['url']})")
                score_col.metric("Semantic score", f"{rec.get('semantic_score', 0):.3f}")
                meta = " · ".join(filter(None, [rec["journal"], rec["year"], f"PMID: {rec['pmid']}"]))
                st.caption(meta)
                if rec["authors"]:
                    st.caption(rec["authors"])
                if rec["abstract"]:
                    with st.expander("Abstract"):
                        st.write(rec["abstract"])
                else:
                    st.caption("_No abstract available._")

elif search_clicked:
    st.warning("Please enter a query.")
