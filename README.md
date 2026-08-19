# PubMed Semantic Search — Streamlit demo

A working implementation of the semantic-search-over-PubMed problem statement:
take a free-text query, translate it into a proper NCBI search, retrieve candidates
via the **ESearch API**, and re-rank them by semantic relevance instead of raw
keyword match — with results shown in a frontend.

## What this build actually does (honest scope)

The full 5-stage architecture discussed (bi-encoder + deterministic parser →
hybrid BM25/HNSW retrieval → edge micro-embedding with 4-bit ONNX → ColBERT-style
late-interaction re-ranking with SIMD) is a **production-grade, GPU-free search
infrastructure**. This build is a faithful, runnable **simplification** of that
pipeline, mapped stage-for-stage:

| Blueprint stage | This build |
|---|---|
| 1. Intent mapping & translation | `deterministic_parse()` — rule-based parser that detects study type (clinical trial, RCT, review, meta-analysis...) and age group (elderly, pediatric, adult...) from free text and converts them into NCBI field tags (`[pt]`, `[mesh]`), leaving the remaining terms as a `[tiab]` topic clause |
| 2. First-pass hybrid retrieval | `esearch()` — calls NCBI's real **ESearch API**, which is itself backed by an inverted-index / BM25-style relevance engine (the "sparse" side of the blueprint) |
| 3. Real-time embedding of candidates | `efetch_records()` fetches title/abstract/journal/authors for each candidate PMID; `embed_texts()` embeds each abstract on the fly |
| 4. Re-ranking | `rerank()` — cosine similarity between the query vector and each candidate's vector. **Note:** this is a single dense-vector comparison, not true token-level late interaction (ColBERT/MaxSim). It captures the same *intent* (semantic re-rank on top of lexical candidates) without the production-scale infrastructure (HNSW index, 4-bit ONNX background workers, SIMD kernels) |
| 5. Frontend display | Streamlit UI renders ranked result cards with title, journal, authors, semantic score, and a link to the PubMed record |

The re-ranker automatically upgrades itself: if `sentence-transformers` is
installed, it uses real dense embeddings (`all-MiniLM-L6-v2`); if not, it falls
back to TF-IDF cosine similarity so the app still runs end-to-end with zero
extra downloads.

## Setup

```bash
cd pubmed_semantic_search
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# optional, for true semantic embeddings instead of the TF-IDF fallback:
pip install sentence-transformers
```

## Run

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`).

## Usage

1. Type a free-text query, e.g. `find clinical trials for lung cancer in elderly patients`.
2. Click **Search**.
3. Expand **"Stage 1 — Parsed query sent to NCBI ESearch"** to see exactly what
   was sent to PubMed (e.g. `Clinical Trial[pt] AND Aged[mesh] AND (lung AND
   cancer AND patients)[tiab]`).
4. Results are fetched from PubMed, then re-ranked by semantic similarity to
   your original free-text query (not just the parsed Boolean string) and
   displayed as cards with a semantic score, abstract, and a direct PubMed link.

## NCBI API usage compliance (built in)

Per NCBI's [API Usage Guidelines](https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Usage_Guidelines_and_Requiremen),
every request now:

- Identifies itself with a `tool` parameter (`pubmed-semantic-search-demo`).
- Sends your `email` if you provide one in the sidebar (recommended, not required).
- Sends an `api_key` if you provide one, which raises the rate limit from
  **3 requests/sec to 10 requests/sec**.
- Is automatically throttled client-side (`_throttle()`) to stay under whichever
  limit applies, so the app won't get rate-limited or blocked even with default
  settings.

Get a free API key at https://www.ncbi.nlm.nih.gov/account/settings/ (under
"API Key Management") if you plan to run frequent searches.

## Notes / next steps for a production version

- Swap the TF-IDF/single-vector re-ranker for a true late-interaction model
  (e.g. ColBERT) if top-10 precision needs to match the ~25%→35% improvement
  target from the original design.
- Cache embeddings for previously seen PMIDs (in-memory dict or a local vector
  store) instead of re-embedding on every search — this is the "vector cache"
  idea from the blueprint.
- Consider persisting the throttle state across Streamlit reruns/sessions
  (currently module-level, which is fine for a single-user local demo but not
  for multi-user deployments — use a shared cache like Redis for that case).
