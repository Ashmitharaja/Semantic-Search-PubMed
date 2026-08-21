"""
eval_harness.py
================
Answers the question: "How do you confirm the retrieved documents are
actually correct for the query?"

Running the pipeline and getting *some* ranked list proves the machinery
works. It does NOT prove the results are right. This script is the part that
actually checks correctness, using the same approach TREC / BioASQ use to
evaluate biomedical retrieval systems: a small set of test queries with
manually-labeled relevant PMIDs (ground truth), scored with standard
information-retrieval metrics.

What this script does
----------------------
1. Loads a labeled test set: {query -> set of PMIDs known to be relevant}.
   A starter set of 5 queries is included in GOLD_QUERIES below — replace or
   extend these with your own labels (ideally from a domain expert, or from
   a published systematic review's "included studies" list) before trusting
   the numbers for anything real.

2. For each query, runs TWO pipelines and compares them:
     - "baseline": plain NCBI ESearch relevance order (no semantic layer at all)
     - "hybrid":   the full pipeline in this project (BM25 + HNSW fusion +
                   MaxSim late-interaction re-rank)

3. Computes, for each query and each pipeline, at k=5 and k=10:
     - Precision@k   — of the top k results, how many are actually relevant
     - Recall@k      — of all known-relevant docs, how many were retrieved
     - MRR           — reciprocal rank of the first relevant result
     - nDCG@k        — rewards relevant docs appearing higher, not just present

4. Prints a comparison table (baseline vs. hybrid) so you can see, with
   numbers, whether the semantic layer is actually helping — not just
   trust that it should be.

Usage
-----
    python eval_harness.py                 # run with the built-in gold set
    python eval_harness.py --email you@x.com --api-key XXXX

This makes real calls to NCBI (ESearch + EFetch), so it needs internet access
and is subject to the same rate limits as the main app (see ncbi_client.py).

Extending the gold set
-----------------------
Add entries to GOLD_QUERIES, or load them from a JSON file:
    [
      {"query": "...", "relevant_pmids": ["12345678", "23456789", ...]},
      ...
    ]
via --gold-file path/to/labels.json
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field

import config
from ncbi_client import deterministic_parse, esearch, efetch_records
from encoder import load_encoder
from vector_cache import VectorCache
from background_worker import EmbeddingWorkerPool
from lexical_index import BM25Index
from dense_index import DenseIndex
from hybrid import fuse
from reranker import rerank_late_interaction


# ----------------------------------------------------------------------------
# Starter gold-standard test set.
#
# THESE ARE ILLUSTRATIVE PLACEHOLDERS, not verified expert labels. Before
# trusting eval numbers for anything beyond wiring/sanity checks, replace
# these with PMIDs a domain expert has actually reviewed, or pull the
# "included studies" PMID list from a published systematic review that
# matches one of your test queries (many Cochrane reviews list included-study
# PMIDs in their methods/appendix).
# ----------------------------------------------------------------------------
GOLD_QUERIES = [
    {
        "query": "clinical trials for lung cancer in elderly patients",
        "relevant_pmids": [],  # <-- fill in with verified relevant PMIDs
    },
    {
        "query": "statins for cardiovascular disease prevention in adults",
        "relevant_pmids": [],
    },
    {
        "query": "systematic review of exercise for depression in older adults",
        "relevant_pmids": [],
    },
    {
        "query": "randomized controlled trial vaccine efficacy covid-19",
        "relevant_pmids": [],
    },
    {
        "query": "meta-analysis of diabetes management in adolescents",
        "relevant_pmids": [],
    },
]


@dataclass
class QueryResult:
    query: str
    ranked_pmids_baseline: list
    ranked_pmids_hybrid: list
    relevant_pmids: set
    metrics: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# IR metrics
# ----------------------------------------------------------------------------
def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for pmid in top_k if pmid in relevant)
    return hits / k


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return float("nan")
    top_k = ranked[:k]
    hits = sum(1 for pmid in top_k if pmid in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: list, relevant: set) -> float:
    for i, pmid in enumerate(ranked, start=1):
        if pmid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    def dcg(pmids):
        return sum(
            (1.0 / math.log2(i + 1)) for i, pmid in enumerate(pmids[:k], start=1) if pmid in relevant
        )

    actual = dcg(ranked)
    ideal_ranking = list(relevant) + [p for p in ranked if p not in relevant]
    ideal = dcg(ideal_ranking)
    if ideal == 0:
        return float("nan")
    return actual / ideal


def compute_metrics(ranked: list, relevant: set) -> dict:
    if not relevant:
        return {
            "P@5": float("nan"), "P@10": float("nan"),
            "R@5": float("nan"), "R@10": float("nan"),
            "MRR": float("nan"), "nDCG@10": float("nan"),
            "note": "no gold labels for this query — metrics undefined",
        }
    return {
        "P@5": precision_at_k(ranked, relevant, 5),
        "P@10": precision_at_k(ranked, relevant, 10),
        "R@5": recall_at_k(ranked, relevant, 5),
        "R@10": recall_at_k(ranked, relevant, 10),
        "MRR": reciprocal_rank(ranked, relevant),
        "nDCG@10": ndcg_at_k(ranked, relevant, 10),
    }


# ----------------------------------------------------------------------------
# Pipeline runners
# ----------------------------------------------------------------------------
def run_baseline(query: str, retmax: int, email: str, api_key: str) -> list:
    """Plain NCBI ESearch, relevance-sorted, no semantic layer at all."""
    structured_query, _, _ = deterministic_parse(query)
    pmids = esearch(structured_query, retmax=retmax, sort="relevance", email=email, api_key=api_key)
    return pmids


def run_hybrid(query: str, retmax: int, email: str, api_key: str,
               encoder, cache, sparse_weight: float = 0.5, max_workers: int = 4) -> list:
    """Full pipeline: BM25 + HNSW/dense fusion + MaxSim late-interaction re-rank."""
    structured_query, _, _ = deterministic_parse(query)
    query_pooled, query_token_vecs = encoder.encode(query)

    pmids = esearch(structured_query, retmax=retmax, sort="relevance", email=email, api_key=api_key)
    if not pmids:
        return []
    records = efetch_records(pmids, email=email, api_key=api_key)
    if not records:
        return []

    bm25_scores = BM25Index(records).score(query)

    worker = EmbeddingWorkerPool(encoder, cache, max_workers=max_workers)
    pooled_map, token_map, _, _, _ = worker.ensure_embedded(records)
    worker.shutdown()

    dense_index = DenseIndex(dim=encoder.dim, path_prefix=".eval_dense_index")
    for r in records:
        if r["pmid"] in pooled_map:
            dense_index.add(r["pmid"], pooled_map[r["pmid"]])
    dense_index.save()
    dense_scores = dict(dense_index.search(query_pooled, k=len(dense_index)))
    dense_scores = {pmid: s for pmid, s in dense_scores.items() if any(r["pmid"] == pmid for r in records)}

    fused_scores = fuse(bm25_scores, dense_scores, sparse_weight=sparse_weight)
    candidates = sorted(records, key=lambda r: fused_scores.get(r["pmid"], 0.0), reverse=True)

    final_ranked = rerank_late_interaction(query_token_vecs, candidates, token_map)
    return [r["pmid"] for r in final_ranked]


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _fmt(x):
    if isinstance(x, float) and math.isnan(x):
        return "  n/a"
    if isinstance(x, float):
        return f"{x:5.2f}"
    return str(x)


def print_report(results: list):
    print()
    print("=" * 100)
    print("EVALUATION REPORT — baseline (plain ESearch) vs. hybrid (BM25+HNSW+MaxSim)")
    print("=" * 100)

    metric_keys = ["P@5", "P@10", "R@5", "R@10", "MRR", "nDCG@10"]
    header = f"{'Query':<55}{'Pipeline':<10}" + "".join(f"{k:>8}" for k in metric_keys)
    print(header)
    print("-" * len(header))

    totals = {"baseline": {k: [] for k in metric_keys}, "hybrid": {k: [] for k in metric_keys}}

    for r in results:
        q_short = (r.query[:52] + "...") if len(r.query) > 52 else r.query
        for label, m in [("baseline", r.metrics["baseline"]), ("hybrid", r.metrics["hybrid"])]:
            row = f"{q_short:<55}{label:<10}" + "".join(f"{_fmt(m[k]):>8}" for k in metric_keys)
            print(row)
            for k in metric_keys:
                if not (isinstance(m[k], float) and math.isnan(m[k])):
                    totals[label][k].append(m[k])
        print()

    print("-" * len(header))
    print("AVERAGES (across queries with gold labels)")
    for label in ["baseline", "hybrid"]:
        row = f"{'':<55}{label:<10}"
        for k in metric_keys:
            vals = totals[label][k]
            avg = sum(vals) / len(vals) if vals else float("nan")
            row += f"{_fmt(avg):>8}"
        print(row)

    n_labeled = sum(1 for r in results if r.relevant_pmids)
    n_total = len(results)
    print()
    print(f"Queries with gold labels: {n_labeled}/{n_total}")
    if n_labeled < n_total:
        print(
            "⚠️  Queries without gold labels show as 'n/a' and are excluded from the "
            "averages above. Fill in GOLD_QUERIES (or pass --gold-file) with real "
            "relevant-PMID labels before trusting these numbers for anything beyond "
            "confirming the harness itself runs correctly."
        )
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against gold-standard queries.")
    parser.add_argument("--gold-file", type=str, default=None, help="JSON file of {query, relevant_pmids} entries")
    parser.add_argument("--retmax", type=int, default=30, help="Candidates to retrieve per query")
    parser.add_argument("--email", type=str, default=config.NCBI_EMAIL, help="Contact email for NCBI E-utilities (defaults to config.py)")
    parser.add_argument("--api-key", type=str, default=config.NCBI_API_KEY, help="NCBI API key (defaults to config.py)")
    parser.add_argument("--sparse-weight", type=float, default=0.5, help="BM25 weight in hybrid fusion (0-1)")
    args = parser.parse_args()

    if args.gold_file:
        with open(args.gold_file) as f:
            gold_queries = json.load(f)
    else:
        gold_queries = GOLD_QUERIES

    print(f"Loading encoder...")
    encoder, is_real = load_encoder(prefer_real=True)
    print(f"Encoder: {encoder.name if hasattr(encoder, 'name') else encoder} (real: {is_real})")
    if not is_real:
        print(
            "⚠️  Real encoder unavailable — hybrid pipeline results will use a "
            "semantically-meaningless stub. Fix this before trusting hybrid metrics."
        )

    cache = VectorCache(path=".eval_vector_cache.pkl")

    results = []
    for entry in gold_queries:
        query = entry["query"]
        relevant = set(entry.get("relevant_pmids", []))
        print(f"\nRunning: {query!r}  ({len(relevant)} gold-labeled relevant PMIDs)")

        t0 = time.time()
        baseline_ranked = run_baseline(query, args.retmax, args.email, args.api_key)
        print(f"  baseline: {len(baseline_ranked)} results in {time.time()-t0:.1f}s")

        t0 = time.time()
        hybrid_ranked = run_hybrid(
            query, args.retmax, args.email, args.api_key,
            encoder, cache, sparse_weight=args.sparse_weight,
        )
        print(f"  hybrid:   {len(hybrid_ranked)} results in {time.time()-t0:.1f}s")

        qr = QueryResult(
            query=query,
            ranked_pmids_baseline=baseline_ranked,
            ranked_pmids_hybrid=hybrid_ranked,
            relevant_pmids=relevant,
        )
        qr.metrics = {
            "baseline": compute_metrics(baseline_ranked, relevant),
            "hybrid": compute_metrics(hybrid_ranked, relevant),
        }
        results.append(qr)

    cache.save()
    print_report(results)


if __name__ == "__main__":
    main()