"""
hybrid.py
=========
Real hybrid candidate selection: combines the real in-memory BM25 sparse index
(lexical_index.py) with the real HNSW dense ANN index (dense_index.py).

Flowchart mapping: "Candidate Selection — sparse + dense results combined."

Both score sets are independently min-max normalized to [0, 1] before summing
so neither BM25's unbounded scale nor cosine similarity's [-1, 1] range
dominates the fusion — a standard, honest way to combine heterogeneous
retrieval scores (reciprocal-rank fusion is another common choice; this uses
weighted score fusion for interpretability in the UI).
"""


def _minmax_normalize(scores: dict) -> dict:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse(bm25_scores: dict, dense_scores: dict, sparse_weight: float = 0.5) -> dict:
    """
    bm25_scores:   {pmid: raw BM25 score}       (from lexical_index.BM25Index.score)
    dense_scores:  {pmid: cosine similarity}     (from dense_index.HNSWIndex.search)
    sparse_weight: 0..1, weight given to the (normalized) BM25 side; the dense
                   side gets (1 - sparse_weight)
    Returns {pmid: fused_score} for the union of both score sets.
    """
    bm25_n = _minmax_normalize(bm25_scores)
    dense_n = _minmax_normalize(dense_scores)
    all_pmids = set(bm25_n) | set(dense_n)

    fused = {}
    for pmid in all_pmids:
        s = bm25_n.get(pmid, 0.0)
        d = dense_n.get(pmid, 0.0)
        fused[pmid] = sparse_weight * s + (1 - sparse_weight) * d
    return fused
