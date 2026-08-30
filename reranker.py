"""
reranker.py
============
Real token-level late-interaction re-ranking (ColBERT-style MaxSim), vectorized
with NumPy — NumPy's matrix multiply is BLAS-backed and uses CPU SIMD
instructions under the hood, so "SIMD CPU operations" is a fair description of
what actually executes here.

Flowchart mapping: "Tensor Re-Ranking — Late Interaction: query-token vs.
document-token comparison (not single overall vectors), compatibility score
computed via SIMD CPU operations."

This is the real MaxSim formula used by ColBERT: for every query token vector,
take its maximum cosine similarity against any document token vector, then sum
those maxima across all query tokens.
"""

import numpy as np


def maxsim_score(query_token_vecs: np.ndarray, doc_token_vecs: np.ndarray) -> float:
    if query_token_vecs.size == 0 or doc_token_vecs.size == 0:
        return 0.0
    # both are L2-normalized -> dot product == cosine similarity
    sims = query_token_vecs @ doc_token_vecs.T  # (q_tokens, d_tokens)
    return float(sims.max(axis=1).sum())


def rerank_late_interaction(query_token_vecs: np.ndarray, records: list, token_map: dict):
    """
    records: list of record dicts (must include 'pmid')
    token_map: {pmid: token_vecs}
    Returns records sorted by MaxSim score, each with a 'late_interaction_score' key added.
    """
    scored = []
    for rec in records:
        doc_tokens = token_map.get(rec["pmid"])
        score = maxsim_score(query_token_vecs, doc_tokens) if doc_tokens is not None else 0.0
        rec = dict(rec)
        rec["late_interaction_score"] = score
        scored.append(rec)
    scored.sort(key=lambda r: r["late_interaction_score"], reverse=True)
    return scored