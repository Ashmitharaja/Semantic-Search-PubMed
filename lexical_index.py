"""
lexical_index.py
=================
Real in-memory sparse (lexical) BM25 index over the current candidate pool.

Flowchart mapping: "Sparse / Lexical: Lucene — In-memory index, BM25 keyword
relevance scoring."

Honesty note: this uses `rank_bm25` (a pure-Python BM25Okapi implementation),
not Apache Lucene itself. The scoring algorithm (BM25) and its role in the
pipeline (in-memory sparse index built over the candidate documents, queried
alongside the dense/HNSW index) are real and match the architecture; the
underlying engine is a lightweight Python reimplementation rather than the JVM
Lucene library NCBI itself runs internally for ESearch.
"""

import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, records: list):
        self.records = records
        self.corpus_tokens = [_tokenize(f"{r['title']} {r['abstract']}") for r in records]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def score(self, query: str) -> dict:
        """Returns {pmid: bm25_score} for every record in this index."""
        if self.bm25 is None:
            return {}
        scores = self.bm25.get_scores(_tokenize(query))
        return {r["pmid"]: float(s) for r, s in zip(self.records, scores)}
