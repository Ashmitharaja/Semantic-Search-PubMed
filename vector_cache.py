"""
vector_cache.py
================
Real in-memory vector cache, persisted to disk between runs.

Flowchart mapping: "In-Memory Token/Vector Cache — fast RAM store of already-
generated representations, checked first for each candidate."
"""

import os
import pickle
import threading


class VectorCache:
    def __init__(self, path: str = ".vector_cache.pkl"):
        self.path = path
        self._lock = threading.Lock()
        self._store = {}
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self._store = pickle.load(f)
            except Exception:
                self._store = {}

    def save(self):
        with self._lock:
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self._store, f)
            os.replace(tmp, self.path)

    def get(self, pmid: str):
        with self._lock:
            v = self._store.get(pmid)
        if v is not None:
            self.hits += 1
        else:
            self.misses += 1
        return v

    def peek(self, pmid: str):
        """Like get(), but doesn't affect hit/miss counters."""
        with self._lock:
            return self._store.get(pmid)

    def set(self, pmid: str, pooled, token_vecs):
        with self._lock:
            self._store[pmid] = (pooled, token_vecs)

    def __len__(self):
        with self._lock:
            return len(self._store)

    def stats(self):
        total = self.hits + self.misses
        rate = (self.hits / total) if total else 0.0
        return {"size": len(self), "hits": self.hits, "misses": self.misses, "hit_rate": rate}
