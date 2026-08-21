"""
dense_index.py
===============
Real dense (vector-similarity) index over pooled document vectors.

Primary backend: HNSW (Hierarchical Navigable Small World) approximate-nearest-
neighbor search via `hnswlib` / `chroma-hnswlib`, matching the architecture's
"Dense: HNSW — graph-based approximate-nearest-neighbor, finds conceptually
similar documents even with different wording."

Fallback backend: if no HNSW package can be imported at all (no compiled
wheel available for your Python version/platform, and no C++ build toolchain
to compile one), the index automatically falls back to a pure-NumPy exact
cosine-similarity search. This needs zero compiled dependencies, so it always
works. It is honestly labeled as "exact search," not HNSW — at the small
candidate-pool scale this app actually operates at (dozens to a few hundred
documents per session), exact brute-force search is not meaningfully slower
than approximate search, and is strictly more accurate.

Either way, `DenseIndex` exposes the same interface: add(), search(), save(),
__len__(), and a `.backend_name` string so the UI can show which one is active.
"""

import os
import json
import pickle
import numpy as np

try:
    import hnswlib
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False


class _HNSWBackend:
    backend_name = "HNSW (approximate, via hnswlib)"

    def __init__(self, dim: int, path_prefix: str):
        self.dim = dim
        self.index_path = f"{path_prefix}.bin"
        self.meta_path = f"{path_prefix}_meta.json"
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.pmid_to_label = {}
        self.label_to_pmid = {}
        self._next_label = 0
        self._load()

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                meta = json.load(f)
            self.pmid_to_label = meta["pmid_to_label"]
            self.label_to_pmid = {int(k): v for k, v in meta["label_to_pmid"].items()}
            self._next_label = meta["next_label"]
            self.index.load_index(self.index_path, max_elements=max(self._next_label + 2000, 2000))
        else:
            self.index.init_index(max_elements=2000, ef_construction=200, M=16)
        self.index.set_ef(64)

    def save(self):
        self.index.save_index(self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(
                {
                    "pmid_to_label": self.pmid_to_label,
                    "label_to_pmid": self.label_to_pmid,
                    "next_label": self._next_label,
                },
                f,
            )

    def add(self, pmid: str, vector: np.ndarray):
        if pmid in self.pmid_to_label:
            return
        label = self._next_label
        self._next_label += 1
        if label >= self.index.get_max_elements():
            self.index.resize_index(self.index.get_max_elements() + 2000)
        self.index.add_items(vector.reshape(1, -1).astype(np.float32), np.array([label]))
        self.pmid_to_label[pmid] = label
        self.label_to_pmid[label] = pmid

    def search(self, query_vector: np.ndarray, k: int = 50):
        if self._next_label == 0:
            return []
        k = min(k, self._next_label)
        labels, distances = self.index.knn_query(query_vector.reshape(1, -1).astype(np.float32), k=k)
        results = []
        for label, dist in zip(labels[0], distances[0]):
            pmid = self.label_to_pmid.get(int(label))
            if pmid:
                results.append((pmid, 1.0 - float(dist)))
        return results

    def __len__(self):
        return self._next_label


class _BruteForceBackend:
    """Exact cosine-similarity search, pure NumPy, no compiled dependencies."""

    backend_name = "Exact cosine search (NumPy fallback — hnswlib unavailable)"

    def __init__(self, dim: int, path_prefix: str):
        self.dim = dim
        self.path = f"{path_prefix}_bruteforce.pkl"
        self.vectors = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self.vectors = pickle.load(f)
            except Exception:
                self.vectors = {}

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.vectors, f)
        os.replace(tmp, self.path)

    def add(self, pmid: str, vector: np.ndarray):
        self.vectors[pmid] = np.asarray(vector, dtype=np.float32)

    def search(self, query_vector: np.ndarray, k: int = 50):
        if not self.vectors:
            return []
        pmids = list(self.vectors.keys())
        mat = np.stack([self.vectors[p] for p in pmids])
        mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        q = np.asarray(query_vector, dtype=np.float32)
        q_n = q / (np.linalg.norm(q) + 1e-9)
        sims = mat_n @ q_n
        k = min(k, len(pmids))
        order = np.argsort(-sims)[:k]
        return [(pmids[i], float(sims[i])) for i in order]

    def __len__(self):
        return len(self.vectors)


class DenseIndex:
    """
    Picks a real backend at construction time: HNSW if a working hnswlib
    install is available, otherwise the exact NumPy fallback. Same public
    interface either way (add / search / save / len / backend_name).
    """

    def __init__(self, dim: int, path_prefix: str = ".dense_index"):
        if _HNSWLIB_AVAILABLE:
            try:
                self._impl = _HNSWBackend(dim, path_prefix)
            except Exception:
                self._impl = _BruteForceBackend(dim, path_prefix)
        else:
            self._impl = _BruteForceBackend(dim, path_prefix)
        self.backend_name = self._impl.backend_name

    def add(self, pmid: str, vector: np.ndarray):
        self._impl.add(pmid, vector)

    def search(self, query_vector: np.ndarray, k: int = 50):
        return self._impl.search(query_vector, k=k)

    def save(self):
        self._impl.save()

    def __len__(self):
        return len(self._impl)


# Backward-compatible alias — app.py and older code import HNSWIndex directly.
HNSWIndex = DenseIndex