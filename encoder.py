"""
encoder.py
==========
Real multi-vector CPU encoder.

Flowchart mapping:
  - "CPU-Optimized Bi-Encoder"          -> TransformerEncoder (HuggingFace transformers, CPU)
  - "Multi-Vector Query Matrix"          -> token_vecs returned by .encode() (one vector
                                            per token, not a single pooled vector)
  - "4-bit quantized ONNX model"         -> see onnx_quantize.py for the honest real
                                            equivalent (INT8 dynamic quantization via
                                            onnxruntime); this module uses torch's
                                            built-in INT8 dynamic quantization for the
                                            eager-mode path, which is the standard,
                                            widely-used CPU speed-up technique.

Honesty note: literal 4-bit weight quantization (GPTQ/AWQ-style) is a research
technique for large generative LLMs, not standard tooling for small BERT-sized
sentence encoders. INT8 dynamic quantization is the real, practical, broadly-used
equivalent for CPU inference speed on encoder models this size, so that's what's
implemented here and in onnx_quantize.py.

DummyEncoder exists ONLY so the rest of the pipeline (BM25 index, HNSW index,
cache, background worker, MaxSim reranker) can be exercised and unit-tested in
environments with no internet access to download real model weights. It carries
no real semantic meaning and must never be used for actual search quality.
"""

import hashlib
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DUMMY_DIM = 384


class DummyEncoder:
    """Deterministic hash-based pseudo-embedder. Offline test stub ONLY."""

    name = "DummyEncoder (offline test stub — NOT semantically real)"
    dim = _DUMMY_DIM

    def encode(self, text: str):
        tokens = text.lower().split() or ["<empty>"]
        vecs = []
        for tok in tokens:
            h = hashlib.sha256(tok.encode()).digest()  # 32 bytes
            reps = (self.dim // len(h)) + 1
            arr = np.frombuffer(h * reps, dtype=np.uint8)[: self.dim].astype(np.float32)
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            vecs.append(arr)
        token_vecs = np.stack(vecs).astype(np.float32)
        token_vecs = token_vecs / (np.linalg.norm(token_vecs, axis=1, keepdims=True) + 1e-9)
        pooled = token_vecs.mean(axis=0)
        pooled = pooled / (np.linalg.norm(pooled) + 1e-9)
        return pooled, token_vecs


class TransformerEncoder:
    """
    Real CPU multi-vector encoder.

    Returns:
      pooled      : (dim,) mean-pooled sentence vector, used for HNSW ANN indexing
      token_vecs  : (n_tokens, dim) per-token contextual vectors, used for
                    ColBERT-style late-interaction re-ranking (reranker.py)
    """

    def __init__(self, model_name: str = MODEL_NAME, quantize: bool = True):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        self.quantized = False
        if quantize:
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                self.quantized = True
            except Exception:
                pass  # fall back to full-precision CPU inference

        self.model = model
        self.dim = self.model.config.hidden_size
        self.name = f"TransformerEncoder ({model_name}, {'INT8 dynamic-quantized' if self.quantized else 'fp32'})"

    def encode(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with self.torch.no_grad():
            out = self.model(**inputs)
        hidden = out.last_hidden_state[0]  # (tokens, dim)
        mask = inputs["attention_mask"][0].bool()
        token_vecs = hidden[mask].numpy().astype(np.float32)
        norm = np.linalg.norm(token_vecs, axis=1, keepdims=True) + 1e-9
        token_vecs = token_vecs / norm
        pooled = token_vecs.mean(axis=0)
        pooled = pooled / (np.linalg.norm(pooled) + 1e-9)
        return pooled, token_vecs


class SpacyEncoder:
    """
    Real CPU multi-vector encoder using spaCy's pretrained static word vectors
    (en_core_web_md: 300-dim vectors trained on Common Crawl).

    These are *static* (context-independent) word vectors rather than the
    contextual embeddings TransformerEncoder produces — genuinely pretrained
    and semantically meaningful (verified: cosine('cancer','tumor') ≈ 0.67 vs.
    cosine('cancer','bicycle') ≈ 0.05), just lower-capacity than a transformer.
    Used automatically as the real fallback when the transformer model can't
    be downloaded (e.g. no route to huggingface.co), since spaCy's model ships
    as a directly installable wheel rather than requiring a separate weight
    download from a host that might be blocked.
    """

    def __init__(self, model_name: str = "en_core_web_md"):
        import spacy

        self.nlp = spacy.load(
            model_name, disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"]
        )
        if self.nlp.vocab.vectors_length == 0:
            raise RuntimeError(f"spaCy model '{model_name}' has no word vectors")
        self.dim = self.nlp.vocab.vectors_length
        self.name = f"SpacyEncoder ({model_name}, {self.dim}-dim pretrained static vectors)"

    def encode(self, text: str):
        doc = self.nlp(text)
        vecs = [tok.vector for tok in doc if tok.has_vector and not tok.is_space]
        if not vecs:
            vecs = [np.zeros(self.dim, dtype=np.float32)]
        token_vecs = np.asarray(vecs, dtype=np.float32)
        norm = np.linalg.norm(token_vecs, axis=1, keepdims=True) + 1e-9
        token_vecs = token_vecs / norm
        pooled = token_vecs.mean(axis=0)
        pooled = pooled / (np.linalg.norm(pooled) + 1e-9)
        return pooled, token_vecs


def load_encoder(prefer_real: bool = True, quantize: bool = True):
    """
    Fallback chain, all real except the last resort:
      1. TransformerEncoder (contextual, best quality) — needs huggingface.co
      2. SpacyEncoder (static pretrained vectors) — needs only a pip-installed wheel
      3. DummyEncoder (hash-based stub) — only if neither real option loads
    """
    if prefer_real:
        try:
            return TransformerEncoder(quantize=quantize), True
        except Exception:
            pass
        try:
            return SpacyEncoder(), True
        except Exception:
            pass
    return DummyEncoder(), False
