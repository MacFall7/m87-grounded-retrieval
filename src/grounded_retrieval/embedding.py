"""Local embedding backend. No API key, no network at query time, no per-token cost.

Design note
-----------
A reviewer who cannot run your system cannot verify your claims about it. Hosted
embedding APIs make a repo unreproducible for anyone without a key and a budget, which
quietly converts "here is a working system" into "here is some code". So the default
backend is a local sentence-transformers model on CPU.

The `EmbeddingBackend` protocol exists so the eval harness can swap in a deterministic
fake. That is not a convenience: an evaluation suite that can only run against a real
model tests the model and the pipeline at the same time, and when it fails you cannot
tell which one broke.

`fingerprint()` is the important part. An index built with one model and queried with
another produces silently degraded results, not an error. Binding the fingerprint into
the index and refusing on mismatch turns a silent correctness bug into a loud failure.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models were trained with an asymmetric query prefix. Omitting it costs real
# retrieval quality, and it is the single most common way a local-embedding RAG
# pipeline underperforms for reasons that never surface as an error.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingBackend(Protocol):
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    def fingerprint(self) -> str: ...


class SentenceTransformerBackend:
    """Local CPU embeddings via sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        self._uses_query_prefix = "bge" in model_name.lower()

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        prefixed = BGE_QUERY_PREFIX + text if self._uses_query_prefix else text
        return self._encode([prefixed])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine reduces to dot product
        )
        return np.asarray(vectors, dtype=np.float32)

    def fingerprint(self) -> str:
        return f"{self.model_name}@{self.dimensions}"


class HashingBackend:
    """Deterministic, dependency-free stand-in for tests and CI.

    Not semantically meaningful, and it is not pretending to be: it exists so the
    pipeline, the store, the fusion, and the refusal logic can be tested without
    downloading 130 MB of weights. Any eval run using this backend must be labeled
    as such, because retrieval-quality numbers from it are meaningless.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimensions, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def fingerprint(self) -> str:
        return f"hashing-stub@{self.dimensions}"
