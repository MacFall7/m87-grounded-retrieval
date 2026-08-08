"""The hashing backend, which is what lets this suite run with no model download."""

from __future__ import annotations

import numpy as np
import pytest

from grounded_retrieval.embedding import HashingBackend


def test_hashing_backend_is_deterministic():
    a = HashingBackend().embed_query("exit code 2")
    b = HashingBackend().embed_query("exit code 2")
    assert np.array_equal(a, b)


def test_hashing_backend_vectors_are_unit_length():
    """Normalized at embed time, so cosine reduces to a dot product in the store."""
    vec = HashingBackend().embed_query("policy engine refusal semantics")
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)


def test_hashing_backend_reports_its_dimensions():
    backend = HashingBackend(dimensions=64)
    assert backend.dimensions == 64
    assert backend.embed_query("anything").shape == (64,)


def test_hashing_backend_batches_match_single_encodes():
    backend = HashingBackend()
    batch = backend.embed_documents(["alpha beta", "gamma delta"])
    assert batch.shape == (2, backend.dimensions)
    assert np.allclose(batch[0], backend.embed_query("alpha beta"))


def test_hashing_backend_handles_an_empty_batch():
    backend = HashingBackend()
    assert backend.embed_documents([]).shape == (0, backend.dimensions)


def test_hashing_backend_handles_empty_text():
    """An empty chunk must not raise or produce NaN; it produces a zero vector."""
    vec = HashingBackend().embed_query("")
    assert not np.isnan(vec).any()


def test_fingerprint_distinguishes_the_stub_from_a_real_model():
    """The fingerprint is what makes an accidental stub-built index detectable."""
    assert HashingBackend().fingerprint() == "hashing-stub@256"
    assert HashingBackend(dimensions=64).fingerprint() != HashingBackend().fingerprint()
