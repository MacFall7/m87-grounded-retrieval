"""End to end over real Postgres with pgvector.

Marked `integration` and skipped when the database is unreachable. Embeddings come
from `HashingBackend`, so even this file downloads no model: it is testing the store,
the fingerprint guard, and the wiring, not the quality of a sentence encoder.
"""

from __future__ import annotations

import os
import uuid

import pytest

from grounded_retrieval.chunking import chunk_markdown
from grounded_retrieval.embedding import HashingBackend
from grounded_retrieval.service import GroundedRetriever
from grounded_retrieval.store import ChunkStore, IndexFingerprintMismatch

pytestmark = pytest.mark.integration

# The `embedding` column is typed once per database by `build_ann_index`, so the stub
# backend has to produce vectors of the deployed dimension or every insert is rejected.
# 384 is the width of the default BGE-small model.
STUB_DIMENSIONS = 384

DSN = os.environ.get(
    "GROUNDED_RETRIEVAL_DSN",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)

DOC = (
    "# Refusal semantics\n\n"
    "The kernel exits with code 2 when a policy denies the request. Refusal is a "
    "first class outcome and carries its own receipt so a caller can branch on it.\n\n"
    "## Hooks\n\n"
    "Hooks observe and never decide. A hook that can veto is a policy under another "
    "name and it breaks the audit trail that the receipt chain depends on.\n"
)


@pytest.fixture()
def store():
    backend = HashingBackend(STUB_DIMENSIONS)
    name = f"test-{uuid.uuid4().hex[:8]}"
    s = ChunkStore(dsn=DSN, index_name=name)
    s.initialize(model_fingerprint=backend.fingerprint(), dimensions=backend.dimensions)
    chunks = chunk_markdown(DOC, source_repo="demo", source_path="README.md")
    s.upsert(chunks, backend.embed_documents([c.embedding_text() for c in chunks]))
    yield s
    s.clear()


def test_upsert_then_count(store):
    assert store.count() > 0


def test_upsert_is_idempotent(store):
    backend = HashingBackend(STUB_DIMENSIONS)
    chunks = chunk_markdown(DOC, source_repo="demo", source_path="README.md")
    before = store.count()
    store.upsert(chunks, backend.embed_documents([c.embedding_text() for c in chunks]))
    assert store.count() == before


def test_dense_search_returns_ranked_rows(store):
    hits = store.dense_search(HashingBackend(STUB_DIMENSIONS).embed_query("kernel exits code"), k=5)
    assert hits
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_round_trip_preserves_provenance(store):
    chunk = store.all_chunks()[0]
    assert chunk.source_repo == "demo"
    assert chunk.source_path == "README.md"
    assert chunk.line_start <= chunk.line_end
    assert chunk.locator().startswith("demo:README.md:L")


def test_fingerprint_mismatch_refuses(store):
    """A wrong-model query returns plausible, wrongly ranked rows. That must be fatal."""
    with pytest.raises(IndexFingerprintMismatch):
        store.assert_compatible("some-other-model@384")


def test_missing_index_refuses():
    missing = ChunkStore(dsn=DSN, index_name=f"absent-{uuid.uuid4().hex[:8]}")
    with pytest.raises(IndexFingerprintMismatch):
        missing.assert_compatible(HashingBackend(STUB_DIMENSIONS).fingerprint())


def test_retriever_answers_an_in_corpus_query(store):
    retriever = GroundedRetriever(store, HashingBackend(STUB_DIMENSIONS))
    answer, receipt = retriever.query("what does the kernel do when a policy denies")
    assert receipt.query
    assert receipt.model_fingerprint == f"hashing-stub@{STUB_DIMENSIONS}"
    assert receipt.retrieved
    assert answer.exit_code in (0, 2)


def test_retriever_refuses_an_out_of_corpus_query(store):
    """The thesis, end to end: no evidence means no answer, not a hedged answer.

    This is the case the recalibration exists for. A fused RRF score cannot express
    aboutness: an off-topic query still produces a top-ranked chunk with a fused score
    well above any constant floor. Raw similarity and query-term coverage can.
    """
    retriever = GroundedRetriever(store, HashingBackend(STUB_DIMENSIONS))
    answer, receipt = retriever.query(
        "what were the quarterly gross margins for the hardware division in 2019"
    )
    assert answer.refused
    assert answer.exit_code == 2
    assert answer.answer == ""
    assert receipt.exit_code == 2
