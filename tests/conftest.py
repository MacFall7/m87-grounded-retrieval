"""Shared fixtures and helpers.

Two rules govern this suite:

1. The whole non-integration suite must run with no Postgres and no model download.
   A test suite that needs infrastructure to run is a test suite that stops being run,
   and a grounding guarantee nobody re-checks is not a guarantee.
2. Anything that genuinely needs Postgres is marked `integration` and skips cleanly
   when the database is absent, so a reviewer with only Python still sees a green run
   and an honest skip count rather than a wall of errors.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import sys
from typing import Sequence

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grounded_retrieval import leash as leash_mod  # noqa: E402
from grounded_retrieval.retrieval import FusedResult  # noqa: E402
from grounded_retrieval.store import ScoredChunk, StoredChunk  # noqa: E402


def make_chunk(
    chunk_id: str,
    text: str = "body text",
    *,
    repo: str = "demo-repo",
    path: str = "README.md",
    heading: tuple[str, ...] = ("Demo",),
    line_start: int = 1,
    line_end: int = 4,
) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        text=text,
        source_repo=repo,
        source_path=path,
        heading_path=heading,
        line_start=line_start,
        line_end=line_end,
    )


def make_scored(chunk: StoredChunk, score: float, rank: int) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, rank=rank)


def make_fused(
    chunk: StoredChunk,
    *,
    fused_score: float = 0.05,
    rank: int = 1,
    dense_rank: int | None = 1,
    lexical_rank: int | None = 1,
    dense_similarity: float | None = 0.80,
    lexical_score: float | None = 4.0,
) -> FusedResult:
    """Build a FusedResult against the recalibrated target API.

    `dense_similarity` and `lexical_score` are the raw, pre-fusion signals the leash
    now decides on. They are constructed explicitly here rather than derived, because
    the leash tests are about the decision boundary and must not depend on whatever
    the retriever happened to score.
    """
    return FusedResult(
        chunk=chunk,
        fused_score=fused_score,
        rank=rank,
        dense_rank=dense_rank,
        lexical_rank=lexical_rank,
        dense_similarity=dense_similarity,
        lexical_score=lexical_score,
    )


def evaluate(results: Sequence[FusedResult], policy=None, *, query: str | None = None):
    """Call `evaluate_support` through a signature shim.

    The target signature is `evaluate_support(results, policy)`, but query-term
    coverage is a property of the query as well as the spans. This shim passes the
    query only when the implementation declares a parameter for it, so the tests
    express the intended behaviour without hard-coding an argument list that is still
    being settled.
    """
    fn = leash_mod.evaluate_support
    params = inspect.signature(fn).parameters
    if query is not None and "query" in params:
        return fn(results, policy, query=query)
    return fn(results, policy)


def postgres_available() -> bool:
    """Probe once, cheaply. Never let a probe failure surface as a test error."""
    try:
        import psycopg2
    except Exception:
        return False
    dsn = os.environ.get(
        "GROUNDED_RETRIEVAL_DSN",
        "postgresql://postgres:postgres@localhost:5432/postgres",
    )
    try:
        conn = psycopg2.connect(dsn, connect_timeout=2)
    except Exception:
        return False
    conn.close()
    return True


_PG = postgres_available()


def pytest_collection_modifyitems(config, items):
    """Skip integration tests when Postgres is unreachable.

    Applied at collection time rather than inside each test so the reason is reported
    uniformly and a reviewer can see at a glance how much of the suite was skipped.
    """
    if _PG:
        return
    skip = pytest.mark.skip(reason="Postgres with pgvector not reachable")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def corpus_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "corpus"
