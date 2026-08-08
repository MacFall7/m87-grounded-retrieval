"""pgvector-backed chunk store with an index fingerprint guard.

Design note
-----------
Two decisions here are worth defending.

**The fingerprint guard.** The index records the embedding model fingerprint that built
it. Querying with a different model is a silent correctness failure: you get results,
they are ranked, and they are wrong. `assert_compatible()` turns that into a refusal.
This is the same reasoning as Spine Lite's exit-code-2 posture, applied to an index
rather than an agent.

**Cosine over inner product.** Vectors are normalized at embed time, so the two are
equivalent, but declaring `vector_cosine_ops` documents intent and survives someone
later swapping in a backend that does not normalize.

The HNSW index is built after bulk insert, not before. Building it first means paying
the insert-time index maintenance cost on every row for no benefit.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import psycopg2
import psycopg2.extras

from .chunking import Chunk

DEFAULT_DSN = os.environ.get(
    "GROUNDED_RETRIEVAL_DSN",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS gr_index_meta (
    index_name        TEXT PRIMARY KEY,
    model_fingerprint TEXT NOT NULL,
    dimensions        INTEGER NOT NULL,
    chunk_count       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS gr_chunks (
    chunk_id     TEXT PRIMARY KEY,
    index_name   TEXT NOT NULL REFERENCES gr_index_meta(index_name) ON DELETE CASCADE,
    text         TEXT NOT NULL,
    source_repo  TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    heading_path JSONB NOT NULL,
    line_start   INTEGER NOT NULL,
    line_end     INTEGER NOT NULL,
    ordinal      INTEGER NOT NULL,
    embedding    vector NOT NULL
);

CREATE INDEX IF NOT EXISTS gr_chunks_index_name_idx ON gr_chunks (index_name);
"""


class IndexFingerprintMismatch(RuntimeError):
    """Raised when an index is queried with a different embedding model than built it.

    Deliberately fatal. A degraded-but-plausible ranking is worse than an error,
    because nothing downstream can detect it.
    """


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    text: str
    source_repo: str
    source_path: str
    heading_path: tuple[str, ...]
    line_start: int
    line_end: int

    def locator(self) -> str:
        return f"{self.source_repo}:{self.source_path}:L{self.line_start}-L{self.line_end}"

    def context_header(self) -> str:
        return " > ".join(self.heading_path) if self.heading_path else self.source_path


@dataclass(frozen=True)
class ScoredChunk:
    chunk: StoredChunk
    score: float
    rank: int


@contextmanager
def connect(dsn: str = DEFAULT_DSN) -> Iterator[psycopg2.extensions.connection]:
    conn = psycopg2.connect(dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class ChunkStore:
    def __init__(self, dsn: str = DEFAULT_DSN, index_name: str = "default") -> None:
        self.dsn = dsn
        self.index_name = index_name

    def initialize(self, *, model_fingerprint: str, dimensions: int) -> None:
        with connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(SCHEMA)
            cur.execute(
                """
                INSERT INTO gr_index_meta (index_name, model_fingerprint, dimensions)
                VALUES (%s, %s, %s)
                ON CONFLICT (index_name) DO UPDATE
                    SET model_fingerprint = EXCLUDED.model_fingerprint,
                        dimensions        = EXCLUDED.dimensions
                """,
                (self.index_name, model_fingerprint, dimensions),
            )

    def assert_compatible(self, model_fingerprint: str) -> None:
        with connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT model_fingerprint FROM gr_index_meta WHERE index_name = %s",
                (self.index_name,),
            )
            row = cur.fetchone()
        if row is None:
            raise IndexFingerprintMismatch(
                f"index {self.index_name!r} does not exist; run ingest first"
            )
        if row[0] != model_fingerprint:
            raise IndexFingerprintMismatch(
                f"index {self.index_name!r} was built with {row[0]!r} but is being "
                f"queried with {model_fingerprint!r}. Refusing: results would be "
                f"silently wrong rather than absent."
            )

    def clear(self) -> None:
        with connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM gr_chunks WHERE index_name = %s", (self.index_name,))

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector length mismatch: {len(chunks)} vs {len(vectors)}"
            )
        if not chunks:
            return 0

        rows = [
            (
                c.chunk_id,
                self.index_name,
                c.text,
                c.source_repo,
                c.source_path,
                json.dumps(list(c.heading_path)),
                c.line_start,
                c.line_end,
                c.ordinal,
                "[" + ",".join(f"{v:.6f}" for v in vec) + "]",
            )
            for c, vec in zip(chunks, vectors)
        ]

        with connect(self.dsn) as conn, conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO gr_chunks (
                    chunk_id, index_name, text, source_repo, source_path,
                    heading_path, line_start, line_end, ordinal, embedding
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text, embedding = EXCLUDED.embedding
                """,
                rows,
                page_size=200,
            )
            cur.execute(
                """
                UPDATE gr_index_meta
                   SET chunk_count = (SELECT COUNT(*) FROM gr_chunks WHERE index_name = %s)
                 WHERE index_name = %s
                """,
                (self.index_name, self.index_name),
            )
        return len(rows)

    def build_ann_index(self, dimensions: int) -> None:
        """Build the HNSW index after bulk load.

        pgvector needs a fixed dimension on the column before it will accept an ANN
        index, so the column is typed here rather than at CREATE TABLE time. This
        keeps the schema usable across different embedding models.
        """
        with connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"ALTER TABLE gr_chunks ALTER COLUMN embedding TYPE vector({dimensions})"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS gr_chunks_embedding_hnsw "
                "ON gr_chunks USING hnsw (embedding vector_cosine_ops)"
            )

    def dense_search(self, query_vector: np.ndarray, k: int) -> list[ScoredChunk]:
        literal = "[" + ",".join(f"{v:.6f}" for v in query_vector) + "]"
        with connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, text, source_repo, source_path, heading_path,
                       line_start, line_end,
                       1 - (embedding <=> %s::vector) AS similarity
                  FROM gr_chunks
                 WHERE index_name = %s
                 ORDER BY embedding <=> %s::vector
                 LIMIT %s
                """,
                (literal, self.index_name, literal, k),
            )
            rows = cur.fetchall()
        return [
            ScoredChunk(chunk=_row_to_chunk(r), score=float(r[7]), rank=i + 1)
            for i, r in enumerate(rows)
        ]

    def all_chunks(self) -> list[StoredChunk]:
        with connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, text, source_repo, source_path, heading_path,
                       line_start, line_end
                  FROM gr_chunks WHERE index_name = %s ORDER BY chunk_id
                """,
                (self.index_name,),
            )
            return [_row_to_chunk(r) for r in cur.fetchall()]

    def count(self) -> int:
        with connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM gr_chunks WHERE index_name = %s", (self.index_name,)
            )
            return int(cur.fetchone()[0])


def _row_to_chunk(row: Sequence) -> StoredChunk:
    heading = row[4]
    if isinstance(heading, str):
        heading = json.loads(heading)
    return StoredChunk(
        chunk_id=row[0],
        text=row[1],
        source_repo=row[2],
        source_path=row[3],
        heading_path=tuple(heading or ()),
        line_start=int(row[5]),
        line_end=int(row[6]),
    )
