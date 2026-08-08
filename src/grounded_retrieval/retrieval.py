"""Hybrid retrieval: dense vectors plus BM25, fused with Reciprocal Rank Fusion.

Design note
-----------
Dense retrieval and lexical retrieval fail in different directions, which is the whole
reason to run both. Dense search finds "how does it refuse" when the document says
"fail-closed behavior" and shares no words with the query. BM25 finds `exit code 2`,
`vector_cosine_ops`, and `EC-M87HUB` -- exact identifiers where an embedding model has
no idea it is looking at a token that must match literally. Technical corpora are full
of the second kind, so a dense-only pipeline over a codebase quietly loses the queries
users are most likely to type.

**Why RRF and not score blending.** Cosine similarity and BM25 scores are not on a
common scale, and BM25's scale moves with corpus statistics. Normalizing them into a
weighted sum means picking a normalization that is itself a tuned parameter, and it
drifts as the corpus grows. RRF discards the magnitudes and fuses ranks:

    score(d) = sum over retrievers of  weight / (k + rank(d))

It has one interpretable constant, it cannot be destabilized by an outlier score, and
it is a pure function of two rank lists -- which means it is fully unit-testable with
no model and no database. That last property is why the fusion logic lives in its own
module instead of inside a query method.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from .store import ScoredChunk, StoredChunk

TOKEN = re.compile(r"[a-z0-9_]+")

# k=60 is the constant from the original RRF paper (Cormack et al., 2009). Kept as a
# named default rather than a magic number so it can be swept in the eval harness.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization that preserves underscores.

    Underscores are kept because identifiers like `governed_request` and
    `vector_cosine_ops` are exactly the tokens lexical search exists to catch.
    Splitting them would hand those queries back to the dense retriever, which is
    the retriever that cannot answer them.
    """
    return TOKEN.findall(text.lower())


@dataclass(frozen=True)
class FusedResult:
    chunk: StoredChunk
    fused_score: float
    rank: int
    dense_rank: int | None
    lexical_rank: int | None
    # Raw, un-fused signals carried through deliberately. RRF destroys magnitude
    # by design, which is what makes it a good RANKER and a useless SUPPORT
    # signal. The Citation Leash needs to know how similar the top hit actually
    # is, not merely that something came back first. See docs/adr-001.
    dense_similarity: float | None = None
    lexical_score: float | None = None

    @property
    def found_by_both(self) -> bool:
        return self.dense_rank is not None and self.lexical_rank is not None


class BM25Index:
    """Okapi BM25 over the chunk set.

    Built in memory from the store. For a corpus this size that is the right call:
    it is exact, it has no index-maintenance path to get wrong, and rebuild is
    sub-second. At a scale where it stops being the right call, the replacement is
    Postgres full-text search behind this same interface.
    """

    def __init__(self, chunks: Sequence[StoredChunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        # Heading path is indexed alongside body text so a query naming a section
        # can match on the section title even when the body never repeats it.
        self._docs = [tokenize(c.context_header() + " " + c.text) for c in self.chunks]
        self._lengths = [len(d) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

        self._df: dict[str, int] = {}
        for doc in self._docs:
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1

        self._tf: list[dict[str, int]] = []
        for doc in self._docs:
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self._tf.append(counts)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms or not self._docs:
            return []

        scores: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            length = self._lengths[i] or 1
            total = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_len or 1)
                )
                total += self._idf(term) * (freq * (self.k1 + 1)) / denom
            if total > 0:
                scores.append((i, total))

        scores.sort(key=lambda x: (-x[1], self.chunks[x[0]].chunk_id))
        return [
            ScoredChunk(chunk=self.chunks[i], score=s, rank=rank + 1)
            for rank, (i, s) in enumerate(scores[:k])
        ]


def reciprocal_rank_fusion(
    dense: Sequence[ScoredChunk],
    lexical: Sequence[ScoredChunk],
    *,
    k: int = RRF_K,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    limit: int = 10,
) -> list[FusedResult]:
    """Fuse two ranked lists. Pure function: no model, no database, no I/O.

    Ties are broken on chunk_id so the output is a total order and the eval harness
    produces identical numbers across runs. Nondeterministic ranking makes regression
    detection impossible, which defeats the point of having a baseline.
    """
    contributions: dict[str, float] = {}
    chunks: dict[str, StoredChunk] = {}
    dense_ranks: dict[str, int] = {}
    lexical_ranks: dict[str, int] = {}
    dense_scores: dict[str, float] = {}
    lexical_scores: dict[str, float] = {}

    for hit in dense:
        cid = hit.chunk.chunk_id
        contributions[cid] = contributions.get(cid, 0.0) + dense_weight / (k + hit.rank)
        chunks[cid] = hit.chunk
        dense_ranks[cid] = hit.rank
        dense_scores[cid] = hit.score

    for hit in lexical:
        cid = hit.chunk.chunk_id
        contributions[cid] = contributions.get(cid, 0.0) + lexical_weight / (k + hit.rank)
        chunks[cid] = hit.chunk
        lexical_ranks[cid] = hit.rank
        lexical_scores[cid] = hit.score

    ordered = sorted(contributions.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        FusedResult(
            chunk=chunks[cid],
            fused_score=score,
            rank=i + 1,
            dense_rank=dense_ranks.get(cid),
            lexical_rank=lexical_ranks.get(cid),
            dense_similarity=dense_scores.get(cid),
            lexical_score=lexical_scores.get(cid),
        )
        for i, (cid, score) in enumerate(ordered[:limit])
    ]
