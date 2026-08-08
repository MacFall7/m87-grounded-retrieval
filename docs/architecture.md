# Architecture note

Scope of this document: why the pieces are shaped the way they are, and where the
seams are. The README covers what the system does. This covers what it would cost to
change.

## Module boundaries

| Module | Depends on | Pure? |
|---|---|---|
| `chunking.py` | nothing | yes |
| `embedding.py` | numpy, optionally sentence-transformers | yes, given a fixed model |
| `store.py` | psycopg2, pgvector | no, this is the I/O boundary |
| `retrieval.py` | `store` types only | BM25 is in-memory, RRF is pure |
| `leash.py` | `retrieval` types only | yes |
| `service.py` | all of the above | no, it orchestrates |

The dependency arrows run one way. `leash.py` knows nothing about Postgres, and
`chunking.py` knows nothing about embeddings. This is not tidiness for its own sake: it
is what makes the whole non-integration test suite runnable with no database and no
model weights, and a suite that needs infrastructure to run is a suite that stops being
run.

## The pipeline in order

### 1. Ingest

`corpus/<repo>/<path...>.md` is walked in sorted order. The first path segment is the
source repo and the rest is the path within it, so provenance is derived from the
filesystem rather than from a manifest that can drift out of sync with the files.

### 2. Chunking

Documents are split at heading boundaries, with a heading stack maintained so each
block carries its full breadcrumb. Code fences are tracked so a `# comment` inside a
shell block does not register as a heading. Blocks larger than `max_chars` are split on
paragraph boundaries, and the overlap between adjacent pieces is applied in whole
paragraphs rather than as a raw character slice, so an overlapping chunk is still
readable prose. Fragments below `min_chars` are dropped: a lone heading or a badge row
scores well against short queries on cosine similarity and contains nothing.

`chunk_id` is `sha256(repo, path, heading path, line span, text)` truncated to 16 hex
characters. Ordinal is deliberately excluded, so inserting a paragraph early in a file
does not renumber every chunk downstream of it.

### 3. Embedding

Local sentence-transformers on CPU, defaulting to `BAAI/bge-small-en-v1.5`. Vectors are
normalized at encode time, which makes cosine equivalent to a dot product. Queries get
the BGE asymmetric prefix; omitting it costs real retrieval quality and never surfaces
as an error.

`fingerprint()` returns `model@dimensions` and is written into the index metadata.

`HashingBackend` is a deterministic, dependency-free stand-in. It is not semantically
meaningful and does not pretend to be.

### 4. Store

pgvector on Postgres 16. Two decisions:

- The embedding column is left untyped at `CREATE TABLE` and given a fixed dimension by
  `build_ann_index` after bulk load. That keeps the schema usable across embedding
  models, and it avoids paying index-maintenance cost on every insert. The tradeoff is
  that one database serves one embedding width at a time, which is why the integration
  tests pin the stub backend to the deployed dimension.
- `assert_compatible()` refuses when the querying model's fingerprint differs from the
  one that built the index. A mismatched index does not error, it returns plausible
  results in the wrong order. Nothing downstream can detect that, so it has to be
  fatal here.

### 5. Retrieval

Dense top-k from pgvector and BM25 top-k from an in-memory index over the same chunks,
fused with Reciprocal Rank Fusion:

```
score(d) = sum over retrievers of  weight / (k + rank(d))
```

`k = 60` from Cormack et al. 2009, kept as a named default so it can be swept.

The two legs fail in different directions, which is the entire reason to run both.
Dense finds "how does it refuse" when the document says "fail-closed behaviour" and
shares no words with the query. BM25 finds `exit code 2` and `vector_cosine_ops`, exact
identifiers where an embedding model does not know it is looking at a token that must
match literally. Technical corpora are full of the second kind, so a dense-only
pipeline over a codebase quietly loses the queries users are most likely to type. The
tokenizer keeps underscores for the same reason.

BM25 is rebuilt in memory at retriever construction. At this corpus size that is exact,
has no index-maintenance path to get wrong, and rebuilds in under a second. The
replacement at a scale where it stops being right is Postgres full-text search behind
the same interface.

Fusion is a pure function of two rank lists, which is why it lives in its own module
instead of inside a query method: it is fully unit-testable with no model and no
database. Ties break on `chunk_id`, producing a total order, because nondeterministic
ranking makes regression detection impossible.

### 6. The leash

`evaluate_support` is the gate. It runs before generation and returns a `LeashVerdict`.

The recalibration worth documenting: the decision is made on raw semantic similarity
and query-term coverage, not on the fused RRF score. RRF discards magnitude on purpose,
so the fused score is a rank artefact. It reports that a chunk won the fusion, not that
the chunk is about the query, and every query produces a winner. A constant floor on it
therefore admits off-topic evidence for exactly the queries the leash exists to catch.

Two thresholds, because there are two distinct failure modes:

- **Low similarity.** Nothing in the corpus is close to the query. Refuse.
- **Low query-term coverage.** Something is semantically adjacent but does not cover
  the query's terms. This is the case where a chunk about audio presets scores well
  against a question about index dimensions. Refuse.

The threshold applies to the *top* candidate rather than the mean, because mean support
rewards a uniformly mediocre result set. One strong span is grounds to answer, ten weak
ones are not. `min_supporting_chunks` guards the opposite failure, where a single lucky
heading match carries an answer it cannot support.

### 7. Generation and verification

If the verdict is `ANSWER`, the generator is called with the admitted spans and nothing
else. Its output is split into sentence-level claims, because sentence granularity is
where a RAG answer actually goes wrong: three sourced sentences and one invented one.
Verifying an answer as a unit lets the invented sentence ride along on its neighbours.

Each claim's content words, stopwords removed, are checked against the admitted spans'
tokens. Below `claim_overlap_threshold` the claim is stripped. If everything is
stripped, the response degrades to `REFUSE_EMPTY_AFTER_STRIPPING` rather than shipping
unsupported text.

### 8. Receipt

Emitted on every query including refusals, and hashed to a digest. It records inputs,
fingerprints, intermediate rankings, and the decision, so a disputed answer can be
re-derived rather than believed. A record that only captures the output is a log line.

## Testing strategy

The unit suite covers chunking purity and determinism, heading-path extraction, code
fence handling, oversized-block splitting, `chunk_id` stability, BM25 scoring, RRF as a
pure function including tie-break determinism, the leash refusal paths, and claim
verification and stripping. None of it touches Postgres and none of it downloads a
model.

Integration tests are marked and skipped when the database is unreachable. They cover
upsert idempotence, dense search, provenance round-trip, the fingerprint guard, and one
end-to-end refusal on an out-of-corpus query.

## Known seams

- One embedding width per database, as described above.
- BM25 is rebuilt per retriever instance, so constructing a retriever per request would
  be quadratic in corpus size.
- Claim verification is lexical. It is a floor, not an entailment check.
- The thresholds are hand-calibrated against this corpus and their generalization is
  untested.
