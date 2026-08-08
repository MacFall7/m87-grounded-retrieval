# m87-grounded-retrieval

A hybrid retrieval service over Postgres and pgvector that refuses to answer when the
corpus does not support an answer.

The retrieval half is conventional: structure-aware markdown chunking with provenance,
local sentence-transformers embeddings, BM25 over the same chunks, and Reciprocal Rank
Fusion over the two rank lists. The part worth reading is the Citation Leash.

MIT licensed. Python 3.11+.

## The Citation Leash

Most RAG hallucination is not a generation problem. It is a permission problem. A
question arrives, retrieval returns weak or off-topic context, and the model answers
anyway from parametric memory. The output is fluent, it looks sourced, and it is wrong.
Instructing the model to "only use the provided context" is a request, not a
constraint.

So this service decides whether an answer is permitted **before** generation runs, in
deterministic code the model does not participate in:

1. Retrieve. Dense and lexical, fused with RRF.
2. Evaluate support against a policy. Pure function, no model, no I/O.
3. If support is insufficient, **refuse**. The generator is never called. There is no
   hedged answer, because "I could not find much, but possibly ..." is a hallucination
   with a disclaimer attached.
4. If support is sufficient, generate strictly over the admitted spans, then verify
   every emitted claim still maps to one. A claim that does not is stripped. If
   stripping empties the answer, the whole response degrades to a refusal.

Step 3 is the ordering that matters. A check that runs after the model has spoken can
only edit output. It cannot withhold it.

### Refusal is a success path

A refusal is not an error. It has its own decision value, its own receipt, and its own
tests. `LeashDecision` uses exit-code semantics borrowed from the Spine Lite kernel
contract:

| Decision | Exit code | Meaning |
|---|---|---|
| `ANSWER` | 0 | Support cleared the policy. An answer bound to spans was produced. |
| `REFUSE_NO_SUPPORT` | 2 | Retrieval returned nothing usable. |
| `REFUSE_LOW_SIMILARITY` | 2 | Best candidate is not semantically close enough to the query. |
| `REFUSE_LOW_COVERAGE` | 2 | Candidates do not cover enough of the query's terms. |
| `REFUSE_INSUFFICIENT_SPANS` | 2 | Fewer spans cleared the policy than are required. |
| `REFUSE_EMPTY_AFTER_STRIPPING` | 2 | Everything generated failed span verification. |

Exit code 2 means incapacity. Exit code 1 would mean a crash. Conflating the two loses
the distinction a caller needs in order to branch correctly, which is the same reason
the Spine Lite kernel separates them.

### What support is measured on

Support is evaluated on raw retrieval signals, not on the fused RRF score. Fusion
throws away magnitude on purpose, because BM25 and cosine are not on a common scale, so
a fused score tells you a chunk won the fusion and nothing about whether it is about the
query. Every query, answerable or not, produces a top-ranked chunk with a respectable
fused score. Thresholding on it therefore admits off-topic evidence.

`LeashPolicy` thresholds:

| Field | Default | What it guards |
|---|---|---|
| `min_semantic_similarity` | 0.45 | Raw cosine on the best candidate. Aboutness. |
| `min_query_term_coverage` | 0.30 | Fraction of the query's content terms present in the candidates. Catches the semantically-adjacent-but-wrong case. |
| `min_supporting_chunks` | 1 | Guards a single lucky heading match carrying a whole answer. |
| `max_spans` | 5 | Upper bound on admitted evidence. |
| `claim_overlap_threshold` | 0.55 | Fraction of a claim's content words that must appear in the admitted spans. |

Thresholds are configuration, not constants buried in a function, so they can be swept.
A threshold you cannot sweep is a threshold you cannot defend.

### Receipts

Every query emits a `QueryReceipt`: the query, the index and model fingerprints, the
retrieved chunk IDs with their ranks and scores, the admitted span locators, the
per-claim verification results, the stripped claims, the decision, and the policy in
force. It hashes to a digest. The point is that a disputed answer can be re-derived by
someone who does not trust the narrative.

## Quickstart

```bash
git clone https://github.com/MacFall7/m87-grounded-retrieval
cd m87-grounded-retrieval
make install
```

Run the tests first. They need neither Postgres nor a model download:

```bash
make test
```

Then bring up the database and ingest the corpus:

```bash
docker compose up -d          # pgvector/pgvector:pg16 on localhost:5432
make ingest                   # downloads BAAI/bge-small-en-v1.5 on first run, CPU only
make query Q="how does the kernel handle a policy denial"
```

`make query` prints the receipt and the answer, and exits 0 on an answer or 2 on a
refusal, so it composes in a shell pipeline without parsing output.

To smoke test the whole path with no model download, `make ingest-stub` uses the
deterministic hashing backend. Retrieval quality numbers from that backend are
meaningless and any run using it must be labeled as such.

## Architecture

```
corpus/<repo>/<path>.md
        |
        v
  chunking.py     structure-aware split, heading path, line span, stable chunk_id
        |
        v
  embedding.py    local sentence-transformers on CPU, or a hashing stub for tests
        |
        v
  store.py        pgvector, HNSW cosine index, model fingerprint bound to the index
        |
        v
  retrieval.py    dense top-k + BM25 top-k, fused with RRF (pure function)
        |
        v
  leash.py        evaluate_support -> ANSWER or REFUSE   (pure function, pre-generation)
        |
        v
  service.py      generate over admitted spans, verify claims, emit receipt
```

Notes on specific choices:

- **Chunking preserves structure.** Chunking by character count mixes topics across a
  heading boundary, which makes a chunk retrievable for two subjects and precise for
  neither. Headings inside code fences do not split blocks, because governance repos
  are dense with shell blocks full of `# comment` lines and a naive heading regex
  shreds them.
- **Provenance is first class.** Every chunk carries source repo, file path, heading
  path, and line span. `locator()` produces `repo:path:L40-L58`, which a reviewer can
  check in a few seconds. "This came from README.md" is an attribution, not a citation.
- **`chunk_id` is content-addressed.** Re-ingesting an unchanged corpus produces
  identical IDs, so the index is idempotent and only changed chunks get new rows.
- **RRF rather than score blending.** Weighted sums of cosine and BM25 require a
  normalization that is itself a tuned parameter and that drifts as the corpus grows.
  RRF has one interpretable constant and is a pure function of two rank lists.
  Tie-breaking is on `chunk_id`, so the ordering is a total order and a baseline can
  detect regressions.
- **The index records the model fingerprint.** Querying an index with a different
  embedding model returns results that are ranked and wrong. `assert_compatible()`
  turns that silent correctness failure into a refusal.
- **The default generator is extractive.** No LLM, no API key. An extractive answer
  cannot hallucinate, so a faithfulness failure against it isolates to retrieval rather
  than confounding retrieval and generation. Pass any callable as `generator` to
  `GroundedRetriever.query`; the leash applies identically, which is the point.

See `docs/architecture.md` for the longer version.

## Tests

```bash
make test        # unit suite, no Postgres, no model download
make test-all    # adds the integration tests against a live pgvector
```

Integration tests are marked `@pytest.mark.integration` and skip cleanly when Postgres
is unreachable, so a reviewer with only Python gets a green run and an honest skip
count rather than a wall of connection errors. CI runs `-m "not integration"` on push.

## What this does not demonstrate

Stated plainly, because a README that overstates its repo is worse evidence than no
repo.

- **Not production RAG.** This has never served a real user, has no traffic, no
  latency budget, no on-call rotation, and no operational history. Nothing here is
  evidence of running retrieval in production or at scale.
- **Not a scale claim.** The corpus is a few dozen markdown files. BM25 is rebuilt in
  memory on startup, which is the right call at this size and the wrong call at any
  serious one. There is no sharding, no incremental index maintenance, no cache, and no
  concurrency story.
- **No retrieval quality measurement in this repo.** There are no hit@k, MRR, nDCG,
  faithfulness, or context-precision numbers here, and none should be inferred from a
  green test suite. The tests check that components behave as specified; they do not
  measure how good the retrieval is. Measurement is a separate harness.
- **The default generator is extractive, not generative.** Refusal behaviour under a
  real LLM generator is not exercised here beyond the interface.
- **The hashing embedding backend is not semantic.** It exists so the suite runs with
  no model download. Any number produced with it is meaningless as a quality signal.
- **Claim verification is lexical.** Overlap of content words against admitted spans is
  a deliberately conservative floor, not a semantic entailment check. It will pass a
  paraphrase-free contradiction that reuses the span's vocabulary, and it will fail a
  correct paraphrase that does not. The floor is chosen because it holds with no model
  and no API, and a grounding guarantee that depends on an API being up is not a
  guarantee.
- **No adversarial security work.** No prompt injection defence, no multi-tenant
  isolation, no authz on the index, no rate limiting.
- **No evaluation of the thresholds themselves.** The defaults are calibrated by hand
  against this corpus. Whether they generalize is untested.
