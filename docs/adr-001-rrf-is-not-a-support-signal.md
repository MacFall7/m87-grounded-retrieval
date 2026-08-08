# ADR-001: Reciprocal Rank Fusion is not a support signal

**Status:** accepted, 2026-08-08
**Supersedes:** the original `LeashPolicy.min_support_score` gate

---

## The defect

The first shipped version of the Citation Leash gated on the RRF fused score:

```python
top_score = results[0].fused_score
if top_score < policy.min_support_score:   # default 0.015
    return refuse(...)
```

That gate never refused anything. Measured, not theorized: an out-of-corpus control
query, "what is the best pizza topping in Naples", was answered with three confident
citations pulled from a Python API reference.

## Why it could not work

Reciprocal Rank Fusion computes

```
score(d) = sum over retrievers of  weight / (k + rank(d))
```

with `k = 60`. Two properties follow directly, and together they make the score
useless as evidence of support.

**RRF encodes position, not quality.** A document at dense rank 1 contributes exactly
`1/61 = 0.0164` whether it is a verbatim answer or unrelated noise. The score is a
function of *where a result landed*, and something always lands first.

**RRF destroys magnitude on purpose.** That is the entire reason to prefer it over
weighted score blending: cosine similarity and BM25 scores are not on a common scale,
and BM25's scale drifts with corpus statistics. Discarding magnitude is what makes RRF
a stable ranker. It is also what makes it incapable of answering "how good is this
match, in absolute terms?"

So the threshold `0.015` sat *below the structural floor* of `0.0164` that any single
rank-1 result produces. Every query that returned anything at all cleared it. The
condition was unfalsifiable.

## The general failure

The gate asserted **shape**: a score exists, and it is above a constant.

The gate needed to assert **state**: the retrieved text is actually about the question.

This is the same defect as a test suite where every boundary is mocked. The assertions
pass, the suite is green, and it is answering a different question than the one anyone
cared about. A gate that cannot fail is not a gate. Both halves of this repository pair
exist because that class of error is easy to ship and hard to see: the system looks
more rigorous with a broken gate than with no gate, because the broken gate produces
reassuring output.

Worth stating plainly: this defect was introduced in the repository whose entire thesis
is that gates must refuse. Writing the thesis down is not the same as implementing it.

## The fix

Ranking and support are separate concerns and now use separate signals.

- **Ranking** still uses RRF. It is good at that.
- **Support** uses the raw cosine similarity of the best dense hit, an absolute,
  scale-stable measure. `FusedResult` now carries `dense_similarity` and
  `lexical_score` through the fusion step so the leash can see the un-fused evidence.
- A second gate, query-term coverage, is implemented and **disabled by default**. See
  below.

## Calibration

Measured on 22 queries against this corpus (85 markdown documents, 969 chunks) with
`BAAI/bge-small-en-v1.5`. Reproduce with `make calibrate`.

| Query class | n | Similarity range | Coverage range |
|---|---|---|---|
| Answerable | 12 | 0.695 to 0.895 | 0.67 to 1.00 |
| Unanswerable | 10 | 0.490 to 0.665 | 0.25 to 0.75 |

`min_semantic_similarity = 0.68` sits inside the 0.665 to 0.695 gap. At that threshold
the sample separates completely: 12 of 12 answerable admitted, 10 of 10 unanswerable
refused.

**The margin is 0.030 and the sample is 22 queries.** That is thin, and the number is a
joint property of this corpus and this embedding model. Swap either and the threshold
must be recalibrated or it becomes decorative. The fingerprint guard in `store.py`
already refuses a model mismatch at the index level for the same reason.

## Query-term coverage ships disabled, and that is a finding

The intuition was that an embedding can report topical similarity while the specific
entity asked about is simply absent, and term overlap would catch it. The measurement
does not support that on this corpus:

- Unanswerable queries reached **0.75** coverage ("what is the SOC 2 certification
  status" matches "status", "certification" against security-audit prose).
- Genuinely answerable queries fell as low as **0.67**.

No threshold separates them. A coverage gate tuned to catch the leaks would have
rejected real questions, and one loose enough to admit real questions catches nothing.
The default is `0.0`, which is off.

Shipping it enabled at a plausible-looking `0.30` would have reproduced the original
defect in a new costume: a gate that reads as a safeguard in code review and does not
discriminate in practice. It is retained as a configurable second gate because it costs
nothing and should help on corpora with denser named entities, with the explicit
instruction to calibrate before enabling.

## Consequences

- `LeashPolicy.min_support_score` is removed. There is no migration path; it never
  worked.
- `LeashDecision` gains `REFUSE_LOW_SIMILARITY` and `REFUSE_LOW_COVERAGE`, both exit
  code 2, consistent with the Spine Lite kernel contract.
- `evaluate_support()` takes an optional `query` argument, needed for coverage.
- Lexical-only hits with no dense signal now refuse rather than answer. A keyword
  coincidence is not support.
- Receipts record `dense_similarity` per result and the calibrated policy, so any
  disputed decision can be re-derived from the receipt alone.

## What this still does not do

Cosine similarity is a topical measure, not an entailment check. A chunk can be highly
similar to a question and contradict it, or discuss the right subject and not contain
the specific fact requested. This gate stops the system from answering when the corpus
is clearly silent. It does not verify that an admitted span actually entails the answer.
That is what the companion evaluation harness measures, and it is why the harness
reports a non-zero false-answer rate rather than claiming the problem is solved.
