"""The Citation Leash: no claim leaves this system without a span that supports it.

Design note
-----------
This is the pattern from Resonance, where every AI-generated feedback claim about a
mix has to trace back to a measurable DSP signal before a user is allowed to see it.
Ported here from audio to text.

The thesis is narrow and worth stating precisely: **most RAG hallucination is not a
generation problem, it is a permission problem.** The model is asked a question,
retrieval returns weak or off-topic context, and the model answers anyway from
parametric memory. The output is fluent, sourced-looking, and wrong. Prompting the
model to "only use the provided context" is a request, not a constraint, and an
adversary or an unlucky sample routes around a request.

So support is decided *before* generation is permitted, by deterministic code the
model does not participate in:

1. Retrieve.
2. Evaluate support against a threshold. This step is a pure function.
3. If support is insufficient, **refuse**. Do not generate. Do not hedge. Do not
   produce "I could not find much, but possibly...", which is a hallucination with
   a disclaimer attached.
4. If support is sufficient, generate strictly over the admitted spans, then verify
   every emitted claim still maps to one. A claim that does not is stripped, and if
   stripping empties the answer the whole response degrades to a refusal.

Refusal is a **success path**, not an error path. It has its own exit code, its own
receipt, and its own tests. This mirrors the exit-code-2 contract in the Spine Lite
kernel: a governance condition that fails produces incapacity, not a warning.

Why a threshold on the *top* score and not the mean: mean support rewards a result set
that is uniformly mediocre. One strong span is grounds to answer; ten weak ones are
not. `min_supporting_chunks` then guards the opposite failure, where a single lucky
match on a heading carries an answer it cannot actually support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

from .retrieval import FusedResult, tokenize


class LeashDecision(IntEnum):
    """Exit-code semantics shared with the Spine Lite kernel.

    ANSWER = 0 mirrors process success. REFUSE_* are 2, matching the kernel's
    exit-code-2 contract: the system is incapable of proceeding, which is different
    from having crashed (1).
    """

    ANSWER = 0
    REFUSE_NO_SUPPORT = 2
    REFUSE_LOW_SIMILARITY = 2
    REFUSE_LOW_COVERAGE = 2
    REFUSE_INSUFFICIENT_SPANS = 2
    REFUSE_EMPTY_AFTER_STRIPPING = 2


@dataclass(frozen=True)
class SupportedSpan:
    """A retrieved span admitted as evidence."""

    chunk_id: str
    locator: str
    score: float
    text: str


@dataclass(frozen=True)
class LeashVerdict:
    decision: LeashDecision
    reason: str
    admitted: tuple[SupportedSpan, ...] = ()
    top_score: float = 0.0

    @property
    def answered(self) -> bool:
        return self.decision == LeashDecision.ANSWER

    @property
    def exit_code(self) -> int:
        return int(self.decision)


@dataclass(frozen=True)
class LeashPolicy:
    """Thresholds are configuration, not constants buried in a function.

    They are the primary tuning surface, and the eval harness sweeps them to produce
    the support-threshold-versus-refusal-rate curve. A threshold you cannot sweep is
    a threshold you cannot defend.
    """

    # Raw cosine similarity of the best dense hit. The primary gate, and an
    # absolute measure rather than a relative one.
    #
    # 0.68 is MEASURED, not chosen. Calibrated on 22 queries against this corpus
    # with BAAI/bge-small-en-v1.5: 12 answerable scored 0.695 to 0.895, and 10
    # unanswerable scored 0.490 to 0.665. The threshold sits in that gap.
    # Reproduce with `make calibrate`.
    #
    # The margin is 0.030, which is THIN. This number is a property of the
    # corpus and the embedding model together, not a universal constant. Swap
    # either one and it must be recalibrated or the gate is decorative.
    min_semantic_similarity: float = 0.68
    # Fraction of the query's content words appearing in the admitted spans.
    #
    # DISABLED BY DEFAULT (0.0) because it measured as non-discriminative on this
    # corpus: unanswerable queries reached 0.75 coverage while genuinely
    # answerable ones sat as low as 0.67, so no threshold separates them. Shipping
    # it enabled would have been a gate that looks like a safeguard and is not.
    # Retained as a configurable second gate because it is cheap and should help
    # on corpora with denser named entities. Turn it on only after calibrating.
    min_query_term_coverage: float = 0.0
    min_supporting_chunks: int = 1
    max_spans: int = 5
    # Fraction of an emitted claim's content words that must appear in the admitted
    # spans for the claim to survive verification.
    claim_overlap_threshold: float = 0.55


# Words that carry no evidentiary weight. A claim matching context only on these is
# not supported by it.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the
    this to was were what when where which who why will with does do can may might
    must should would could not no yes into than then there these those they you your
    i we our us it's about over under between within without""".split()
)


def _content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS]


def query_term_coverage(query: str, spans: Sequence[SupportedSpan]) -> float:
    """Fraction of the query's content words present in the admitted spans.

    The second gate, and the one that catches the failure cosine alone misses.
    An embedding can report high topical similarity for "what is the SOC 2 audit
    schedule" against a document about security auditing generally, while the
    specific thing asked about is simply absent. Term coverage notices that the
    entity is missing. The two gates fail in different directions, which is the
    same reason retrieval runs dense and lexical together.
    """
    terms = _content_terms(query)
    if not terms:
        return 0.0
    haystack = set()
    for span in spans:
        haystack |= set(tokenize(span.text))
        haystack |= set(tokenize(span.locator))
    return sum(1 for t in terms if t in haystack) / len(terms)


def evaluate_support(
    results: Sequence[FusedResult],
    policy: LeashPolicy | None = None,
    query: str | None = None,
) -> LeashVerdict:
    """Decide whether generation is permitted. Pure function.

    Called before any generation happens. That ordering is the entire point: a check
    that runs after the model has spoken can only edit output, not withhold it.

    **This function was wrong in the first shipped version and the fix is the most
    instructive thing in this repo.** It originally gated on the RRF fused score.
    RRF encodes rank position and deliberately destroys magnitude, so the rank-1
    result scores 1/(60+1) = 0.0164 whether it is a perfect match or unrelated
    noise. Any absolute threshold below that floor admits every query that returns
    anything at all, which is every query. The gate asserted SHAPE (a score exists
    and is above a constant) rather than STATE (the retrieved text is actually
    about the question). It passed an unanswerable control query with a confident
    answer. Full write-up in docs/adr-001-rrf-is-not-a-support-signal.md.

    The gates are now, in order:
      1. Raw cosine similarity of the best dense hit, an absolute measure.
      2. Query-term coverage over the admitted spans.
    Both must pass. Ranking still uses RRF, which is what RRF is good at.
    """
    policy = policy or LeashPolicy()

    if not results:
        return LeashVerdict(
            decision=LeashDecision.REFUSE_NO_SUPPORT,
            reason="retrieval returned no candidates",
        )

    similarities = [r.dense_similarity for r in results if r.dense_similarity is not None]
    top_similarity = max(similarities) if similarities else 0.0

    if not similarities:
        # No dense signal at all means lexical-only hits. Those are keyword
        # coincidences until proven otherwise, so refuse rather than guess.
        return LeashVerdict(
            decision=LeashDecision.REFUSE_LOW_SIMILARITY,
            reason=(
                "no dense similarity signal available; lexical-only matches are not "
                "sufficient support"
            ),
            top_score=0.0,
        )

    if top_similarity < policy.min_semantic_similarity:
        return LeashVerdict(
            decision=LeashDecision.REFUSE_LOW_SIMILARITY,
            reason=(
                f"best semantic similarity {top_similarity:.4f} is below the threshold "
                f"{policy.min_semantic_similarity:.4f}; the corpus does not appear to "
                f"contain an answer to this question"
            ),
            top_score=top_similarity,
        )

    qualifying = [
        r
        for r in results
        if r.dense_similarity is not None
        and r.dense_similarity >= policy.min_semantic_similarity
    ]
    if len(qualifying) < policy.min_supporting_chunks:
        return LeashVerdict(
            decision=LeashDecision.REFUSE_INSUFFICIENT_SPANS,
            reason=(
                f"{len(qualifying)} span(s) cleared the similarity threshold but "
                f"{policy.min_supporting_chunks} are required"
            ),
            top_score=top_similarity,
        )

    candidate = tuple(
        SupportedSpan(
            chunk_id=r.chunk.chunk_id,
            locator=r.chunk.locator(),
            score=r.dense_similarity if r.dense_similarity is not None else r.fused_score,
            text=r.chunk.text,
        )
        for r in qualifying[: policy.max_spans]
    )

    if query is not None:
        coverage = query_term_coverage(query, candidate)
        if coverage < policy.min_query_term_coverage:
            return LeashVerdict(
                decision=LeashDecision.REFUSE_LOW_COVERAGE,
                reason=(
                    f"query-term coverage {coverage:.2f} is below the threshold "
                    f"{policy.min_query_term_coverage:.2f}; the retrieved spans are "
                    f"topically near the question but do not contain what it asks about"
                ),
                top_score=top_similarity,
            )

    return LeashVerdict(
        decision=LeashDecision.ANSWER,
        reason=(
            f"{len(candidate)} span(s) cleared similarity {policy.min_semantic_similarity:.2f} "
            f"(best {top_similarity:.4f})"
        ),
        admitted=candidate,
        top_score=top_similarity,
    )


@dataclass(frozen=True)
class VerifiedClaim:
    text: str
    supported: bool
    overlap: float
    supporting_locators: tuple[str, ...] = ()


def split_claims(answer: str) -> list[str]:
    """Split generated text into individually verifiable claims.

    Sentence-level, because that is the granularity at which a RAG answer actually
    goes wrong: three sourced sentences and one invented one. Verifying a whole
    answer as a unit lets the invented sentence ride along on its neighbours.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", answer.strip())
    return [p.strip() for p in parts if p.strip()]


def verify_claims(
    answer: str,
    admitted: Sequence[SupportedSpan],
    policy: LeashPolicy | None = None,
) -> list[VerifiedClaim]:
    """Check each emitted claim against the admitted spans.

    Lexical overlap is a deliberately conservative check, not a semantic one. It runs
    with no model, in microseconds, and it is the floor rather than the ceiling: the
    companion eval harness layers an LLM-as-a-judge faithfulness metric on top. The
    floor has to hold when the judge is unavailable, because a grounding guarantee
    that depends on an API being up is not a guarantee.
    """
    policy = policy or LeashPolicy()
    support_tokens = set()
    per_span: list[tuple[str, set[str]]] = []
    for span in admitted:
        toks = set(tokenize(span.text))
        per_span.append((span.locator, toks))
        support_tokens |= toks

    verified: list[VerifiedClaim] = []
    for claim in split_claims(answer):
        content = [t for t in tokenize(claim) if t not in _STOPWORDS]
        if not content:
            continue
        hits = sum(1 for t in content if t in support_tokens)
        overlap = hits / len(content)
        supporters = tuple(
            loc
            for loc, toks in per_span
            if sum(1 for t in content if t in toks) / len(content)
            >= policy.claim_overlap_threshold
        )
        verified.append(
            VerifiedClaim(
                text=claim,
                supported=overlap >= policy.claim_overlap_threshold,
                overlap=overlap,
                supporting_locators=supporters,
            )
        )
    return verified


@dataclass
class LeashedAnswer:
    """Final response. Either an answer bound to spans, or an explicit refusal."""

    verdict: LeashVerdict
    answer: str = ""
    claims: list[VerifiedClaim] = field(default_factory=list)
    stripped: list[VerifiedClaim] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return not self.verdict.answered

    @property
    def exit_code(self) -> int:
        return self.verdict.exit_code

    def citations(self) -> list[str]:
        seen: list[str] = []
        for claim in self.claims:
            for loc in claim.supporting_locators:
                if loc not in seen:
                    seen.append(loc)
        return seen
