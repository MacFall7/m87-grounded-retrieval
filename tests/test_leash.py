"""The Citation Leash: refusal paths and claim verification.

Written against the recalibrated leash API. The decision is made on raw semantic
similarity and query-term coverage, not on the fused RRF score, because the fused
score is a rank artefact: it tells you a chunk won the fusion, not that it is about
the query. A corpus with no answer still produces a top-ranked chunk.
"""

from __future__ import annotations

import pytest
from conftest import evaluate, make_chunk, make_fused

from grounded_retrieval.leash import (
    LeashDecision,
    LeashPolicy,
    SupportedSpan,
    split_claims,
    verify_claims,
)

REFUSAL_EXIT_CODE = 2

SPAN_TEXT = (
    "The kernel refuses when a policy denies the request and exits with code 2. "
    "Refusal is a first class outcome with its own receipt."
)


def _span(locator: str = "demo:README.md:L1-L4", text: str = SPAN_TEXT) -> SupportedSpan:
    return SupportedSpan(chunk_id="aaa", locator=locator, score=0.9, text=text)


# ---------------------------------------------------------------- policy defaults


def test_policy_defaults_are_the_calibrated_values():
    p = LeashPolicy()
    # 0.68 is measured, not chosen: answerable queries scored 0.695 to 0.895 and
    # unanswerable 0.490 to 0.665 on this corpus. See docs/adr-001.
    assert p.min_semantic_similarity == pytest.approx(0.68)
    # Coverage ships DISABLED because it measured as non-discriminative here.
    # Unanswerable queries reached 0.75 coverage, answerable fell to 0.67.
    assert p.min_query_term_coverage == pytest.approx(0.0)
    assert p.min_supporting_chunks == 1
    assert p.max_spans == 5
    assert p.claim_overlap_threshold == pytest.approx(0.55)


def test_policy_is_immutable():
    """Thresholds are configuration, and configuration that mutates mid-query is a bug."""
    p = LeashPolicy()
    with pytest.raises(Exception):
        p.min_semantic_similarity = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------- refusal paths


def test_empty_retrieval_refuses():
    verdict = evaluate([], LeashPolicy(), query="anything at all")
    assert not verdict.answered
    assert verdict.decision == LeashDecision.REFUSE_NO_SUPPORT
    assert verdict.exit_code == REFUSAL_EXIT_CODE
    assert verdict.admitted == ()


def test_low_semantic_similarity_refuses():
    chunk = make_chunk("aaa", SPAN_TEXT)
    results = [make_fused(chunk, dense_similarity=0.31, lexical_score=5.0)]
    verdict = evaluate(results, LeashPolicy(), query="how does the kernel refuse")
    assert not verdict.answered
    assert verdict.exit_code == REFUSAL_EXIT_CODE
    # REFUSE_* members share the value 2 by design, so the decision alone cannot
    # distinguish them. The reason string is what a receipt reader actually reads.
    assert "similar" in verdict.reason.lower()


def test_similarity_at_the_threshold_is_admitted():
    """Boundary is inclusive, so a policy set to an observed score does not refuse it."""
    chunk = make_chunk("aaa", SPAN_TEXT)
    results = [make_fused(chunk, dense_similarity=0.68, lexical_score=5.0)]
    verdict = evaluate(
        results, LeashPolicy(), query="kernel refuses policy denies request code"
    )
    assert verdict.answered


def test_high_fused_score_does_not_rescue_low_similarity():
    """The recalibration in one assertion: fused rank is not evidence of aboutness."""
    chunk = make_chunk("aaa", SPAN_TEXT)
    results = [make_fused(chunk, fused_score=99.0, dense_similarity=0.05)]
    verdict = evaluate(results, LeashPolicy(), query="how does the kernel refuse")
    assert not verdict.answered


def test_low_query_term_coverage_refuses():
    """Semantically close but sharing none of the query's terms is not support."""
    chunk = make_chunk(
        "aaa",
        "Colour grading presets are stored alongside the timeline metadata.",
    )
    results = [make_fused(chunk, dense_similarity=0.92, lexical_score=0.0)]
    # Coverage is off by default, so this test must opt in explicitly. That is
    # the point of the test: the gate works when enabled, and shipping it off is
    # a calibration decision rather than a missing feature.
    verdict = evaluate(
        results,
        LeashPolicy(min_query_term_coverage=0.5),
        query="pgvector hnsw index dimensions fingerprint mismatch",
    )
    assert not verdict.answered
    assert verdict.exit_code == REFUSAL_EXIT_CODE
    assert "coverage" in verdict.reason.lower()


def test_insufficient_supporting_chunks_refuses():
    chunk = make_chunk("aaa", SPAN_TEXT)
    results = [make_fused(chunk, dense_similarity=0.90)]
    policy = LeashPolicy(min_supporting_chunks=3)
    verdict = evaluate(results, policy, query="kernel refuses policy denies request")
    assert not verdict.answered
    assert verdict.decision == LeashDecision.REFUSE_INSUFFICIENT_SPANS
    assert verdict.exit_code == REFUSAL_EXIT_CODE


def test_every_refusal_carries_exit_code_two():
    """Mirrors the Spine Lite kernel contract: incapacity is 2, a crash is 1."""
    for member in (
        LeashDecision.REFUSE_NO_SUPPORT,
        LeashDecision.REFUSE_INSUFFICIENT_SPANS,
        LeashDecision.REFUSE_EMPTY_AFTER_STRIPPING,
        LeashDecision.REFUSE_LOW_SIMILARITY,
        LeashDecision.REFUSE_LOW_COVERAGE,
    ):
        assert int(member) == REFUSAL_EXIT_CODE
    assert int(LeashDecision.ANSWER) == 0


def test_refusal_reason_is_specific_enough_to_act_on():
    verdict = evaluate([], LeashPolicy(), query="anything")
    assert verdict.reason
    assert len(verdict.reason) > 10


# ---------------------------------------------------------------- answer path


def test_sufficient_support_admits_spans():
    chunks = [make_chunk(f"c{i}", SPAN_TEXT) for i in range(3)]
    results = [make_fused(c, dense_similarity=0.80, rank=i + 1) for i, c in enumerate(chunks)]
    verdict = evaluate(
        results, LeashPolicy(), query="kernel refuses policy denies request code"
    )
    assert verdict.answered
    assert verdict.decision == LeashDecision.ANSWER
    assert verdict.exit_code == 0
    assert len(verdict.admitted) == 3


def test_admitted_spans_are_capped_by_max_spans():
    chunks = [make_chunk(f"c{i}", SPAN_TEXT) for i in range(9)]
    results = [make_fused(c, dense_similarity=0.80, rank=i + 1) for i, c in enumerate(chunks)]
    verdict = evaluate(
        results, LeashPolicy(max_spans=2), query="kernel refuses policy denies request"
    )
    assert verdict.answered
    assert len(verdict.admitted) == 2


def test_admitted_spans_carry_a_checkable_locator():
    chunk = make_chunk("aaa", SPAN_TEXT, repo="M87-Spine-lite", path="docs/ARCHITECTURE.md",
                       line_start=40, line_end=58)
    results = [make_fused(chunk, dense_similarity=0.80)]
    verdict = evaluate(
        results, LeashPolicy(), query="kernel refuses policy denies request code"
    )
    assert verdict.answered
    assert verdict.admitted[0].locator == "M87-Spine-lite:docs/ARCHITECTURE.md:L40-L58"


def test_spans_below_the_similarity_floor_are_not_admitted():
    strong = make_chunk("strong", SPAN_TEXT)
    weak = make_chunk("weak", SPAN_TEXT)
    results = [
        make_fused(strong, dense_similarity=0.80, rank=1),
        make_fused(weak, dense_similarity=0.10, rank=2),
    ]
    verdict = evaluate(
        results, LeashPolicy(), query="kernel refuses policy denies request code"
    )
    assert verdict.answered
    assert [s.chunk_id for s in verdict.admitted] == ["strong"]


# ---------------------------------------------------------------- claim splitting


def test_split_claims_splits_on_sentence_boundaries():
    claims = split_claims("The kernel refuses. It exits with code 2. Hooks only observe.")
    assert claims == [
        "The kernel refuses.",
        "It exits with code 2.",
        "Hooks only observe.",
    ]


def test_split_claims_ignores_empty_input():
    assert split_claims("   ") == []


def test_split_claims_keeps_a_single_sentence_whole():
    assert split_claims("One sentence only") == ["One sentence only"]


# ---------------------------------------------------------------- verification


def test_supported_claim_survives_verification():
    claims = verify_claims("The kernel refuses when a policy denies the request.", [_span()])
    assert len(claims) == 1
    assert claims[0].supported
    assert claims[0].overlap == pytest.approx(1.0)
    assert claims[0].supporting_locators == ("demo:README.md:L1-L4",)


def test_unsupported_claim_is_marked_unsupported():
    claims = verify_claims(
        "Latency improved by forty percent after sharding the write path.", [_span()]
    )
    assert len(claims) == 1
    assert not claims[0].supported
    assert claims[0].supporting_locators == ()


def test_mixed_answer_separates_supported_from_invented():
    """The realistic failure: three sourced sentences carrying one invented one."""
    answer = (
        "The kernel refuses when a policy denies the request. "
        "Refusal exits with code 2. "
        "Benchmarks show a ninety nine percent reduction in customer escalations."
    )
    claims = verify_claims(answer, [_span()])
    supported = [c for c in claims if c.supported]
    stripped = [c for c in claims if not c.supported]
    assert len(supported) == 2
    assert len(stripped) == 1
    assert "escalations" in stripped[0].text


def test_stopwords_do_not_create_support():
    """A claim that matches only on function words matches on nothing."""
    claims = verify_claims("What about the ones over there?", [_span()])
    assert all(not c.supported for c in claims)


def test_verification_threshold_is_configurable():
    answer = "The kernel refuses when latency spikes."
    lenient = verify_claims(answer, [_span()], LeashPolicy(claim_overlap_threshold=0.3))
    strict = verify_claims(answer, [_span()], LeashPolicy(claim_overlap_threshold=0.99))
    assert lenient[0].supported
    assert not strict[0].supported


def test_verification_against_no_spans_supports_nothing():
    claims = verify_claims("The kernel refuses when a policy denies the request.", [])
    assert claims and all(not c.supported for c in claims)


def test_verification_is_deterministic():
    answer = "The kernel refuses when a policy denies the request. Hooks only observe."
    first = verify_claims(answer, [_span()])
    second = verify_claims(answer, [_span()])
    assert [(c.text, c.supported, c.overlap) for c in first] == [
        (c.text, c.supported, c.overlap) for c in second
    ]


def test_supporting_locators_name_the_span_that_carried_the_claim():
    other = SupportedSpan(
        chunk_id="bbb",
        locator="demo:OTHER.md:L9-L12",
        score=0.5,
        text="Colour grading presets live beside the timeline metadata.",
    )
    claims = verify_claims(
        "The kernel refuses when a policy denies the request.", [_span(), other]
    )
    assert claims[0].supporting_locators == ("demo:README.md:L1-L4",)
