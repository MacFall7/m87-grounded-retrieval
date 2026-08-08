"""BM25 and Reciprocal Rank Fusion, both exercised with no model and no database."""

from __future__ import annotations

import pytest
from conftest import make_chunk, make_scored

from grounded_retrieval.retrieval import (
    RRF_K,
    BM25Index,
    reciprocal_rank_fusion,
    tokenize,
)

CORPUS = [
    make_chunk(
        "aaa",
        "The kernel exits with exit code 2 when a policy denies the request.",
        heading=("Spine Lite", "Refusal semantics"),
    ),
    make_chunk(
        "bbb",
        "Vectors are stored with vector_cosine_ops so intent survives a backend swap.",
        heading=("Store", "Index"),
    ),
    make_chunk(
        "ccc",
        "Hooks observe and never decide. A hook that vetoes is a policy in disguise.",
        heading=("Spine Lite", "Hooks"),
    ),
    make_chunk(
        "ddd",
        "A governed_request carries the policy decision and the receipt digest.",
        heading=("Contracts",),
    ),
]


def test_tokenize_lowercases_and_keeps_underscores():
    assert tokenize("Vector_Cosine_Ops and EXIT code 2") == [
        "vector_cosine_ops",
        "and",
        "exit",
        "code",
        "2",
    ]


def test_tokenize_drops_punctuation():
    assert tokenize("fail-closed, not fail-open.") == ["fail", "closed", "not", "fail", "open"]


def test_bm25_ranks_the_lexically_matching_chunk_first():
    index = BM25Index(CORPUS)
    hits = index.search("vector_cosine_ops", k=5)
    assert hits
    assert hits[0].chunk.chunk_id == "bbb"


def test_bm25_matches_identifiers_verbatim():
    """The whole reason for a lexical leg: an embedding cannot know `governed_request`."""
    index = BM25Index(CORPUS)
    hits = index.search("governed_request", k=5)
    assert [h.chunk.chunk_id for h in hits] == ["ddd"]


def test_bm25_indexes_the_heading_path():
    index = BM25Index(CORPUS)
    hits = index.search("Contracts", k=5)
    assert hits and hits[0].chunk.chunk_id == "ddd"


def test_bm25_ranks_are_dense_and_one_indexed():
    index = BM25Index(CORPUS)
    hits = index.search("policy", k=5)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_bm25_scores_are_descending():
    index = BM25Index(CORPUS)
    hits = index.search("policy decision receipt", k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_bm25_respects_k():
    index = BM25Index(CORPUS)
    assert len(index.search("policy", k=1)) == 1


def test_bm25_returns_nothing_for_an_out_of_vocabulary_query():
    index = BM25Index(CORPUS)
    assert index.search("zzzz_nonexistent_token", k=5) == []


def test_bm25_handles_an_empty_query():
    assert BM25Index(CORPUS).search("", k=5) == []


def test_bm25_handles_an_empty_corpus():
    assert BM25Index([]).search("policy", k=5) == []
    assert BM25Index([]).size == 0


def test_bm25_is_deterministic():
    a = [h.chunk.chunk_id for h in BM25Index(CORPUS).search("policy", k=5)]
    b = [h.chunk.chunk_id for h in BM25Index(CORPUS).search("policy", k=5)]
    assert a == b


def test_bm25_rewards_repeated_terms_via_saturation():
    """Term frequency must help, but sub-linearly. Ten repeats is not ten times better."""
    once = make_chunk("one", "policy")
    many = make_chunk("many", "policy " * 10)
    index = BM25Index([once, many])
    hits = {h.chunk.chunk_id: h.score for h in index.search("policy", k=2)}
    assert hits["many"] > hits["one"]
    assert hits["many"] < 10 * hits["one"]


def test_rrf_is_a_pure_function_of_the_rank_lists():
    dense = [make_scored(CORPUS[0], 0.9, 1), make_scored(CORPUS[1], 0.7, 2)]
    lexical = [make_scored(CORPUS[1], 4.0, 1), make_scored(CORPUS[2], 2.0, 2)]

    first = reciprocal_rank_fusion(dense, lexical)
    second = reciprocal_rank_fusion(dense, lexical)
    assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]
    assert [r.fused_score for r in first] == [r.fused_score for r in second]


def test_rrf_score_matches_the_formula():
    dense = [make_scored(CORPUS[0], 0.9, 1)]
    lexical = [make_scored(CORPUS[0], 4.0, 3)]
    fused = reciprocal_rank_fusion(dense, lexical)
    expected = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 3)
    assert fused[0].fused_score == pytest.approx(expected)


def test_rrf_rewards_agreement_between_retrievers():
    both = CORPUS[0]
    dense_only = CORPUS[1]
    dense = [make_scored(both, 0.9, 1), make_scored(dense_only, 0.8, 2)]
    lexical = [make_scored(both, 5.0, 1)]
    fused = reciprocal_rank_fusion(dense, lexical)
    assert fused[0].chunk.chunk_id == both.chunk_id
    assert fused[0].found_by_both is True
    assert fused[1].found_by_both is False


def test_rrf_ignores_score_magnitudes():
    """BM25 magnitudes move with corpus statistics, so only ranks may influence fusion."""
    dense = [make_scored(CORPUS[0], 0.9, 1), make_scored(CORPUS[1], 0.1, 2)]
    inflated = [make_scored(CORPUS[0], 900.0, 1), make_scored(CORPUS[1], 0.0001, 2)]
    a = reciprocal_rank_fusion(dense, [])
    b = reciprocal_rank_fusion(inflated, [])
    assert [r.fused_score for r in a] == [r.fused_score for r in b]


def test_rrf_tie_break_is_deterministic_on_chunk_id():
    """Equal contributions must resolve to a total order, or baselines cannot regress."""
    x = make_chunk("zzz", "same rank in each list")
    y = make_chunk("aaa2", "same rank in each list")
    dense = [make_scored(x, 0.5, 1), make_scored(y, 0.5, 2)]
    lexical = [make_scored(y, 1.0, 1), make_scored(x, 1.0, 2)]

    fused = reciprocal_rank_fusion(dense, lexical)
    assert fused[0].fused_score == pytest.approx(fused[1].fused_score)
    assert [r.chunk.chunk_id for r in fused] == ["aaa2", "zzz"]


def test_rrf_tie_break_is_independent_of_input_order():
    x = make_chunk("zzz", "tie")
    y = make_chunk("aaa2", "tie")
    forward = reciprocal_rank_fusion(
        [make_scored(x, 0.5, 1), make_scored(y, 0.5, 2)],
        [make_scored(y, 1.0, 1), make_scored(x, 1.0, 2)],
    )
    backward = reciprocal_rank_fusion(
        [make_scored(y, 0.5, 2), make_scored(x, 0.5, 1)],
        [make_scored(x, 1.0, 2), make_scored(y, 1.0, 1)],
    )
    assert [r.chunk.chunk_id for r in forward] == [r.chunk.chunk_id for r in backward]


def test_rrf_ranks_are_dense_and_one_indexed():
    dense = [make_scored(c, 1.0, i + 1) for i, c in enumerate(CORPUS)]
    fused = reciprocal_rank_fusion(dense, [])
    assert [r.rank for r in fused] == list(range(1, len(CORPUS) + 1))


def test_rrf_respects_the_limit():
    dense = [make_scored(c, 1.0, i + 1) for i, c in enumerate(CORPUS)]
    assert len(reciprocal_rank_fusion(dense, [], limit=2)) == 2


def test_rrf_weights_shift_the_ordering():
    dense_only = CORPUS[0]
    lexical_only = CORPUS[1]
    dense = [make_scored(dense_only, 0.9, 1)]
    lexical = [make_scored(lexical_only, 5.0, 1)]

    lexical_heavy = reciprocal_rank_fusion(
        dense, lexical, dense_weight=1.0, lexical_weight=4.0
    )
    dense_heavy = reciprocal_rank_fusion(
        dense, lexical, dense_weight=4.0, lexical_weight=1.0
    )
    assert lexical_heavy[0].chunk.chunk_id == lexical_only.chunk_id
    assert dense_heavy[0].chunk.chunk_id == dense_only.chunk_id


def test_rrf_on_two_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_records_which_leg_found_each_chunk():
    dense = [make_scored(CORPUS[0], 0.9, 1)]
    lexical = [make_scored(CORPUS[1], 5.0, 1)]
    fused = {r.chunk.chunk_id: r for r in reciprocal_rank_fusion(dense, lexical)}
    assert fused["aaa"].dense_rank == 1 and fused["aaa"].lexical_rank is None
    assert fused["bbb"].lexical_rank == 1 and fused["bbb"].dense_rank is None


def test_rrf_carries_raw_signals_through_for_the_leash():
    """Target API: the leash decides on raw similarity, so fusion must not discard it.

    RRF deliberately throws away magnitude for *ranking*. The magnitudes still have to
    reach the leash, which is a different decision on a different scale.
    """
    dense = [make_scored(CORPUS[0], 0.83, 1)]
    lexical = [make_scored(CORPUS[0], 6.25, 1)]
    fused = reciprocal_rank_fusion(dense, lexical)[0]
    assert fused.dense_similarity == pytest.approx(0.83)
    assert fused.lexical_score == pytest.approx(6.25)


def test_rrf_leaves_missing_raw_signals_as_none():
    dense = [make_scored(CORPUS[0], 0.83, 1)]
    fused = reciprocal_rank_fusion(dense, [])[0]
    assert fused.dense_similarity == pytest.approx(0.83)
    assert fused.lexical_score is None
