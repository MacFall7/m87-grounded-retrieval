"""Receipts, the extractive generator, and the answer wrapper.

Everything here is exercised without a database. The pieces that need one live in
`test_store_integration.py`.
"""

from __future__ import annotations

import json

from conftest import make_chunk

from grounded_retrieval.leash import (
    LeashDecision,
    LeashedAnswer,
    LeashVerdict,
    SupportedSpan,
    VerifiedClaim,
)
from grounded_retrieval.service import QueryReceipt, extractive_generator


def _span(text: str, locator: str = "demo:README.md:L1-L4") -> SupportedSpan:
    return SupportedSpan(chunk_id="aaa", locator=locator, score=0.9, text=text)


def test_extractive_generator_returns_nothing_without_spans():
    assert extractive_generator("anything", []) == ""


def test_extractive_generator_strips_heading_lines():
    """A heading is a label, not a claim. Emitting it as prose fabricates a sentence."""
    out = extractive_generator(
        "how does it refuse",
        [_span("## Refusal semantics\nThe kernel exits with code 2.")],
    )
    assert "##" not in out
    assert "The kernel exits with code 2." in out


def test_extractive_generator_is_bounded():
    spans = [_span(f"Sentence number {i} from a span.") for i in range(9)]
    out = extractive_generator("q", spans)
    assert "Sentence number 3" not in out


def test_extractive_generator_is_deterministic():
    spans = [_span("The kernel refuses.")]
    assert extractive_generator("q", spans) == extractive_generator("q", spans)


def _receipt(**kw) -> QueryReceipt:
    base = dict(
        query="how does the kernel refuse",
        index_name="default",
        model_fingerprint="hashing-stub@256",
        decision="ANSWER",
        exit_code=0,
        reason="1 span(s) cleared the support threshold",
        top_score=0.0328,
    )
    base.update(kw)
    return QueryReceipt(**base)


def test_receipt_digest_is_stable():
    assert _receipt().digest() == _receipt().digest()


def test_receipt_digest_changes_with_any_field():
    """A receipt whose digest ignores a field cannot be used to detect tampering."""
    assert _receipt().digest() != _receipt(query="something else").digest()
    assert _receipt().digest() != _receipt(exit_code=2).digest()
    assert _receipt().digest() != _receipt(decision="REFUSE_NO_SUPPORT").digest()


def test_receipt_json_is_parseable_and_carries_its_digest():
    body = json.loads(_receipt().to_json())
    assert body["receipt_digest"] == _receipt().digest()
    assert body["query"] == "how does the kernel refuse"


def test_receipt_records_the_model_fingerprint():
    """Without the fingerprint a receipt cannot be re-derived, only believed."""
    body = json.loads(_receipt().to_json())
    assert body["model_fingerprint"] == "hashing-stub@256"


def test_refused_answer_reports_refusal_and_exit_code_two():
    verdict = LeashVerdict(
        decision=LeashDecision.REFUSE_NO_SUPPORT, reason="no candidates"
    )
    answer = LeashedAnswer(verdict=verdict)
    assert answer.refused
    assert answer.exit_code == 2
    assert answer.answer == ""
    assert answer.citations() == []


def test_answered_response_exposes_deduplicated_citations():
    verdict = LeashVerdict(decision=LeashDecision.ANSWER, reason="ok")
    claims = [
        VerifiedClaim("a", True, 1.0, ("demo:README.md:L1-L4",)),
        VerifiedClaim("b", True, 1.0, ("demo:README.md:L1-L4", "demo:OTHER.md:L2-L3")),
    ]
    answer = LeashedAnswer(verdict=verdict, answer="a b", claims=claims)
    assert not answer.refused
    assert answer.exit_code == 0
    assert answer.citations() == ["demo:README.md:L1-L4", "demo:OTHER.md:L2-L3"]


def test_stored_chunk_locator_matches_the_chunk_locator_format():
    """Ingest-time and query-time locators must agree or citations stop resolving."""
    stored = make_chunk("aaa", "text", repo="r", path="d/f.md", line_start=3, line_end=9)
    assert stored.locator() == "r:d/f.md:L3-L9"
