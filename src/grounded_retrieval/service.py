"""Query service: retrieve, decide, generate under leash, emit a receipt."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from .embedding import EmbeddingBackend
from .leash import (
    LeashDecision,
    LeashedAnswer,
    LeashPolicy,
    SupportedSpan,
    evaluate_support,
    verify_claims,
)
from .retrieval import BM25Index, FusedResult, reciprocal_rank_fusion
from .store import ChunkStore


@dataclass
class QueryReceipt:
    """Offline-verifiable record of one query.

    Same pattern as the Audit Agent receipt chain: enough state that a third party
    can reconstruct why the system did what it did, without trusting the narrative.
    A receipt that only records the output is a log line. This records the inputs,
    the fingerprints, the intermediate rankings, and the decision, so a disputed
    answer can be re-derived.
    """

    query: str
    index_name: str
    model_fingerprint: str
    decision: str
    exit_code: int
    reason: str
    top_score: float
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    admitted_locators: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    stripped_claims: list[str] = field(default_factory=list)
    policy: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        body = asdict(self)
        body["receipt_digest"] = self.digest()
        return json.dumps(body, indent=2, sort_keys=True)


Generator = Callable[[str, Sequence[SupportedSpan]], str]


def extractive_generator(query: str, spans: Sequence[SupportedSpan]) -> str:
    """Default generator: extractive, no LLM, no API key.

    Chosen so the repo's grounding guarantee can be demonstrated and tested end to
    end with nothing installed but Python and Postgres. It is also the honest
    baseline: an extractive answer cannot hallucinate, so any faithfulness failure
    the eval harness reports against it is a *retrieval* failure, which isolates the
    two error sources instead of confounding them.

    Swap in an LLM via the `generator` argument to `GroundedRetriever.query`. The
    leash applies identically either way, which is the point: the guarantee lives in
    the harness, not in the model's cooperation.
    """
    if not spans:
        return ""
    lines = []
    for span in spans[:3]:
        first = span.text.strip().split("\n")
        body = " ".join(x for x in first if x.strip() and not x.strip().startswith("#"))
        if body:
            lines.append(body.strip())
    return " ".join(lines)


class GroundedRetriever:
    def __init__(
        self,
        store: ChunkStore,
        backend: EmbeddingBackend,
        *,
        policy: LeashPolicy | None = None,
        dense_k: int = 20,
        lexical_k: int = 20,
        limit: int = 10,
    ) -> None:
        self.store = store
        self.backend = backend
        self.policy = policy or LeashPolicy()
        self.dense_k = dense_k
        self.lexical_k = lexical_k
        self.limit = limit
        # Fail fast and loudly if the index was built by a different model.
        self.store.assert_compatible(backend.fingerprint())
        self._bm25 = BM25Index(self.store.all_chunks())

    def retrieve(self, query: str) -> list[FusedResult]:
        dense = self.store.dense_search(self.backend.embed_query(query), self.dense_k)
        lexical = self._bm25.search(query, self.lexical_k)
        return reciprocal_rank_fusion(dense, lexical, limit=self.limit)

    def query(
        self, query: str, *, generator: Generator | None = None
    ) -> tuple[LeashedAnswer, QueryReceipt]:
        generate = generator or extractive_generator
        results = self.retrieve(query)
        verdict = evaluate_support(results, self.policy, query=query)

        receipt = QueryReceipt(
            query=query,
            index_name=self.store.index_name,
            model_fingerprint=self.backend.fingerprint(),
            decision=verdict.decision.name,
            exit_code=verdict.exit_code,
            reason=verdict.reason,
            top_score=round(verdict.top_score, 6),
            retrieved=[
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk.chunk_id,
                    "locator": r.chunk.locator(),
                    "fused_score": round(r.fused_score, 6),
                    "dense_similarity": (
                        round(r.dense_similarity, 6) if r.dense_similarity is not None else None
                    ),
                    "dense_rank": r.dense_rank,
                    "lexical_rank": r.lexical_rank,
                }
                for r in results
            ],
            policy={
                "min_semantic_similarity": self.policy.min_semantic_similarity,
                "min_query_term_coverage": self.policy.min_query_term_coverage,
                "min_supporting_chunks": self.policy.min_supporting_chunks,
                "max_spans": self.policy.max_spans,
                "claim_overlap_threshold": self.policy.claim_overlap_threshold,
            },
        )

        # The refusal path returns BEFORE the generator is ever called. Not an
        # optimization: if generation runs and its output is filtered afterwards,
        # the guarantee is a filter, and filters leak. Here the model is never
        # given the opportunity to speak without evidence.
        if not verdict.answered:
            return LeashedAnswer(verdict=verdict), receipt

        raw = generate(query, verdict.admitted)
        claims = verify_claims(raw, verdict.admitted, self.policy)
        supported = [c for c in claims if c.supported]
        stripped = [c for c in claims if not c.supported]

        if not supported:
            # Generation produced nothing traceable to an admitted span. Degrade to
            # a refusal rather than shipping the unsupported text.
            from .leash import LeashVerdict

            downgraded = LeashVerdict(
                decision=LeashDecision.REFUSE_EMPTY_AFTER_STRIPPING,
                reason=(
                    f"all {len(claims)} generated claim(s) failed span verification; "
                    "refusing rather than emitting unsupported text"
                ),
                admitted=verdict.admitted,
                top_score=verdict.top_score,
            )
            receipt.decision = downgraded.decision.name
            receipt.exit_code = downgraded.exit_code
            receipt.reason = downgraded.reason
            receipt.stripped_claims = [c.text for c in stripped]
            return LeashedAnswer(verdict=downgraded, stripped=stripped), receipt

        answer = " ".join(c.text for c in supported)
        result = LeashedAnswer(
            verdict=verdict, answer=answer, claims=supported, stripped=stripped
        )
        receipt.admitted_locators = [s.locator for s in verdict.admitted]
        receipt.claims = [
            {
                "text": c.text,
                "overlap": round(c.overlap, 4),
                "supporting_locators": list(c.supporting_locators),
            }
            for c in supported
        ]
        receipt.stripped_claims = [c.text for c in stripped]
        return result, receipt
