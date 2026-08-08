"""Grounded retrieval with the Citation Leash: no claim without a supporting span."""

from .chunking import Chunk, chunk_markdown
from .embedding import HashingBackend, SentenceTransformerBackend
from .leash import LeashDecision, LeashedAnswer, LeashPolicy, evaluate_support, verify_claims
from .retrieval import BM25Index, reciprocal_rank_fusion
from .service import GroundedRetriever, QueryReceipt
from .store import ChunkStore, IndexFingerprintMismatch

__version__ = "0.1.0"
__all__ = [
    "Chunk", "chunk_markdown", "ChunkStore", "IndexFingerprintMismatch",
    "SentenceTransformerBackend", "HashingBackend", "BM25Index",
    "reciprocal_rank_fusion", "LeashPolicy", "LeashDecision", "LeashedAnswer",
    "evaluate_support", "verify_claims", "GroundedRetriever", "QueryReceipt",
]
