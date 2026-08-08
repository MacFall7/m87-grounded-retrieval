"""Corpus ingestion: walk markdown, chunk, embed, upsert, build the ANN index."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import dataclass

from .chunking import Chunk, chunk_markdown
from .embedding import HashingBackend, SentenceTransformerBackend
from .store import DEFAULT_DSN, ChunkStore


@dataclass
class IngestReport:
    documents: int
    chunks: int
    seconds: float
    model_fingerprint: str
    dimensions: int

    def render(self) -> str:
        rate = self.chunks / self.seconds if self.seconds else 0.0
        return (
            f"ingested {self.chunks} chunks from {self.documents} documents "
            f"in {self.seconds:.1f}s ({rate:.0f} chunks/s)\n"
            f"model: {self.model_fingerprint}  dimensions: {self.dimensions}"
        )


def collect_chunks(corpus_root: pathlib.Path) -> tuple[list[Chunk], int]:
    """Walk `corpus/<repo>/<path...>.md` and chunk each document.

    The directory layout carries provenance: the first path segment is the source
    repo, the rest is the path within it. That keeps provenance derivable from the
    filesystem rather than a manifest that can drift out of sync with the files.
    """
    chunks: list[Chunk] = []
    docs = 0
    for path in sorted(corpus_root.rglob("*.md")):
        rel = path.relative_to(corpus_root)
        if len(rel.parts) < 2:
            continue
        repo = rel.parts[0]
        source_path = str(pathlib.PurePosixPath(*rel.parts[1:]))
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_markdown(text, source_repo=repo, source_path=source_path))
        docs += 1
    return chunks, docs


def run_ingest(
    corpus_root: pathlib.Path,
    *,
    dsn: str = DEFAULT_DSN,
    index_name: str = "default",
    model: str | None = None,
    use_stub: bool = False,
) -> IngestReport:
    started = time.time()
    backend = HashingBackend() if use_stub else SentenceTransformerBackend(
        model or "BAAI/bge-small-en-v1.5"
    )

    chunks, docs = collect_chunks(corpus_root)
    if not chunks:
        raise SystemExit(f"no markdown found under {corpus_root}")

    store = ChunkStore(dsn=dsn, index_name=index_name)
    store.initialize(
        model_fingerprint=backend.fingerprint(), dimensions=backend.dimensions
    )
    store.clear()

    vectors = backend.embed_documents([c.embedding_text() for c in chunks])
    store.upsert(chunks, vectors)
    store.build_ann_index(backend.dimensions)

    return IngestReport(
        documents=docs,
        chunks=len(chunks),
        seconds=time.time() - started,
        model_fingerprint=backend.fingerprint(),
        dimensions=backend.dimensions,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a markdown corpus into pgvector.")
    parser.add_argument("--corpus", default="corpus", type=pathlib.Path)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--index", default="default")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--stub-embeddings",
        action="store_true",
        help="use the deterministic hashing backend (CI only; retrieval quality "
        "numbers from this backend are meaningless and must be labeled as such)",
    )
    args = parser.parse_args(argv)

    report = run_ingest(
        args.corpus,
        dsn=args.dsn,
        index_name=args.index,
        model=args.model,
        use_stub=args.stub_embeddings,
    )
    print(report.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
