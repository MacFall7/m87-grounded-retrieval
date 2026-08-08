"""Corpus walking. Pure filesystem work, no database and no model."""

from __future__ import annotations

import pathlib

from grounded_retrieval.ingest import collect_chunks


def _write(root: pathlib.Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


BODY = (
    "# Title\n\n"
    "A paragraph long enough to survive the minimum chunk size, describing how the "
    "kernel refuses rather than degrading when a policy denies a request.\n"
)


def test_provenance_is_derived_from_the_directory_layout(tmp_path):
    """Provenance comes from the filesystem so it cannot drift out of sync with it."""
    _write(tmp_path, "repo-one/docs/ARCHITECTURE.md", BODY)
    chunks, docs = collect_chunks(tmp_path)
    assert docs == 1
    assert chunks[0].source_repo == "repo-one"
    assert chunks[0].source_path == "docs/ARCHITECTURE.md"


def test_top_level_markdown_outside_a_repo_directory_is_skipped():
    """A file with no repo segment has no provenance, so it must not enter the index."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _write(root, "stray.md", BODY)
        chunks, docs = collect_chunks(root)
        assert (chunks, docs) == ([], 0)


def test_non_markdown_is_ignored(tmp_path):
    _write(tmp_path, "repo-one/notes.txt", BODY)
    _write(tmp_path, "repo-one/README.md", BODY)
    _, docs = collect_chunks(tmp_path)
    assert docs == 1


def test_collection_is_deterministic(tmp_path):
    for name in ("b", "a", "c"):
        _write(tmp_path, f"repo-{name}/README.md", BODY)
    first = [(c.source_repo, c.chunk_id) for c in collect_chunks(tmp_path)[0]]
    second = [(c.source_repo, c.chunk_id) for c in collect_chunks(tmp_path)[0]]
    assert first == second
    # Sorted walk order, so two machines produce the same ordinals for the same corpus.
    assert [repo for repo, _ in first] == ["repo-a", "repo-b", "repo-c"]


def test_reingesting_unchanged_files_produces_identical_ids(tmp_path):
    """Idempotence: an unchanged corpus must upsert onto the same rows."""
    _write(tmp_path, "repo-one/README.md", BODY)
    before = {c.chunk_id for c in collect_chunks(tmp_path)[0]}
    _write(tmp_path, "repo-one/README.md", BODY)
    after = {c.chunk_id for c in collect_chunks(tmp_path)[0]}
    assert before == after


def test_the_committed_corpus_chunks(corpus_root):
    """Guards against a chunking change that silently empties the index."""
    if not corpus_root.exists():
        import pytest

        pytest.skip("corpus not present")
    chunks, docs = collect_chunks(corpus_root)
    assert docs > 0
    assert chunks
    assert all(c.line_start <= c.line_end for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
