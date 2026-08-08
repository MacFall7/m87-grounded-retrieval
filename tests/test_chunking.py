"""Chunking is the ceiling on retrieval quality, so it is tested as a pure function."""

from __future__ import annotations

import textwrap

from grounded_retrieval.chunking import Chunk, chunk_markdown

DOC = textwrap.dedent(
    """\
    # Spine Lite

    The kernel refuses rather than degrading. Refusal is a first class outcome and it
    carries its own exit code so a caller can branch on it without parsing text.

    ## Policy engine

    Policies are evaluated before any effect is applied. A policy that cannot be
    evaluated is treated as a denial, never as a pass.

    ### Refusal semantics

    Exit code 2 means the kernel is incapable of proceeding. Exit code 1 means it
    crashed. Conflating the two loses the distinction the contract depends on.

    ## Hooks

    Hooks observe, they do not decide. A hook that can veto is a policy wearing a
    different name and it breaks the audit trail.
    """
)


def _chunk(text: str, **kw) -> list[Chunk]:
    kw.setdefault("min_chars", 20)
    return chunk_markdown(text, source_repo="demo", source_path="README.md", **kw)


def test_chunking_is_deterministic():
    first = _chunk(DOC)
    second = _chunk(DOC)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_chunking_is_pure_and_does_not_mutate_input():
    original = DOC
    _chunk(DOC)
    assert DOC == original


def test_heading_path_is_hierarchical():
    chunks = _chunk(DOC)
    by_heading = {c.heading_path for c in chunks}
    assert ("Spine Lite",) in by_heading
    assert ("Spine Lite", "Policy engine") in by_heading
    assert ("Spine Lite", "Policy engine", "Refusal semantics") in by_heading


def test_heading_path_pops_back_to_sibling_level():
    """A later h2 must not inherit the previous h3, which would mislabel provenance."""
    chunks = _chunk(DOC)
    hooks = [c for c in chunks if c.heading_path and c.heading_path[-1] == "Hooks"]
    assert hooks, "expected a chunk under the Hooks heading"
    assert hooks[0].heading_path == ("Spine Lite", "Hooks")


def test_headings_inside_code_fences_do_not_split_blocks():
    doc = textwrap.dedent(
        """\
        ## Install

        Run the installer and then verify the checksum before trusting the binary.

        ```bash
        # Heading looking comment
        ## Another one
        make install
        ### And a third
        ```

        The fenced block above is one shell snippet and must survive as one chunk.
        """
    )
    chunks = _chunk(doc)
    assert len(chunks) == 1
    only = chunks[0]
    assert only.heading_path == ("Install",)
    assert "# Heading looking comment" in only.text
    assert "### And a third" in only.text


def test_tilde_fences_are_also_respected():
    doc = textwrap.dedent(
        """\
        ## Config

        Configuration lives in one file so drift between environments is visible.

        ~~~yaml
        # not a heading
        key: value
        ~~~

        Everything above belongs to the Config section and stays together.
        """
    )
    chunks = _chunk(doc)
    assert len(chunks) == 1
    assert "# not a heading" in chunks[0].text


def test_oversized_block_splits_on_paragraph_boundaries():
    paragraphs = [
        f"Paragraph {i} explains one governance invariant in enough words to be "
        f"a realistic unit of prose rather than a token."
        for i in range(12)
    ]
    doc = "## Long section\n\n" + "\n\n".join(paragraphs) + "\n"
    chunks = _chunk(doc, max_chars=300, overlap_chars=120)

    assert len(chunks) > 1
    for c in chunks:
        # A split that lands mid sentence produces embedding noise, so every piece
        # must begin at a paragraph start.
        assert c.text.startswith("Paragraph") or c.text.startswith("## Long section")
        assert not c.text.startswith(" ")


def test_oversized_split_preserves_heading_path_on_every_piece():
    doc = "## Deep section\n\n" + "\n\n".join(
        f"Sentence block number {i} with enough content to be worth retrieving alone."
        for i in range(10)
    )
    chunks = _chunk(doc, max_chars=250, overlap_chars=80)
    assert len(chunks) > 1
    assert all(c.heading_path == ("Deep section",) for c in chunks)


def test_line_spans_are_ordered_and_within_the_document():
    total_lines = len(DOC.splitlines())
    chunks = _chunk(DOC)
    for c in chunks:
        assert 1 <= c.line_start <= c.line_end <= total_lines


def test_ordinals_are_dense_and_ascending():
    chunks = _chunk(DOC)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_short_fragments_are_dropped():
    doc = "# T\n\n![badge](x)\n\n## U\n\nok\n"
    assert _chunk(doc, min_chars=80) == []


def test_chunk_id_is_stable_across_calls():
    a = _chunk(DOC)[0]
    b = _chunk(DOC)[0]
    assert a.chunk_id == b.chunk_id


def test_chunk_id_changes_when_content_changes():
    base = _chunk(DOC)[0]
    moved = Chunk(
        text=base.text + " extra",
        source_repo=base.source_repo,
        source_path=base.source_path,
        heading_path=base.heading_path,
        line_start=base.line_start,
        line_end=base.line_end,
        ordinal=base.ordinal,
    )
    assert moved.chunk_id != base.chunk_id


def test_chunk_id_ignores_ordinal():
    """Ordinal is presentation order, not identity.

    Inserting a paragraph earlier in a file must not renumber every downstream chunk
    id, otherwise an unchanged section reingests as new rows and the index stops
    being idempotent.
    """
    base = _chunk(DOC)[0]
    shifted = Chunk(
        text=base.text,
        source_repo=base.source_repo,
        source_path=base.source_path,
        heading_path=base.heading_path,
        line_start=base.line_start,
        line_end=base.line_end,
        ordinal=base.ordinal + 7,
    )
    assert shifted.chunk_id == base.chunk_id


def test_chunk_id_is_sensitive_to_provenance():
    base = _chunk(DOC)[0]
    elsewhere = Chunk(
        text=base.text,
        source_repo="other-repo",
        source_path=base.source_path,
        heading_path=base.heading_path,
        line_start=base.line_start,
        line_end=base.line_end,
        ordinal=base.ordinal,
    )
    assert elsewhere.chunk_id != base.chunk_id


def test_locator_is_human_checkable():
    c = _chunk(DOC)[0]
    assert c.locator() == f"demo:README.md:L{c.line_start}-L{c.line_end}"


def test_context_header_falls_back_to_path_without_headings():
    chunks = chunk_markdown(
        "Plain prose with no headings at all, long enough to survive the minimum.",
        source_repo="demo",
        source_path="notes.md",
        min_chars=20,
    )
    assert chunks[0].heading_path == ()
    assert chunks[0].context_header() == "notes.md"


def test_embedding_text_prefixes_the_breadcrumb():
    c = _chunk(DOC)[1]
    assert c.embedding_text().startswith(c.context_header())
    assert c.text in c.embedding_text()


def test_empty_document_yields_no_chunks():
    assert chunk_markdown("", source_repo="demo", source_path="empty.md") == []
