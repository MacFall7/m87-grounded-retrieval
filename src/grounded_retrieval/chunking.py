"""Structure-aware markdown chunking with first-class provenance.

Design note
-----------
Most RAG pipelines chunk by character count and throw away structure. That makes two
problems that only show up later, in evaluation:

1. A chunk that straddles a heading boundary mixes two topics, so it is retrievable
   for both and precise for neither.
2. Without a line span you cannot cite. "This came from README.md" is not a citation,
   it is an attribution. A citation has to be checkable by a human in a few seconds.

So chunking here is a pure function over parsed structure, and every chunk carries the
provenance needed to verify it: source repo, file path, heading path, and the exact
line span. `Chunk.locator()` produces a string a reviewer can paste into a file viewer.

Purity matters for a second reason: it makes chunking testable without a model, a
database, or a network. The retrieval quality of this system is bounded by the quality
of this file, so it is the part that most needs to be independently verifiable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text with everything needed to cite it."""

    text: str
    source_repo: str
    source_path: str
    heading_path: tuple[str, ...]
    line_start: int  # 1-indexed, inclusive
    line_end: int  # 1-indexed, inclusive
    ordinal: int  # position within the source document

    @property
    def chunk_id(self) -> str:
        """Stable content-addressed id.

        Derived from provenance plus content, so re-ingesting an unchanged corpus
        produces identical ids and the index is idempotent. If a file changes, only
        the chunks that actually changed get new ids.
        """
        payload = "\x1f".join(
            [
                self.source_repo,
                self.source_path,
                "/".join(self.heading_path),
                str(self.line_start),
                str(self.line_end),
                self.text,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def locator(self) -> str:
        """Human-checkable pointer, e.g. 'M87-Spine-lite:README.md:L40-L58'."""
        return f"{self.source_repo}:{self.source_path}:L{self.line_start}-L{self.line_end}"

    def context_header(self) -> str:
        """Heading breadcrumb prepended to the embedded text.

        A chunk that says 'it fails closed' is ambiguous on its own. The same chunk
        prefixed with 'Spine Lite > Policy engine > Refusal semantics' is not. This
        measurably improves retrieval on short chunks and costs a few tokens.
        """
        if not self.heading_path:
            return self.source_path
        return " > ".join(self.heading_path)

    def embedding_text(self) -> str:
        return f"{self.context_header()}\n\n{self.text}"


@dataclass
class _Block:
    lines: list[str] = field(default_factory=list)
    start: int = 0
    heading_path: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _iter_structural_blocks(lines: Sequence[str]) -> Iterator[_Block]:
    """Split a document at heading boundaries, ignoring headings inside code fences.

    The fence tracking is not decoration. Governance repos are dense with shell and
    Python blocks full of `# comment` lines, and a naive heading regex shreds them
    into meaningless fragments.
    """
    current = _Block(start=1)
    stack: list[tuple[int, str]] = []
    fence: str | None = None

    for idx, raw in enumerate(lines, start=1):
        fence_match = FENCE.match(raw)
        if fence_match:
            marker = fence_match.group(1)[0] * 3
            if fence is None:
                fence = marker
            elif raw.strip().startswith(fence):
                fence = None
            current.lines.append(raw)
            continue

        if fence is not None:
            current.lines.append(raw)
            continue

        heading = ATX_HEADING.match(raw)
        if heading:
            if current.text:
                yield current
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = _Block(
                start=idx,
                heading_path=tuple(t for _, t in stack),
            )
            current.lines.append(raw)
            continue

        current.lines.append(raw)

    if current.text:
        yield current


def _split_oversized(
    block: _Block, max_chars: int, overlap_chars: int
) -> Iterator[tuple[list[str], int]]:
    """Split a too-large block on paragraph boundaries, never mid-sentence.

    Overlap is applied in whole paragraphs rather than a raw character slice, so an
    overlapping chunk is still readable prose. A chunk that begins mid-word is
    embedding noise.
    """
    paragraphs: list[tuple[list[str], int]] = []
    buf: list[str] = []
    buf_start = block.start

    for offset, line in enumerate(block.lines):
        if line.strip() == "" and buf:
            paragraphs.append((buf, buf_start))
            buf = []
            buf_start = block.start + offset + 1
        else:
            if not buf:
                buf_start = block.start + offset
            buf.append(line)
    if buf:
        paragraphs.append((buf, buf_start))

    window: list[tuple[list[str], int]] = []
    size = 0
    for para, start in paragraphs:
        para_len = sum(len(x) + 1 for x in para)
        if window and size + para_len > max_chars:
            flat = [ln for p, _ in window for ln in p]
            yield flat, window[0][1]
            carry: list[tuple[list[str], int]] = []
            carried = 0
            for p, s in reversed(window):
                p_len = sum(len(x) + 1 for x in p)
                if carried + p_len > overlap_chars:
                    break
                carry.insert(0, (p, s))
                carried += p_len
            window = carry
            size = carried
        window.append((para, start))
        size += para_len

    if window:
        flat = [ln for p, _ in window for ln in p]
        yield flat, window[0][1]


def chunk_markdown(
    content: str,
    *,
    source_repo: str,
    source_path: str,
    max_chars: int = 1200,
    min_chars: int = 80,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Chunk one markdown document. Pure: same input always gives the same output.

    `min_chars` drops fragments too small to carry meaning (a lone heading, a badge
    row). Those pollute an index: they are short, so they score well on cosine
    similarity against short queries, and they contain nothing.
    """
    lines = content.splitlines()
    chunks: list[Chunk] = []
    ordinal = 0

    for block in _iter_structural_blocks(lines):
        block_len = len(block.text)
        if block_len == 0:
            continue

        if block_len <= max_chars:
            pieces = [(block.lines, block.start)]
        else:
            pieces = list(_split_oversized(block, max_chars, overlap_chars))

        for piece_lines, piece_start in pieces:
            text = "\n".join(piece_lines).strip()
            if len(text) < min_chars:
                continue
            chunks.append(
                Chunk(
                    text=text,
                    source_repo=source_repo,
                    source_path=source_path,
                    heading_path=block.heading_path,
                    line_start=piece_start,
                    line_end=piece_start + len(piece_lines) - 1,
                    ordinal=ordinal,
                )
            )
            ordinal += 1

    return chunks
