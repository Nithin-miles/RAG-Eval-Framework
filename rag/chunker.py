"""Cut documents into chunks.

Why chunk at all?
    * A retriever compares the question against *chunks*, not whole documents.
    * If chunks are too big, the answer is buried in noise and the LLM gets
      confused (and you pay for tokens you do not need).
    * If chunks are too small, the answer gets split across two chunks and
      retrieval misses it.

So chunk size is a real trade-off, and it is one of the first things you
experiment with. Try max_chars=120 vs 240 vs 480 and watch the scores move.

The strategy here is "sentence-aware packing with overlap":
    1. split text into sentences
    2. keep adding sentences until the chunk would exceed max_chars
    3. emit the chunk, then start the next one with a little overlap
       (the overlap stops an answer from being cut in half at a boundary)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split on sentence-ending punctuation followed by whitespace.
# Good enough for this corpus; production systems use a real sentence splitter.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A searchable piece of a document."""

    doc_id: str
    chunk_id: str  # stable id like "refunds#0" - used in results and debugging
    text: str
    chunk_index: int
    title: str = ""  # the document title, used for retrieval only

    @property
    def embed_text(self) -> str:
        """Text used for matching: the title plus the prose.

        Keeping the title here (and *only* here) gives retrieval a strong
        document-level signal without letting "## Refund Policy" leak into an
        answer. This is a miniature version of "contextual chunk headers".
        """
        return f"{self.title}. {self.text}" if self.title else self.text

    def __str__(self) -> str:  # handy when printing retrieval results
        return f"[{self.chunk_id}] {self.text}"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, dropping empty fragments."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def chunk_text(
    doc_id: str,
    text: str,
    *,
    title: str = "",
    max_chars: int = 240,
    overlap_chars: int = 60,
) -> list[Chunk]:
    """Split one document's text into overlapping, sentence-aligned chunks.

    ``overlap_chars`` is a target: we repeat trailing sentences of the
    previous chunk while they still fit inside that budget.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than max_chars")

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        """Emit the current buffer as a chunk, then seed the overlap."""
        nonlocal current, current_len
        if not current:
            return
        body = " ".join(current).strip()
        if body:
            index = len(chunks)
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{index}",
                    text=body,
                    chunk_index=index,
                    title=title,
                )
            )

        # Carry the last few sentences into the next chunk as overlap.
        carry: list[str] = []
        carry_len = 0
        for sentence in reversed(current):
            if carry_len + len(sentence) + 1 > overlap_chars:
                break
            carry.insert(0, sentence)
            carry_len += len(sentence) + 1
        # Never let the overlap swallow the whole chunk or nothing progresses.
        if len(carry) == len(current):
            carry = carry[-1:]
        current = list(carry)
        current_len = sum(len(s) + 1 for s in current)

    for sentence in split_sentences(text):
        # A single very long sentence becomes its own chunk.
        if current and current_len + len(sentence) + 1 > max_chars:
            flush()
        current.append(sentence)
        current_len += len(sentence) + 1

    # Final chunk: no overlap needed afterwards.
    if current:
        body = " ".join(current).strip()
        if body:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{len(chunks)}",
                    text=body,
                    chunk_index=len(chunks),
                    title=title,
                )
            )

    return chunks
