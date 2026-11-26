"""Retrievers: given a question, return the most useful chunks.

Two implementations, on purpose:

``KeywordRetriever``
    The naive baseline. Counts shared words between the question and the chunk.
    Fast, zero dependencies, and it fails in a very specific way: it cannot
    match synonyms. Ask "can I get my money back" and it scores badly against
    a chunk that says "refunds are available", because no words are shared.

``VectorRetriever``
    Embeds the question and every chunk, then ranks by cosine similarity.
    With real embedding models this fixes the synonym problem. With our offline
    hashing embedder it only partly fixes it - which is realistic, and it is why
    the project lets you swap in a real model later.

Running the evaluation against both retrievers, with everything else held
fixed, is the single most useful experiment in this repo: it isolates the
effect of the retrieval stage on final answer quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag.chunker import Chunk
from rag.embedder import HashingEmbedder, cosine_similarity, tokenize


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk plus the score that earned it a place in the top-k."""

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class Retriever(Protocol):
    """The interface every retriever must satisfy."""

    name: str

    def retrieve(self, question: str, k: int = 3) -> list[RetrievedChunk]:
        ...


class KeywordRetriever:
    """Naive lexical baseline: cosine over the sets of question/chunk words."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.name = "keyword"
        self._chunks = list(chunks)
        # Match on embed_text: the document title is a useful signal here, and
        # it is deliberately kept out of the chunk prose (see Chunk.embed_text).
        self._token_sets = [set(tokenize(c.embed_text)) for c in self._chunks]

    def retrieve(self, question: str, k: int = 3) -> list[RetrievedChunk]:
        q_tokens = set(tokenize(question))
        if not q_tokens:
            return []

        scored: list[RetrievedChunk] = []
        for chunk, chunk_tokens in zip(self._chunks, self._token_sets):
            union = len(q_tokens | chunk_tokens)
            score = len(q_tokens & chunk_tokens) / union if union else 0.0
            scored.append(RetrievedChunk(chunk=chunk, score=score))

        # Sort by score, then chunk_id: deterministic ties make scores reproducible.
        scored.sort(key=lambda r: (-r.score, r.chunk_id))
        return scored[:k]


class VectorRetriever:
    """Rank chunks by cosine similarity between embeddings."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: HashingEmbedder | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("VectorRetriever needs at least one chunk")
        self.name = "vector"
        self._chunks = list(chunks)
        self._embedder = embedder or HashingEmbedder()
        # Precompute once; retrieval is then a single matrix multiply.
        self._matrix = self._embedder.embed_many([c.embed_text for c in self._chunks])

    def retrieve(self, question: str, k: int = 3) -> list[RetrievedChunk]:
        q_vec = self._embedder.embed(question)
        # Every row of the matrix is already unit-normalised, so the dot
        # product *is* the cosine similarity.
        scores = self._matrix @ q_vec
        order = sorted(
            range(len(self._chunks)),
            key=lambda i: (-scores[i], self._chunks[i].chunk_id),
        )
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in order[:k]
        ]


def build_retriever(kind: str, chunks: list[Chunk]) -> Retriever:
    """Factory used by the CLI: ``--retriever keyword|vector``."""
    kind = kind.lower()
    if kind == "keyword":
        return KeywordRetriever(chunks)
    if kind == "vector":
        return VectorRetriever(chunks)
    raise ValueError(f"Unknown retriever {kind!r}. Choose 'keyword' or 'vector'.")
