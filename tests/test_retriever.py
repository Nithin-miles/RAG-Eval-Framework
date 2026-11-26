"""Tests for retrieval: does the right chunk come back, and is it reproducible?"""

import pytest

from rag.chunker import Chunk
from rag.embedder import HashingEmbedder, cosine_similarity
from rag.pipeline import build_chunks
from rag.retriever import (
    KeywordRetriever,
    VectorRetriever,
    build_retriever,
)


@pytest.fixture(scope="module")
def chunks():
    return build_chunks("data/corpus")


def _make(chunk_id: str, text: str) -> Chunk:
    return Chunk(doc_id=chunk_id.split("#")[0], chunk_id=chunk_id, text=text, chunk_index=0)


def test_embedder_is_deterministic_across_instances():
    a = HashingEmbedder().embed("refund window")
    b = HashingEmbedder().embed("refund window")
    assert (a == b).all()
    assert abs(float((a * a).sum()) - 1.0) < 1e-9, "vectors must be unit length"


def test_cosine_similarity_bounds_and_symmetry():
    embedder = HashingEmbedder()
    a = embedder.embed("shipping takes five days")
    b = embedder.embed("shipping takes five days")
    c = embedder.embed("gift card refund policy")
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-9
    assert cosine_similarity(a, c) < cosine_similarity(a, b)
    assert -1.0001 <= cosine_similarity(a, c) <= 1.0001


def test_empty_text_has_zero_similarity():
    embedder = HashingEmbedder()
    assert cosine_similarity(embedder.embed(""), embedder.embed("refund")) == 0.0


def test_vector_retriever_finds_the_shipping_chunk(chunks):
    retriever = VectorRetriever(chunks)
    results = retriever.retrieve("How long does standard shipping take?", k=3)
    assert results
    # The requirement is that the needed chunk is in the top-k, not that it is
    # rank 1: longer chunks dilute cosine similarity (see docs/05-experiments.md).
    assert "shipping" in [r.chunk.doc_id for r in results]
    assert results[0].score >= results[-1].score, "results must be ranked"


def test_retrieval_respects_k(chunks):
    for retriever in (VectorRetriever(chunks), KeywordRetriever(chunks)):
        assert len(retriever.retrieve("refund", k=2)) == 2
        assert len(retriever.retrieve("refund", k=len(chunks) + 5)) == len(chunks)


def test_retrieval_is_deterministic_ties_included():
    chunks = [_make("a#0", "refund policy"), _make("b#0", "refund policy")]
    for retriever in (VectorRetriever(chunks), KeywordRetriever(chunks)):
        first = [(r.chunk_id, r.score) for r in retriever.retrieve("refund", k=2)]
        second = [(r.chunk_id, r.score) for r in retriever.retrieve("refund", k=2)]
        assert first == second


def test_empty_question_returns_nothing(chunks):
    assert KeywordRetriever(chunks).retrieve("   ", k=3) == []


def test_scores_are_non_negative(chunks):
    for retriever in (VectorRetriever(chunks), KeywordRetriever(chunks)):
        for result in retriever.retrieve("can I get my money back", k=3):
            assert 0.0 <= result.score <= 1.0 + 1e-9


def test_build_retriever_rejects_unknown_kind(chunks):
    with pytest.raises(ValueError):
        build_retriever("telepathic", chunks)
    assert build_retriever("vector", chunks).name == "vector"
    assert build_retriever("keyword", chunks).name == "keyword"
