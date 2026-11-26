"""Tests for chunking: the step that decides what retrieval can and cannot find."""

from rag.chunker import chunk_text, split_sentences

TEXT = (
    "Refunds are available within 30 days of purchase. "
    "The customer must provide proof of purchase. "
    "Refund processing takes 5 to 7 business days. "
    "Shipping charges are not refunded unless the item arrived damaged. "
    "Digital products and gift cards are non-refundable."
)


def test_split_sentences_returns_each_sentence():
    sentences = split_sentences(TEXT)
    assert len(sentences) == 5
    assert sentences[0].startswith("Refunds are available")
    assert sentences[-1].endswith("non-refundable.")


def test_split_sentences_drops_blank_fragments():
    assert split_sentences("One.   Two.  ") == ["One.", "Two."]


def test_chunks_are_never_empty_and_have_stable_ids():
    chunks = chunk_text("refunds", TEXT, max_chars=120, overlap_chars=30)
    assert chunks, "expected at least one chunk"
    assert all(chunk.text.strip() for chunk in chunks)
    assert [c.chunk_id for c in chunks] == [f"refunds#{i}" for i in range(len(chunks))]


def test_chunks_respect_the_size_budget():
    chunks = chunk_text("refunds", TEXT, max_chars=120, overlap_chars=30)
    for chunk in chunks:
        # A single sentence longer than the budget is allowed to overflow,
        # but nothing made of several sentences may.
        single_sentence = len(split_sentences(chunk.text)) == 1
        assert single_sentence or len(chunk.text) <= 120


def test_chunking_is_deterministic():
    first = chunk_text("refunds", TEXT, max_chars=120, overlap_chars=30)
    second = chunk_text("refunds", TEXT, max_chars=120, overlap_chars=30)
    assert [c.text for c in first] == [c.text for c in second]


def test_overlap_keeps_context_across_the_boundary():
    # Short sentences so that one whole sentence fits inside the overlap budget;
    # overlap is sentence-aligned and best-effort by design.
    text = "Alpha beta. Gamma delta. Epsilon zeta. Eta theta."
    chunks = chunk_text("short", text, max_chars=30, overlap_chars=15)
    assert len(chunks) > 1, "test needs more than one chunk"
    first_sentences = split_sentences(chunks[0].text)
    second_sentences = split_sentences(chunks[1].text)
    # The last sentence of chunk N must be the first sentence of chunk N+1.
    assert first_sentences[-1] == second_sentences[0]


def test_invalid_parameters_are_rejected():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("x", TEXT, max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("x", TEXT, max_chars=100, overlap_chars=100)
