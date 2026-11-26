"""Tests for the deterministic generator, including its refusal behaviour."""

from rag.chunker import Chunk
from rag.generator import REFUSAL_TEXT, ExtractiveGenerator
from rag.retriever import RetrievedChunk


def retrieved(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(doc_id=f"doc{i}", chunk_id=f"doc{i}#0", text=text, chunk_index=0),
            score=1.0 - i * 0.1,
        )
        for i, text in enumerate(texts)
    ]


def test_answers_from_the_most_relevant_sentence():
    context = retrieved(
        "Standard shipping takes 5 to 7 business days. "
        "Express shipping costs an additional 15 dollars."
    )
    answer = ExtractiveGenerator().generate("How long does standard shipping take?", context)
    assert not answer.refused
    assert answer.text.startswith("Standard shipping takes 5 to 7 business days")


def test_off_topic_second_sentence_is_filtered_out():
    context = retrieved(
        "Refund processing takes 5 to 7 business days. "
        "Gift cards can be combined with a credit card on a single order."
    )
    answer = ExtractiveGenerator().generate("How long does refund processing take?", context)
    assert "Gift cards" not in answer.text


def test_refuses_when_nothing_in_context_matches():
    context = retrieved("Standard shipping takes 5 to 7 business days.")
    answer = ExtractiveGenerator().generate("Do you sell refurbished laptops?", context)
    assert answer.refused
    assert answer.text == REFUSAL_TEXT


def test_refuses_when_retrieval_returned_nothing():
    answer = ExtractiveGenerator().generate("What is the refund window?", [])
    assert answer.refused


def test_refuses_on_empty_question():
    answer = ExtractiveGenerator().generate("   ", retrieved("Refunds within 30 days."))
    assert answer.refused


def test_max_sentences_is_respected():
    context = retrieved(
        "Refunds are available within 30 days of purchase. "
        "Refund processing takes 5 to 7 business days."
    )
    answer = ExtractiveGenerator(max_sentences=1).generate("refund processing days", context)
    assert answer.text.count(".") <= 2  # one sentence plus its full stop


def test_answer_is_deterministic():
    context = retrieved("Refunds are available within 30 days of purchase.")
    first = ExtractiveGenerator().generate("refund window", context).text
    second = ExtractiveGenerator().generate("refund window", context).text
    assert first == second
