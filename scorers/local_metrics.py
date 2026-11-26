"""Offline metrics - the heart of this project.

Every metric here is a *proxy* for what you actually care about. None of them
is ground truth. The point of writing them out in plain Python is that you can
see exactly what each one rewards, and therefore when it will lie to you.

What they measure, and how they lie
-----------------------------------
faithfulness
    "Is the answer supported by the retrieved context?"
    Ours: fraction of the answer's content words that appear in the context.
    Lies when: the answer is copied from context (score -> 1.0 even if it
    answers the wrong question), or when a correct paraphrase uses different
    words (score too low). Real RAGAS faithfulness uses an LLM to decompose the
    answer into claims and check each one; it costs money and is still a proxy.

answer_relevance
    "Does the answer address the question?"
    Ours: cosine similarity between question and answer embeddings.
    Lies when: short answers score low, or an answer reuses the question's
    words while adding nothing.

context_precision
    "Were the relevant chunks ranked above the irrelevant ones?"
    Ours: average precision over the retrieved chunks, graded by how much of
    the reference answer each chunk covers.
    Lies when: several chunks jointly cover the answer - no single chunk looks
    relevant, so precision is understated.

context_recall
    "Did retrieval bring back everything needed to answer?"
    Ours: fraction of the reference answer's content words covered by the
    union of retrieved chunks.
    Lies when: the reference answer paraphrases the source, so token overlap
    is low even though retrieval was correct.

answer_correctness
    "Does the answer agree with the reference answer?"
    Ours: 50% token F1 + 50% embedding cosine against the golden answer.
    Lies when: the answer is right but worded differently (F1 drops), or the
    answer is wrong but reuses the reference's words (F1 stays high).

refusal_accuracy
    "For questions the corpus cannot answer, did it decline?"
    Measured only on items marked ``"answerable": false``. This is the metric
    almost every tutorial skips, and it is the one users feel most: a confident
    wrong answer is worse than an honest "I don't know".

Design rule: a metric that has nothing to score (e.g. faithfulness on an
unanswerable question) returns ``None`` rather than 0. Averaging in "no data"
as a failure would quietly poison every number in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.chunker import split_sentences
from rag.embedder import HashingEmbedder, cosine_similarity, tokenize
from rag.pipeline import RunRecord


@dataclass(frozen=True)
class MetricScore:
    """A metric's value plus how many items it was actually computed on."""

    value: float
    count: int

    def as_dict(self) -> dict:
        return {"value": round(self.value, 4), "count": self.count}


def _content_tokens(text: str) -> set[str]:
    """Meaningful words only - stopwords are already dropped by tokenize()."""
    return set(tokenize(text))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def faithfulness(record: RunRecord) -> float | None:
    """Fraction of the answer's content words that the context supports."""
    if not record.answerable or record.refused:
        return None
    context_tokens = _content_tokens(" ".join(record.contexts))
    if not context_tokens:
        return 0.0

    ratios: list[float] = []
    for sentence in split_sentences(record.answer):
        tokens = _content_tokens(sentence)
        if tokens:
            ratios.append(len(tokens & context_tokens) / len(tokens))
    return _mean(ratios)


def answer_relevance(record: RunRecord, embedder: HashingEmbedder) -> float | None:
    """Cosine similarity between the question and the answer."""
    if not record.answerable or record.refused:
        return None
    similarity = cosine_similarity(
        embedder.embed(record.question), embedder.embed(record.answer)
    )
    return max(0.0, similarity)  # negative similarity is just "unrelated"


def context_precision(record: RunRecord) -> float | None:
    """Average precision over retrieved chunks, graded by reference coverage."""
    if not record.answerable:
        return None
    gt_tokens = _content_tokens(record.ground_truth)
    if not gt_tokens or not record.contexts:
        return None

    relevance = [
        len(gt_tokens & _content_tokens(chunk)) / len(gt_tokens)
        for chunk in record.contexts
    ]
    if sum(relevance) == 0.0:
        return 0.0

    weighted_sum = 0.0
    cumulative = 0.0
    for position, rel in enumerate(relevance, start=1):
        cumulative += rel
        weighted_sum += rel * (cumulative / position)
    return weighted_sum / sum(relevance)


def context_recall(record: RunRecord) -> float | None:
    """Fraction of the reference answer's content covered by retrieved chunks."""
    if not record.answerable:
        return None
    gt_tokens = _content_tokens(record.ground_truth)
    if not gt_tokens:
        return None
    covered = _content_tokens(" ".join(record.contexts))
    return len(gt_tokens & covered) / len(gt_tokens)


def _token_f1(prediction: str, reference: str) -> float:
    """Set-based F1 between two strings (order-blind, repetition-blind)."""
    pred = _content_tokens(prediction)
    ref = _content_tokens(reference)
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def answer_correctness(record: RunRecord, embedder: HashingEmbedder) -> float | None:
    """Blend of token F1 and embedding similarity against the golden answer."""
    if not record.answerable or record.refused:
        return None
    if not record.ground_truth:
        return None
    f1 = _token_f1(record.answer, record.ground_truth)
    similarity = max(
        0.0, cosine_similarity(embedder.embed(record.answer), embedder.embed(record.ground_truth))
    )
    return 0.5 * f1 + 0.5 * similarity


def refusal_accuracy(record: RunRecord) -> float | None:
    """Only defined for questions the corpus cannot answer."""
    if record.answerable:
        return None
    return 1.0 if record.refused else 0.0


METRIC_FUNCTIONS = {
    "faithfulness": lambda r, e: faithfulness(r),
    "answer_relevance": lambda r, e: answer_relevance(r, e),
    "context_precision": lambda r, e: context_precision(r),
    "context_recall": lambda r, e: context_recall(r),
    "answer_correctness": lambda r, e: answer_correctness(r, e),
    "refusal_accuracy": lambda r, e: refusal_accuracy(r),
}


def score_records(
    records: list[RunRecord],
    embedder: HashingEmbedder | None = None,
) -> dict[str, MetricScore]:
    """Compute every offline metric over a whole run."""
    embedder = embedder or HashingEmbedder()
    scores: dict[str, MetricScore] = {}

    for metric_name, fn in METRIC_FUNCTIONS.items():
        values = []
        for record in records:
            value = fn(record, embedder)
            if value is not None:
                values.append(value)
        scores[metric_name] = MetricScore(value=_mean(values), count=len(values))

    return scores


def per_item_scores(
    records: list[RunRecord],
    embedder: HashingEmbedder | None = None,
) -> list[dict]:
    """Every metric for every question.

    This is what you read when a number drops: one aggregate score tells you
    *that* something broke, the per-item table tells you *what*.
    """
    embedder = embedder or HashingEmbedder()
    rows: list[dict] = []

    for record in records:
        row: dict = {
            "question": record.question,
            "answer": record.answer,
            "refused": record.refused,
            "answerable": record.answerable,
            "contexts": record.chunk_ids,
        }
        for metric_name, fn in METRIC_FUNCTIONS.items():
            value = fn(record, embedder)
            row[metric_name] = None if value is None else round(value, 4)
        rows.append(row)

    return rows
