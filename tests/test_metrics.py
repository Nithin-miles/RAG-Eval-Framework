"""Tests for the offline metrics.

These are the most important tests in the repo: an evaluation harness with a
buggy metric is worse than no evaluation, because it gives you confidence you
have not earned.
"""

import pytest

from rag.embedder import HashingEmbedder
from rag.pipeline import RunRecord
from scorers.local_metrics import (
    METRIC_FUNCTIONS,
    answer_correctness,
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
    per_item_scores,
    refusal_accuracy,
    score_records,
)

GT = "Refunds are available within 30 days of purchase."
REFUND_CHUNK = "Refunds are available within 30 days of purchase for unused items."
SHIPPING_CHUNK = "Standard shipping takes 5 to 7 business days."
# Shares no content words with GT - needed, because "days" alone would leak in.
UNRELATED_CHUNK = "Gift cards can be combined with a credit card on a single order."


def record(**overrides) -> RunRecord:
    base = dict(
        question="What is the refund window?",
        answer=REFUND_CHUNK,
        contexts=[REFUND_CHUNK],
        chunk_ids=["refunds#0"],
        ground_truth=GT,
        answerable=True,
        refused=False,
    )
    base.update(overrides)
    return RunRecord(**base)


def test_perfect_answer_scores_one():
    rec = record()
    assert faithfulness(rec) == 1.0
    assert context_recall(rec) == 1.0
    assert answer_correctness(rec, HashingEmbedder()) > 0.80


def test_unsupported_answer_has_zero_faithfulness():
    rec = record(answer="Bananas are an excellent source of potassium.")
    assert faithfulness(rec) == 0.0


def test_refusals_are_excluded_not_penalised():
    rec = record(answer="I don't have enough information.", refused=True)
    assert faithfulness(rec) is None
    assert answer_correctness(rec, HashingEmbedder()) is None


def test_context_recall_drops_when_retrieval_misses_the_answer():
    rec = record(contexts=[SHIPPING_CHUNK])
    assert context_recall(rec) < 0.5


def test_context_precision_rewards_relevant_chunks_first():
    good = record(contexts=[REFUND_CHUNK, SHIPPING_CHUNK])
    bad = record(contexts=[SHIPPING_CHUNK, REFUND_CHUNK])
    assert context_precision(good) > context_precision(bad)
    assert context_precision(bad) < 1.0


def test_context_precision_is_zero_when_nothing_is_relevant():
    rec = record(contexts=[UNRELATED_CHUNK])
    assert context_precision(rec) == 0.0


def test_refusal_accuracy_only_applies_to_unanswerable_questions():
    assert refusal_accuracy(record(answerable=True)) is None
    assert refusal_accuracy(record(answerable=False, refused=True)) == 1.0
    assert refusal_accuracy(record(answerable=False, refused=False)) == 0.0


def test_every_metric_stays_in_range():
    embedder = HashingEmbedder()
    cases = [
        record(),
        record(answer="", refused=False),
        record(contexts=[], chunk_ids=[]),
        record(answerable=False, refused=True),
        record(answerable=False, refused=False),
        record(ground_truth=""),
    ]
    for rec in cases:
        for name, fn in METRIC_FUNCTIONS.items():
            value = fn(rec, embedder)
            assert value is None or 0.0 <= value <= 1.0, f"{name} out of range: {value}"


def test_aggregate_counts_only_measured_items():
    records = [record(), record(answerable=False, refused=True)]
    scores = score_records(records)
    assert scores["faithfulness"].count == 1  # the unanswerable one is excluded
    assert scores["refusal_accuracy"].count == 1
    assert scores["faithfulness"].value == 1.0


def test_empty_run_does_not_crash():
    scores = score_records([])
    assert all(score.value == 0.0 and score.count == 0 for score in scores.values())


def test_per_item_scores_expose_failures_for_debugging():
    rows = per_item_scores([record(), record(answer="Something unrelated entirely.")])
    assert len(rows) == 2
    low = [r for r in rows if r["faithfulness"] is not None and r["faithfulness"] < 0.5]
    assert len(low) == 1, "one item should be visibly unfaithful"
