"""Optional DeepEval adapter (``--mode deepeval``).

DeepEval is a pytest-style LLM testing framework. We use four of its metrics,
chosen because they line up with the stages of a RAG pipeline:

    FaithfulnessMetric            -> generation (is the answer grounded?)
    AnswerRelevancyMetric         -> generation (does it answer the question?)
    ContextualPrecisionMetric     -> retrieval  (are the useful chunks ranked up?)
    ContextualRecallMetric        -> retrieval  (are all the needed chunks there?)

Note what is *not* here: DeepEval has no ``AnswerCorrectnessMetric`` (checked
against 4.3, where ``dir(deepeval.metrics)`` has no such class). The previous
version of this project imported an answer-correctness-shaped class, actually
ran ``AnswerRelevancyMetric``, and published the result under the key
``answer_correctness``. A metric with the wrong name is worse than a missing
one, because it looks like it is working.
"""

from __future__ import annotations

from scorers import ScorerUnavailable, _require_api_key
from scorers.local_metrics import MetricScore


_METRIC_SPECS = (
    # (canonical name, deepeval attribute name)
    ("faithfulness", "FaithfulnessMetric"),
    ("answer_relevance", "AnswerRelevancyMetric"),
    ("context_precision", "ContextualPrecisionMetric"),
    ("context_recall", "ContextualRecallMetric"),
)


def _load_metrics() -> dict[str, type]:
    try:
        from deepeval import metrics as dm
    except Exception as exc:
        raise ScorerUnavailable(
            f"Could not import deepeval ({type(exc).__name__}: {exc}).\n"
            "Install the optional extras with: pip install -r requirements-llm.txt"
        ) from exc

    loaded: dict[str, type] = {}
    for canonical, attribute in _METRIC_SPECS:
        cls = getattr(dm, attribute, None)
        if cls is not None:
            loaded[canonical] = cls
    if not loaded:
        raise ScorerUnavailable(
            "None of the expected DeepEval metrics were found. "
            f"Available: {[n for n in dir(dm) if n.endswith('Metric')]}"
        )
    return loaded


def score_with_deepeval(records) -> dict[str, MetricScore]:
    """Grade answerable run records with DeepEval's LLM-judged metrics."""
    _require_api_key()

    try:
        from deepeval.test_case import LLMTestCase
    except Exception as exc:
        raise ScorerUnavailable(f"Could not import deepeval test case: {exc}") from exc

    metric_classes = _load_metrics()
    usable = [r for r in records if r.answerable]
    if not usable:
        raise ScorerUnavailable("No answerable records to score with DeepEval.")

    # Instantiate once: these objects are stateless between measure() calls and
    # creating one per record would be slow.
    metric_objects = {
        name: cls(threshold=0.0, include_reason=False)
        for name, cls in metric_classes.items()
    }

    collected: dict[str, list[float]] = {name: [] for name in metric_objects}

    for record in usable:
        test_case = LLMTestCase(
            input=record.question,
            actual_output=record.answer,
            expected_output=record.ground_truth,
            retrieval_context=list(record.contexts),
        )
        for name, metric in metric_objects.items():
            try:
                metric.measure(test_case)
                collected[name].append(float(metric.score))
            except Exception as exc:  # one bad metric should not kill the run
                print(f"[deepeval] {name} failed on one item: {exc}")

    scores: dict[str, MetricScore] = {}
    for name, values in collected.items():
        if values:
            scores[name] = MetricScore(
                value=sum(values) / len(values),
                count=len(values),
            )

    if not scores:
        raise ScorerUnavailable("DeepEval produced no scores; see errors above.")
    return scores
