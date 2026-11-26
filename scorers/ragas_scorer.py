"""Optional RAGAS adapter (``--mode ragas``).

RAGAS is the library that popularised LLM-judge RAG metrics: faithfulness,
answer relevancy, context precision, context recall, answer correctness.

Two practical warnings, both learned the hard way in this repo:

* RAGAS changes its public API between minor versions (0.2 renamed metrics,
  0.4 moved them to ``ragas.metrics.collections``). This adapter tries the
  stable entry points and reports honestly if none of them work.
* An import can fail for reasons unrelated to your code - for example a
  transitive ``langchain-community`` module that no longer exists. That is a
  real failure, not something to paper over with fake scores.
"""

from __future__ import annotations

from scorers import ScorerUnavailable, _require_api_key
from scorers.local_metrics import MetricScore


def _ragas_metrics() -> list:
    """Return metric objects from whichever RAGAS API is installed."""
    # RAGAS >= 0.4: metric classes live in ragas.metrics.collections.
    try:
        from ragas.metrics.collections import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            ResponseRelevancy,
        )

        return [
            Faithfulness(),
            ResponseRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]
    except Exception:
        pass

    # RAGAS 0.1 / 0.2: module-level singleton metrics.
    try:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        return [faithfulness, answer_relevancy, context_precision, context_recall]
    except Exception as exc:
        raise ScorerUnavailable(
            "Could not import RAGAS metrics from this installation "
            f"({type(exc).__name__}: {exc}).\n"
            "RAGAS's API changed a lot across versions. Try:\n"
            "    pip install -r requirements-llm.txt\n"
            "and check the version with: python -c \"import ragas; print(ragas.__version__)\""
        ) from exc


_NAME_MAP = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevance",
    "response_relevancy": "answer_relevance",
    "context_precision": "context_precision",
    "context_recall": "context_recall",
    "answer_correctness": "answer_correctness",
}


def score_with_ragas(records) -> dict[str, MetricScore]:
    """Grade answerable run records with RAGAS.

    Unanswerable questions are excluded: RAGAS metrics assume a real answer
    exists, and scoring a correct refusal as "unfaithful" would be meaningless.
    """
    _require_api_key()

    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
    except Exception as exc:
        raise ScorerUnavailable(
            f"Could not import RAGAS ({type(exc).__name__}: {exc}).\n"
            "Install the optional extras with: pip install -r requirements-llm.txt"
        ) from exc

    usable = [r for r in records if r.answerable]
    if not usable:
        raise ScorerUnavailable("No answerable records to score with RAGAS.")

    samples = [
        SingleTurnSample(
            user_input=r.question,
            response=r.answer,
            retrieved_contexts=list(r.contexts),
            reference=r.ground_truth,
        )
        for r in usable
    ]

    try:
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(dataset, metrics=_ragas_metrics())
        frame = result.to_pandas()
    except Exception as exc:
        raise ScorerUnavailable(f"RAGAS evaluation failed: {exc}") from exc

    scores: dict[str, MetricScore] = {}
    for column in frame.columns:
        canonical = _NAME_MAP.get(column.lower())
        if not canonical:
            continue
        series = frame[column].dropna()
        if series.empty:
            continue
        scores[canonical] = MetricScore(value=float(series.mean()), count=len(series))

    if not scores:
        raise ScorerUnavailable(
            "RAGAS returned no recognised metric columns. "
            f"Columns were: {list(frame.columns)}"
        )

    return scores
