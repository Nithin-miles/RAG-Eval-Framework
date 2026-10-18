import os


def score_ragas(results, mock=False):
    """Returns a dict of RAGAS metrics. Falls back to deterministic
    mock scores if OPENAI_API_KEY is missing or --mock is passed."""
    use_mock = mock or not os.getenv("OPENAI_API_KEY")

    if use_mock:
        print("[ragas] MOCK mode - returning synthetic scores")
        return {
            "faithfulness": 0.87,
            "answer_relevance": 0.82,
            "context_precision": 0.79,
        }

    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset

    data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }
    ds = Dataset.from_dict(data)
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    return {
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevance": float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
    }
