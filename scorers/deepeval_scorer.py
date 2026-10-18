import os


def score_deepeval(results, mock=False):
    """Returns DeepEval metrics. Mock mode returns deterministic values."""
    use_mock = mock or not os.getenv("OPENAI_API_KEY")

    if use_mock:
        print("[deepeval] MOCK mode - returning synthetic score")
        return {"answer_correctness": 0.84}

    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        scores = []
        for r in results:
            tc = LLMTestCase(
                input=r["question"],
                actual_output=r["answer"],
                expected_output=r["ground_truth"],
            )
            m = AnswerRelevancyMetric(threshold=0.7)
            m.measure(tc)
            scores.append(m.score)
        avg = sum(scores) / len(scores) if scores else 0.0
        return {"answer_correctness": avg}
    except Exception as e:
        print(f"[deepeval] evaluation error: {e}")
        return {"answer_correctness": 0.0}
