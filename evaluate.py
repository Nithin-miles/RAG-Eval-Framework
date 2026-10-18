"""Entry point: run the full RAG evaluation suite and gate on thresholds."""
import argparse
import json
import sys

import yaml
from dotenv import load_dotenv

from rag.simple_rag import SimpleRAG
from scorers.ragas_scorer import score_ragas
from scorers.deepeval_scorer import score_deepeval

load_dotenv()


def load_thresholds(path="eval_threshold.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run(dataset_path="golden_set.json", mock=False):
    with open(dataset_path) as f:
        dataset = json.load(f)

    rag = SimpleRAG()
    results = []
    for item in dataset:
        answer, contexts = rag.query(item["question"])
        results.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })

    scores = {}
    scores.update(score_ragas(results, mock=mock))
    scores.update(score_deepeval(results, mock=mock))

    thresholds = load_thresholds()
    failed = []
    print("\n--- Evaluation Results ---")
    for metric, threshold in thresholds.items():
        score = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"[{status}] {metric:20s} {score:.3f}  (threshold {threshold})")
        if score < threshold:
            failed.append(metric)

    if failed:
        print(f"\n[FAIL] {len(failed)} metric(s) below threshold: {failed}")
        sys.exit(1)

    print("\n[OK] All metrics passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="golden_set.json")
    parser.add_argument("--mock", action="store_true",
                        help="Run without API key using deterministic scores")
    args = parser.parse_args()
    run(args.dataset, mock=args.mock)
