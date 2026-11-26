#!/usr/bin/env python3
"""Run the evaluation suite and gate on quality thresholds.

Typical use
-----------
    python evaluate.py                    # offline run, vector retriever, gates applied
    python evaluate.py --retriever keyword # same questions, naive baseline
    python evaluate.py --compare           # both, side by side (the interesting one)
    python evaluate.py --check-baseline    # fail if quality regressed vs baseline.json
    python evaluate.py --mode ragas        # grade with RAGAS instead (needs a key)

Exit code is 0 only if every gated metric met its threshold (and, with
``--check-baseline``, did not regress). That is the whole point: this script is
designed to be the thing that turns an eval score into a red or green CI check.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rag.generator import ExtractiveGenerator, OpenAIGenerator
from rag.pipeline import RAGPipeline, build_chunks, load_dataset, run_dataset
from rag.retriever import build_retriever
from scorers import ScorerUnavailable
from scorers.local_metrics import MetricScore, per_item_scores, score_records

# Optional convenience: honour a local .env file if python-dotenv is installed.
# The offline path does not need it, so a missing dependency must not break it.
try:  # pragma: no cover - trivial environment plumbing
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# -- Defaults: one place to change the experiment ---------------------------
DEFAULT_K = 3
DEFAULT_CHUNK_SIZE = 240
DEFAULT_OVERLAP = 60
DEFAULT_RETRIEVER = "vector"
DEFAULT_GENERATOR = "extractive"
DEFAULT_MODE = "local"
DEFAULT_THRESHOLDS = "eval_threshold.yaml"
DEFAULT_DATASET = "golden_set.json"
DEFAULT_CORPUS = "data/corpus"
BASELINE_PATH = "baseline.json"
RESULTS_DIR = "results"


# -- Config ----------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a RAG pipeline against a golden set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--retriever", choices=["vector", "keyword"], default=DEFAULT_RETRIEVER)
    parser.add_argument("--generator", choices=["extractive", "openai"], default=DEFAULT_GENERATOR)
    parser.add_argument("--mode", choices=["local", "ragas", "deepeval"], default=DEFAULT_MODE,
                        help="local = offline metrics; ragas/deepeval = LLM judge (needs OPENAI_API_KEY)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="chunks retrieved per question")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--json", default=None, help="where to write the full result (default: results/latest.json)")
    parser.add_argument("--check-baseline", action="store_true", help="fail on regression vs baseline.json")
    parser.add_argument("--compare", action="store_true", help="run keyword and vector retrievers and compare")
    parser.add_argument("-v", "--verbose", action="store_true", help="show per-question failures")
    return parser.parse_args(argv)


def load_thresholds(path: str = DEFAULT_THRESHOLDS) -> dict[str, float]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping of metric -> threshold")
    return {str(k): float(v) for k, v in raw.items()}


# -- Running ---------------------------------------------------------------
def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    chunks = build_chunks(
        args.corpus, max_chars=args.chunk_size, overlap_chars=args.overlap
    )
    retriever = build_retriever(args.retriever, chunks)

    if args.generator == "openai":
        generator = OpenAIGenerator()
    else:
        generator = ExtractiveGenerator()

    return RAGPipeline(retriever, generator, k=args.k)


def run_suite(args: argparse.Namespace, dataset: list[dict]) -> dict:
    """Run the pipeline and score it. Returns a JSON-serialisable dict."""
    pipeline = build_pipeline(args)
    records = run_dataset(pipeline, dataset)

    scores = score_records(records)
    used_mode = "local"

    if args.mode in ("ragas", "deepeval"):
        try:
            if args.mode == "ragas":
                from scorers.ragas_scorer import score_with_ragas as judge
            else:
                from scorers.deepeval_scorer import score_with_deepeval as judge

            library_scores = judge(records)
            # Library scores replace the offline proxies for the metrics it
            # returns; anything it does not return (refusal_accuracy) stays local.
            scores.update(library_scores)
            used_mode = args.mode
        except ScorerUnavailable as exc:
            print(f"[warn] {args.mode} scorer unavailable - falling back to offline metrics.")
            print(f"       reason: {exc}")
            used_mode = "local (fallback)"

    return {
        "config": {
            "retriever": args.retriever,
            "generator": args.generator,
            "mode": used_mode,
            "k": args.k,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "dataset": args.dataset,
            "corpus": args.corpus,
        },
        "counts": {
            "questions": len(records),
            "answerable": sum(1 for r in records if r.answerable),
            "unanswerable": sum(1 for r in records if not r.answerable),
            "refused": sum(1 for r in records if r.refused),
        },
        "scores": {name: score.as_dict() for name, score in scores.items()},
        "per_item": per_item_scores(records),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# -- Reporting -------------------------------------------------------------
def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_table(
    result: dict,
    thresholds: dict[str, float],
) -> tuple[list[str], list[tuple[str, float | None, float]]]:
    """Render the score table; return lines plus the list of breaches."""
    lines = [
        "",
        f"=== {result['config']['retriever']} retriever | "
        f"mode={result['config']['mode']} | k={result['config']['k']} | "
        f"chunk={result['config']['chunk_size']} ===",
        f"    questions: {result['counts']['questions']} "
        f"({result['counts']['answerable']} answerable, "
        f"{result['counts']['unanswerable']} unanswerable, "
        f"{result['counts']['refused']} refused)",
        "",
        f"    {'metric':<20} {'score':>7}   {'gate':>7}   {'n':>3}  status",
        "    " + "-" * 54,
    ]

    breaches: list[tuple[str, float | None, float]] = []
    for name, threshold in thresholds.items():
        score = result["scores"].get(name)
        value = score["value"] if score else None
        count = score["count"] if score else 0

        if count == 0 or value is None:
            status = "n/a"
        elif value >= threshold:
            status = "PASS"
        else:
            status = "FAIL"
            breaches.append((name, value, threshold))

        lines.append(
            f"    {name:<20} {_fmt(value):>7}   {threshold:>7.2f}   {count:>3}  {status}"
        )

    lines.append("")
    return lines, breaches


def gate(
    scores: dict[str, dict],
    thresholds: dict[str, float],
) -> list[tuple[str, float | None, float]]:
    """Return every metric that is below its threshold."""
    breaches = []
    for name, threshold in thresholds.items():
        score = scores.get(name)
        if not score or score["count"] == 0:
            continue  # nothing was measured, so there is nothing to gate
        if score["value"] < threshold:
            breaches.append((name, score["value"], threshold))
    return breaches


def regression_failures(
    scores: dict[str, dict],
    baseline: dict,
    tolerance: float = 0.01,
) -> list[tuple[str, float, float]]:
    """Return metrics that fell meaningfully below the committed baseline."""
    failures: list[tuple[str, float, float]] = []
    base_scores = baseline.get("scores", {})
    for name, base in base_scores.items():
        current = scores.get(name)
        if not current or current["count"] == 0 or base.get("count", 0) == 0:
            continue
        if current["value"] < base["value"] - tolerance:
            failures.append((name, current["value"], base["value"]))
    return failures


def print_worst(result: dict, metric: str, limit: int = 3) -> None:
    """Show the lowest-scoring questions for one metric - where you start debugging."""
    rows = [r for r in result["per_item"] if r.get(metric) is not None]
    rows.sort(key=lambda r: r[metric])
    print(f"    lowest {metric}:")
    for row in rows[:limit]:
        print(f"      {row[metric]:.3f}  {row['question']}")
        print(f"             answer: {row['answer'][:110]}")
    print()


# -- Main ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
        thresholds = load_thresholds(args.thresholds)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}")
        return 2

    results = {}
    for retriever in (["keyword", "vector"] if args.compare else [args.retriever]):
        args.retriever = retriever
        results[retriever] = run_suite(args, dataset)

    primary = results[args.retriever]
    lines, breaches = render_table(primary, thresholds)
    print("\n".join(lines))

    if args.compare and len(results) == 2:
        print(f"    {'metric':<20} {'keyword':>9} {'vector':>9} {'delta':>9}")
        print("    " + "-" * 50)
        for name in thresholds:
            kw = results["keyword"]["scores"].get(name, {}).get("value")
            vec = results["vector"]["scores"].get(name, {}).get("value")
            delta = "n/a" if None in (kw, vec) else f"{vec - kw:+.3f}"
            print(
                f"    {name:<20} {_fmt(kw):>9} {_fmt(vec):>9} {delta:>9}"
            )
        print()

    exit_code = 0

    if breaches:
        exit_code = 1
        print(f"    [FAIL] {len(breaches)} metric(s) below threshold:")
        for name, value, threshold in breaches:
            print(f"           {name}: {value:.3f} < {threshold:.2f}")
        if args.verbose:
            print()
            print_worst(primary, breaches[0][0])
    else:
        print("    [OK] all gated metrics passed")

    if args.check_baseline:
        path = Path(BASELINE_PATH)
        if not path.exists():
            print(f"    [warn] {BASELINE_PATH} not found - skipping regression check")
        else:
            baseline = json.loads(path.read_text(encoding="utf-8"))
            regressions = regression_failures(primary["scores"], baseline)
            if regressions:
                exit_code = 1
                print(f"    [FAIL] regression vs {BASELINE_PATH}:")
                for name, current, base in regressions:
                    print(f"           {name}: {current:.3f} < baseline {base:.3f}")
            else:
                print(f"    [OK] no regression vs {BASELINE_PATH}")

    # Persist the full run: aggregates + per-question evidence.
    out_path = Path(args.json) if args.json else Path(RESULTS_DIR) / "latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = primary if not args.compare else {"primary": args.retriever, "runs": results}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"    results written to {out_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
