"""Tests for the runner: gating, regression detection, and the CLI contract.

The CLI contract is the part that matters in CI: exit code 0 means "quality is
acceptable", anything else means "do not merge". If these tests pass but the
contract is wrong, the whole project is decorative.
"""

import json

import evaluate


def test_gate_reports_breaches():
    scores = {"faithfulness": {"value": 0.5, "count": 10}}
    thresholds = {"faithfulness": 0.9}
    assert evaluate.gate(scores, thresholds) == [("faithfulness", 0.5, 0.9)]


def test_gate_ignores_metrics_with_no_measurements():
    scores = {"refusal_accuracy": {"value": 0.0, "count": 0}}
    thresholds = {"refusal_accuracy": 0.9}
    assert evaluate.gate(scores, thresholds) == []


def test_gate_passes_on_equal_value():
    scores = {"faithfulness": {"value": 0.9, "count": 5}}
    assert evaluate.gate(scores, {"faithfulness": 0.9}) == []


def test_regression_detects_a_real_drop():
    baseline = {"scores": {"faithfulness": {"value": 0.9, "count": 30}}}
    current = {"faithfulness": {"value": 0.8, "count": 30}}
    assert evaluate.regression_failures(current, baseline) == [("faithfulness", 0.8, 0.9)]


def test_regression_tolerates_noise():
    baseline = {"scores": {"faithfulness": {"value": 0.90, "count": 30}}}
    current = {"faithfulness": {"value": 0.895, "count": 30}}
    assert evaluate.regression_failures(current, baseline, tolerance=0.01) == []


def test_cli_exits_zero_when_gates_pass(run_dir):
    out = run_dir / "run.json"
    assert evaluate.main(["--json", str(out)]) == 0
    assert out.exists()


def test_cli_writes_scores_and_per_item_evidence(run_dir):
    out = run_dir / "run.json"
    evaluate.main(["--json", str(out)])
    payload = json.loads(out.read_text())
    assert payload["counts"]["questions"] > 0
    assert payload["scores"]["faithfulness"]["count"] > 0
    assert len(payload["per_item"]) == payload["counts"]["questions"]
    assert "answer" in payload["per_item"][0]


def test_cli_exits_nonzero_when_a_gate_fails(run_dir):
    thresholds = run_dir / "impossible.yaml"
    thresholds.write_text("faithfulness: 1.5\n", encoding="utf-8")
    out = run_dir / "run.json"
    assert evaluate.main(["--thresholds", str(thresholds), "--json", str(out)]) == 1


def test_cli_exits_two_on_missing_dataset(run_dir):
    assert evaluate.main(["--dataset", str(run_dir / "nope.json")]) == 2


def test_check_baseline_passes_for_the_committed_baseline(run_dir):
    assert evaluate.main(["--check-baseline", "--json", str(run_dir / "run.json")]) == 0
