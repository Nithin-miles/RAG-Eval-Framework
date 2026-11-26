# RAG-Eval-Framework

> A beginner-friendly, **offline-first** evaluation harness for RAG (Retrieval-Augmented Generation) pipelines — the tests you write when the "correct answer" is a paragraph, not an `assertEqual`.

[![RAG Evaluation](https://github.com/Nithin-miles/RAG-Eval-Framework/actions/workflows/eval.yml/badge.svg)](https://github.com/Nithin-miles/RAG-Eval-Framework/actions/workflows/eval.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

No API key. No model download. No network. Clone it and run it in under a minute.

---

## What this is

Two things in one small, readable repository:

1. **A toy RAG pipeline** — 8 markdown documents, chunked, embedded, searched, and answered. Deliberately simple so you can read every line.
2. **An evaluation harness** — 40 golden questions, 6 metrics, thresholds that fail the build, and per-question evidence you can debug with.

The interesting part is #2. Anyone can build a RAG demo; the hard question is *"did my change make it better or worse?"* This repo answers that question with code instead of vibes, and shows the whole reasoning behind the answer.

## Why this exists

LLM output is non-deterministic. `assert response == expected` breaks forever. But shipping a RAG app with no evaluation is shipping blind — you change a prompt, and quietly break three answers nobody re-checked.

This project treats evaluation as a **test suite**:

- every answer is scored against a golden set
- every score is compared to a threshold
- the process exits non-zero when quality drops
- CI runs it on every push, so a regression is a red build, not a support ticket

## Quickstart

```bash
git clone https://github.com/Nithin-miles/RAG-Eval-Framework.git
cd RAG-Eval-Framework
python -m pip install -r requirements.txt
python evaluate.py
```

Expected output:

```text
=== vector retriever | mode=local | k=3 | chunk=240 ===
    questions: 40 (35 answerable, 5 unanswerable, 5 refused)

    metric                 score      gate     n  status
    ------------------------------------------------------
    faithfulness           1.000      0.90    35  PASS
    answer_relevance       0.484      0.42    35  PASS
    context_precision      0.807      0.75    35  PASS
    context_recall         0.981      0.90    35  PASS
    answer_correctness     0.832      0.70    35  PASS
    refusal_accuracy       1.000      0.80     5  PASS

    [OK] all gated metrics passed
    results written to results/latest.json
```

Then the two commands that make it a learning tool rather than a demo:

```bash
python evaluate.py --compare      # keyword retriever vs vector retriever
python evaluate.py --check-baseline   # fail if quality regressed vs baseline.json
python -m pytest                  # 44 tests on the harness itself
```

## How it works

```text
data/corpus/*.md
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│  chunker.py │────▶│ retriever.py │────▶│ generator.py │────▶│  answer +   │
│ sentences → │     │ top-k chunks │     │ extractive   │     │  contexts   │
│   chunks    │     │ by similarity│     │ or an LLM    │     │             │
└─────────────┘     └──────────────┘     └──────────────┘     └──────┬──────┘
                           ▲                                        │
                    embedder.py                                     ▼
              (hashing vectors, offline)                    scorers/local_metrics.py
                                                             6 metrics per question
                                                                     │
                                                                     ▼
                                                            evaluate.py → gate → exit code
                                                                     │
                                                              results/latest.json
                                                              (aggregates + per-item evidence)
```

Every stage is swappable behind a small interface, which is what makes the experiments honest: change *one* variable, keep the golden set and metrics fixed, compare scores.

## Project layout

| Path | What it is |
|---|---|
| `rag/` | The pipeline under test: corpus → chunks → embeddings → retrieval → generation |
| `rag/chunker.py` | Sentence-aware chunking with overlap |
| `rag/embedder.py` | Deterministic hashing embedder (offline stand-in for a real embedding model) |
| `rag/retriever.py` | `KeywordRetriever` (naive baseline) and `VectorRetriever` |
| `rag/generator.py` | `ExtractiveGenerator` (offline) and `OpenAIGenerator` (optional) |
| `scorers/local_metrics.py` | Six offline metrics, plus the reasoning about when each one lies |
| `scorers/ragas_scorer.py` | Optional RAGAS adapter (lazy import, clear failure) |
| `scorers/deepeval_scorer.py` | Optional DeepEval adapter (uses the *correct* metric names) |
| `evaluate.py` | Runner: scores, thresholds, baselines, exit codes |
| `golden_set.json` | 40 questions: 35 answerable + 5 deliberately unanswerable |
| `eval_threshold.yaml` | The quality gates, and how to choose them |
| `baseline.json` | Committed reference scores for `--check-baseline` |
| `tests/` | 44 tests — most of them guard the metrics themselves |
| `docs/` | The guided explanation, in reading order |
| `LEARN.md` | A 7-session study plan with exercises |

## The metrics

Every one is a **proxy**, and the docs are explicit about how each one lies. That is the most important thing this project teaches.

| Metric | The question it answers | How it is computed offline | Where it lies |
|---|---|---|---|
| `faithfulness` | Is the answer supported by the retrieved context? | Fraction of the answer's content words present in the context | Copied answers score 1.0 regardless of relevance; correct paraphrases score too low |
| `answer_relevance` | Does the answer actually address the question? | Cosine similarity between question and answer embeddings | Short or terse answers are penalised; question echo scores high |
| `context_precision` | Are useful chunks ranked above noisy ones? | Average precision, graded by reference coverage | Understates when several chunks jointly cover the answer |
| `context_recall` | Did retrieval bring back everything needed? | Fraction of the reference answer covered by retrieved chunks | Paraphrased references lose overlap even when retrieval was right |
| `answer_correctness` | Does the answer agree with the golden answer? | 50% token F1 + 50% embedding similarity | Rewording costs points; copying the reference's words can hide a wrong answer |
| `refusal_accuracy` | On unanswerable questions, did it decline? | 1 if the pipeline refused, else 0 — only for `"answerable": false` | A real LLM can refuse for the wrong reason and still score 1 |

Aggregate scores come with an `n` column. A metric with nothing to measure reports `n/a` instead of `0`, because averaging "no data" in as a failure quietly poisons every number.

## Results

Baseline run — vector retriever, `k=3`, `chunk_size=240`, offline metrics:

| Metric | Score | Gate |
|---|---|---|
| faithfulness | 1.000 | 0.90 |
| answer_relevance | 0.484 | 0.42 |
| context_precision | 0.807 | 0.75 |
| context_recall | 0.981 | 0.90 |
| answer_correctness | 0.832 | 0.70 |
| refusal_accuracy | 1.000 | 0.80 |

**Retriever comparison** (`python evaluate.py --compare`):

| Metric | keyword | vector | delta |
|---|---|---|---|
| faithfulness | 1.000 | 1.000 | +0.000 |
| answer_relevance | 0.482 | 0.484 | +0.002 |
| context_precision | 0.769 | 0.807 | +0.038 |
| context_recall | 0.979 | 0.981 | +0.002 |
| answer_correctness | 0.814 | 0.832 | +0.018 |
| refusal_accuracy | 1.000 | 1.000 | +0.000 |

That gap is small, and it is reported honestly: **both** retrievers are lexical, and the offline hashing embedder only approximates semantic similarity. A real embedding model is expected to widen the gap on paraphrased questions. This repo gives you the harness to measure that, rather than a claim that vector search is magic.

**Parameter sweep** (vector retriever):

| Config | faithfulness | answer_relevance | context_precision | context_recall | answer_correctness |
|---|---|---|---|---|---|
| `k=1` | 1.000 | 0.467 | 0.831 | 0.831 | 0.805 |
| `k=3` | 1.000 | 0.484 | 0.807 | 0.981 | 0.832 |
| `k=5` | 1.000 | 0.487 | 0.744 | 1.000 | 0.815 |
| `chunk=120` | 1.000 | 0.485 | 0.809 | 0.950 | 0.838 |
| `chunk=240` | 1.000 | 0.484 | 0.807 | 0.981 | 0.832 |
| `chunk=480` | 1.000 | 0.486 | 0.864 | 1.000 | 0.834 |

The precision/recall trade-off shows up exactly where the theory says it should: raising `k` recovers more of the answer (`context_recall` 0.831 → 1.000) while diluting ranking quality (`context_precision` 0.831 → 0.744). Chunk size does *not* show a clean monotonic trend on a corpus this small — 480-character chunks score well here because a few large chunks are enough to cover short documents, and the extractive generator still picks the right sentence. On a real corpus with long documents that would not hold, which is precisely why you measure instead of assuming. `docs/05-experiments.md` walks through reproducing and interpreting every row.

## Quality gates and CI

`evaluate.py` exits `0` only if every gated metric is at or above its threshold, and — with `--check-baseline` — has not regressed against the committed baseline. `.github/workflows/eval.yml` runs the unit tests and the gates on every push and pull request, then uploads `results/latest.json` as a build artifact.

That is the whole idea in one sentence: **a quality regression should be a red build, not an opinion.**

## Optional: real LLMs and LLM-judge metrics

The offline path is the default because it is free, fast, and deterministic. If you want the real thing:

```bash
python -m pip install -r requirements-llm.txt
export OPENAI_API_KEY=...          # or put it in .env (see .env.example)

python evaluate.py --generator openai           # real answers
python evaluate.py --mode ragas                 # RAGAS LLM-judge metrics
python evaluate.py --mode deepeval              # DeepEval LLM-judge metrics
```

Two honest warnings, both learned while building this:

- **These libraries break between versions.** This repo's earlier version silently returned **fake mock scores** when a key was missing — a green CI that proved nothing. That behaviour is gone: the adapters either measure or raise `ScorerUnavailable` with an actionable message, and the runner prints the reason and falls back to offline metrics *visibly*.
- **The adapters are version-tolerant, not version-proof.** RAGAS's metric API has moved between `ragas.metrics` and `ragas.metrics.collections`; DeepEval 4.x has no `AnswerCorrectnessMetric` (the previous code mislabelled an answer-relevancy score as answer-correctness — a bug that looks like a working metric). Pin your versions if you need reproducibility.

## Honest limitations

- **The embedding model is lexical.** `HashingEmbedder` hashes words and bigrams. It cannot match synonyms, so it is closer to keyword search than to a real embedding model. Swapping in `sentence-transformers` or an API embedder is the single highest-value change you can make.
- **`faithfulness` is pinned at 1.000.** The extractive generator copies sentences out of the context, so of course it is grounded. A metric that cannot fail carries no information — this is the clearest demonstration of why the generator matters, and it becomes informative the moment you use `--generator openai`.
- **40 questions is a small golden set.** Fine for learning the mechanics, too small for real statistical claims. Real systems need hundreds, with paraphrase clusters and adversarial cases.
- **Offline metrics are not RAGAS.** They are transparent approximations that run without a judge model. Treat the numbers as *comparable between runs of this repo*, not as comparable to published benchmarks.
- **Thresholds are a policy, not a law.** The values in `eval_threshold.yaml` were chosen with a margin below measured performance so the build is not flaky — and that also means they will not catch every regression.

## Learning path

Read in this order; each doc stands alone but later ones assume the earlier ones.

| # | Document | What you get |
|---|---|---|
| 1 | [docs/01-what-is-rag.md](docs/01-what-is-rag.md) | The pipeline, and why each stage exists |
| 2 | [docs/02-how-evaluation-works.md](docs/02-how-evaluation-works.md) | Golden sets, gates, baselines, troubleshooting |
| 3 | [docs/03-metrics-explained.md](docs/03-metrics-explained.md) | Each metric with a worked example and its failure modes |
| 4 | [docs/04-tour-of-the-code.md](docs/04-tour-of-the-code.md) | File-by-file walkthrough, one question traced end to end |
| 5 | [docs/05-experiments.md](docs/05-experiments.md) | Hands-on labs: retriever, `k`, chunk size, adding a metric |
| — | [LEARN.md](LEARN.md) | A 7-session study plan with exercises and self-checks |

## Requirements

- Python 3.10+
- `numpy`, `PyYAML`, `pytest` — that is all for the default path
- Optional: `openai`, `ragas`, `deepeval`, `datasets`, `python-dotenv` for the LLM paths

## License

MIT — see [LICENSE](LICENSE).
