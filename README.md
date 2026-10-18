# RAG-Eval-Framework

> Automated evaluation pipeline for RAG (Retrieval-Augmented Generation) applications using RAGAS and DeepEval, integrated into CI/CD.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![RAGAS](https://img.shields.io/badge/RAGAS-evaluation-orange)](https://github.com/explodinggradients/ragas)
[![DeepEval](https://img.shields.io/badge/DeepEval-testing-green)](https://github.com/confident-ai/deepeval)

## TL;DR

A QA-first framework that tests RAG applications the way a unit test tests code. It measures faithfulness, answer relevance, context precision, and answer correctness - and fails the build if quality drops below threshold.

## Why this exists

LLM outputs are non-deterministic. Traditional assertions break. But shipping a RAG app without evaluation is shipping blind. This project treats evaluation as a test suite.

## Metrics

| Metric | What it catches | Tool |
|---|---|---|
| Faithfulness | Hallucination - answer not grounded in context | RAGAS |
| Answer Relevance | Off-topic or evasive answers | RAGAS |
| Context Precision | Retriever pulling noisy chunks | RAGAS |
| Answer Correctness | Semantic mismatch vs ground truth | DeepEval |

## Quick start

```bash
git clone https://github.com/Nithin-miles/RAG-Eval-Framework.git
cd RAG-Eval-Framework
pip install -r requirements.txt
cp .env.example .env
python evaluate.py --mock
```

Expected output:

```
[ragas] MOCK mode - returning synthetic scores
[deepeval] MOCK mode - returning synthetic score

--- Evaluation Results ---
[PASS] faithfulness         0.870  (threshold 0.85)
[PASS] answer_relevance     0.820  (threshold 0.8)
[PASS] context_precision    0.790  (threshold 0.75)
[PASS] answer_correctness   0.840  (threshold 0.8)

[OK] All metrics passed
```

## Project structure

```
evaluate.py            # entrypoint
scorers/               # RAGAS + DeepEval wrappers
rag/                   # minimal retriever + corpus
golden_set.json        # curated Q&A pairs
eval_threshold.yaml    # quality gates
.github/workflows/eval.yml
```

## Limitations & next steps

- Golden set is small; v2 will add synthetic test generation.
- No live production monitoring yet - planned: LangFuse integration.
