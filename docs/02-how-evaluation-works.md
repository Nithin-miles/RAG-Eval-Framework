# How evaluation works

You cannot improve a RAG pipeline by reading a handful of answers and nodding. You need a repeatable measurement that tells you whether a change helped or hurt. This document explains how this project turns the pipeline from `docs/01-what-is-rag.md` into numbers, how those numbers become pass/fail decisions, and how to debug a drop.

## Why you cannot just eyeball outputs

Eyeballing fails for three reasons.

- **The output space is too large.** A change to chunk size or retrieval can affect all 40 questions, and the changed answers may look fine in isolation.
- **LLM output is non-deterministic.** A real generator can phrase the same correct answer differently across runs, so a one-off comparison cannot separate a real regression from noise. The offline default is deterministic on purpose, but the corpus, chunking, and code still change over time.
- **Regressions are silent.** Nobody notices that a small refactor broke the answer to question 27 until a user does. A numeric gate notices.

An evaluation harness gives you a **regression signal**: run the same questions the same way, score them the same way, and fail when the numbers move in the wrong direction.

## The golden set

A **golden set** is a fixed list of questions with reference answers and metadata. This project keeps 40 items in `golden_set.json`. Of those, 35 are answerable from the corpus and 5 are deliberately unanswerable.

A minimal answerable item looks like this:

```json
{
  "question": "How long does standard shipping take?",
  "ground_truth": "Standard shipping takes 5 to 7 business days."
}
```

An unanswerable item adds `"answerable": false` and leaves the reference answer empty, for example `{"question": "Do you sell refurbished laptops?", "ground_truth": "", "answerable": false}`. The flag defaults to true. `refusal_accuracy` is computed only on those 5 items, and the other metrics skip them entirely. To build a set like this:

- **Real questions.** Start with questions users actually ask. If you invent all of them, you measure your imagination.
- **Paraphrases.** Ask the same thing two or three ways. Retrieval should survive wording changes, and different metrics will expose when it does not.
- **Edge cases.** Include questions whose answer spans two topics (for example shipping plus cancellation), questions about exceptions, and questions at the boundary of the corpus.
- **Unanswerable questions.** This is the step most tutorials skip. If every question is answerable, a pipeline that always answers can score well while hallucinating on real traffic. The 5 unanswerable items exist to catch exactly that.

The golden set is a curriculum, not a benchmark. With 40 questions you can compare versions against each other, but you cannot claim broad coverage. Treat each item as a test case you are willing to maintain.

## Offline metrics vs LLM-as-judge metrics

Both families are supported:

| | Offline (`--mode local`) | LLM judge (`--mode ragas` / `--mode deepeval`) |
| --- | --- | --- |
| Cost and repeatability | Free; same input, same score | API calls; can vary by model and prompt |
| Nuance | Word overlap and vector similarity only | Can judge meaning and support |
| CI | Runs with no key | Needs a key and network access |

The default is offline (`--mode local`). It needs only numpy and pyyaml, and it runs with zero API keys. That is why the project can make evaluation the normal, boring part of development instead of an occasional ritual.

The optional `--mode ragas` and `--mode deepeval` paths need `OPENAI_API_KEY` and the optional extras. If the judge library is unavailable, `evaluate.py` prints a warning and falls back to the offline metrics rather than crashing. A real LLM judge is still a proxy, just a more expensive one.

The six offline metrics are defined as follows. "Measured on" matters because each metric only scores items where it is meaningful.

| Metric | The question it asks | Measured on |
| --- | --- | --- |
| `faithfulness` | Is the answer supported by the retrieved context? | Answerable, not refused |
| `answer_relevance` | Does the answer address the question? | Answerable, not refused |
| `context_precision` | Are the relevant chunks ranked above the irrelevant ones? | Answerable |
| `context_recall` | Did retrieval bring back everything needed? | Answerable |
| `answer_correctness` | Does the answer agree with the reference answer? | Answerable, not refused |
| `refusal_accuracy` | Did the pipeline decline on questions it cannot answer? | Unanswerable only |

When a metric has nothing to score, it returns `None` and is excluded from the average. Each score is stored with a `count` so you can see how many items it was computed on. Averaging in "no data" as a failure would quietly poison every number.

## Metrics are proxies, not truth

Every metric here is a small formula standing in for a judgment you actually care about. None of them is ground truth. Knowing how each one lies is more useful than memorizing its name.

- `faithfulness` checks whether the answer's content words appear in the context. Copying a sentence from the context scores 1.0 even if that sentence answers a different question.
- `answer_relevance` compares question and answer embeddings. Very short answers can score low even when the sentence is exactly right.
- `context_precision` grades each chunk by how much of the reference answer it contains. When several chunks together cover the answer, no single chunk looks relevant, so precision is understated.
- `context_recall` counts reference words covered by retrieved chunks. A paraphrase with different words can score low even when retrieval was correct.
- `answer_correctness` blends token F1 with embedding similarity. A correct answer worded differently loses F1, and a wrong answer that reuses the reference's words can keep it.
- `refusal_accuracy` only checks whether a refusal happened, not whether the refusal was well phrased.

There is a deeper limitation: the offline embedder is lexical. Unlike a trained embedding model, it cannot match synonyms. Numbers from this project are useful for comparing versions of this project, not for comparing against a different RAG system.

## Thresholds are a policy

Thresholds turn a score into a decision. They live in `eval_threshold.yaml`:

```yaml
faithfulness: 0.90
answer_relevance: 0.42
context_precision: 0.75
context_recall: 0.90
answer_correctness: 0.70
refusal_accuracy: 0.80
```

`evaluate.py` compares each metric against its gate. A score at or above the threshold passes; below it fails. If the metric was measured on zero items, it shows `n/a` and is not gated, because there is nothing to decide.

Choosing a threshold honestly is a four-step process, described in the file's own comments:

1. Run the current system and note the scores.
2. Decide the floor you refuse to ship below.
3. Set the gate with a margin, about 0.05 to 0.10 below that floor.
4. When the system improves, raise the gate so it cannot silently regress.

The committed values match the offline baseline. Here is the comparison:

| Metric | Baseline | Gate |
| --- | --- | --- |
| `faithfulness` | 1.000 | 0.90 |
| `answer_relevance` | 0.484 | 0.42 |
| `context_precision` | 0.807 | 0.75 |
| `context_recall` | 0.981 | 0.90 |
| `answer_correctness` | 0.832 | 0.70 |
| `refusal_accuracy` | 1.000 | 0.80 |

The refusal gate is a special case. Only 5 items are unanswerable, so each one is worth 0.20. The 0.80 gate allows at most one miss. With a larger set you would choose a tighter number and mean it.

Thresholds are not universal constants. Swap in a real LLM generator, change the corpus, or change `k`, and the right gates change. A gate should be low enough to allow harmless variation and high enough that a real regression trips it.

## Baseline comparison catches regressions

An absolute threshold answers "is this good enough?" A baseline answers "did this get worse?" Those are different questions, and you want both.

`baseline.json` stores a previously accepted run: config, counts, scores, and a timestamp. With `--check-baseline`, the harness compares the current run to that file. Any metric that falls more than 0.01 below its baseline is a regression, even if it still passes its absolute gate.

That is the point. Suppose `context_recall` has a gate of 0.90 and a baseline of 0.981. A change drops recall to 0.95. The gate passes, so a threshold-only CI check goes green. The baseline check fails, because the system lost measurable quality. Without it, repeated small regressions can erode quality while every individual run stays "acceptable".

## Per-item evidence tells you what broke

An aggregate score tells you **that** something broke. The per-item table tells you **what**. Every run writes both to `results/latest.json` by default:

- `config` and `counts`: what ran, and how many questions were answerable, unanswerable, or refused.
- `scores` and `per_item`: each aggregate metric with its `count`, plus one record per question with the answer, refusal flag, retrieved chunk IDs, and every metric value.
- `run_at`: when the run happened.

For example, the shipping question from the first document has these per-item values in the committed run:

| Field | Value |
| --- | --- |
| `contexts` | `warranty#2`, `shipping#0`, `warranty#1` |
| `faithfulness` | 1.0 |
| `answer_relevance` | 0.5536 |
| `context_precision` | 0.619 |
| `context_recall` | 1.0 |
| `answer_correctness` | 0.6978 |

The aggregate `answer_correctness` is 0.832, so this item is below average while still looking fine on faithfulness. Without the per-item row you would know the average was pulled down but not which question did it. With the row, you can open `warranty#2` and `shipping#0` and see why the extractor added an extra, less relevant sentence.

The `-v` flag prints the lowest-scoring questions for the first breached metric, which is often enough to start debugging without opening the JSON.

## How this maps to CI

`evaluate.py` is designed to be a CI step. Its exit code is the signal:

| Exit code | Meaning |
| --- | --- |
| `0` | Every gated metric met its threshold, and with `--check-baseline`, nothing regressed. |
| `1` | At least one threshold was breached, or a baseline regression was found. |
| `2` | The dataset or threshold file could not be loaded. |

A typical job runs:

```bash
python evaluate.py --check-baseline
```

and lets the exit code fail the build. The full result is written every time, including when a gate fails, so the artifact is available for inspection. The harness does not need to be told which metric failed; the printed table shows each metric, its score, its gate, and PASS or FAIL.

The 44 pytest tests cover the pieces separately: chunking, retrieval, generation, metric math, and the CLI gate behavior. Running both the tests and the evaluation gives you fast unit feedback plus end-to-end quality feedback.

## If context_recall drops but faithfulness stays at 1.0

This is the most instructive failure in the project, because it looks contradictory until you remember what each metric measures.

`context_recall` asks whether the retrieved chunks contain the reference answer. `faithfulness` asks whether the generated answer is supported by those same chunks. If retrieval stops returning the right chunk but the generator still answers from what it did receive, every word it says is in context, so faithfulness stays at 1.0, while recall falls because the evidence needed for the reference answer never arrived.

Faithfulness cannot see what retrieval missed. It only inspects the answer against the context that exists.

So debug retrieval first, not generation:

- Open the per-item `contexts` for the affected questions and check which chunk IDs arrived.
- Try a larger `k` and see whether recall recovers.
- Compare `--retriever keyword` with `--retriever vector` to see whether the ranking method is the problem.
- Check whether the answer sits on a chunk boundary; a different `--chunk-size` or overlap may fix it.
- Confirm the reference answer is actually in the corpus and not just in your head.

Use this table to map other drops to a stage:

| Symptom | Likely stage | First checks |
| --- | --- | --- |
| `context_recall` down | Retrieval | Missing chunk IDs, `k`, chunk size, corpus content |
| `context_precision` down | Retrieval ranking | Relevant chunk pushed below noise; compare retrievers |
| `faithfulness` down | Generation | Answer contains words absent from context; check for outside knowledge |
| `answer_relevance` down | Generation or retrieval | Answer is off-topic; look at the actual sentence chosen |
| `answer_correctness` down | Generation | Answer disagrees with the reference; check per-item rows |
| `refusal_accuracy` down | Refusal behavior | Unanswerable item was answered; inspect the refusal path |

## What this evaluation cannot tell you

- A passing run does not mean the system is production-ready. Forty questions in a small corpus cannot cover real traffic.
- A metric can improve while quality gets worse. Optimizing the proxy is not the same as helping users.
- The deterministic generator makes the default scores stable, but it also means they do not predict how a real LLM will behave.
- LLM judges are also proxies. They add nuance and cost, not truth.
- Thresholds are a policy choice. Changing them to make CI green is not an improvement.

Use the harness to catch changes you did not intend, and read the per-item evidence before believing any aggregate number.
