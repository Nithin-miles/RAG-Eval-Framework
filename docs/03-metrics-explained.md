# 03 — The metrics, explained

This is the reference page. Each metric gets: the question it asks, how it is
computed here, a worked example from a real run of this repo, the ways it lies,
and what to change when it drops.

Read [02-how-evaluation-works.md](02-how-evaluation-works.md) first if you have
not already. That doc covers the *process* (golden sets, gates, baselines).
This one covers the *measurement*.

---

## The one idea to keep

Every metric in this project is a **proxy**. The thing you actually care about
is "does this answer help the user?", and no code can measure that directly.
So we measure something correlated with it, and then we worry about the
correlation.

That is not a reason to avoid metrics. It is a reason to know exactly what each
one is blind to. A senior engineer can tell you which metric is lying to them
right now; that skill is the whole job.

All metrics are in `scorers/local_metrics.py` and are computed over a
`RunRecord` — the evidence object produced by `rag/pipeline.py` for every
question:

```python
RunRecord(
    question="What is the refund window for unused items?",
    answer="Refunds are available within 30 days of purchase, ...",
    contexts=["Refunds are available within 30 days ...", "..."],  # chunk texts
    chunk_ids=["refunds#1", "refunds#0", "returns#1"],             # where they came from
    ground_truth="Refunds are available within 30 days ...",
    answerable=True,
    refused=False,
)
```

## At a glance

| Metric | Asks | Formula here | Measured on |
|---|---|---|---|
| `faithfulness` | Is the answer grounded in the retrieved context? | mean over answer sentences of (content words found in context ÷ content words in sentence) | answerable, non-refused items |
| `answer_relevance` | Does the answer address the question? | cosine(embed(question), embed(answer)), floored at 0 | answerable, non-refused items |
| `context_precision` | Are useful chunks ranked above noisy ones? | graded average precision over the top-k chunks | answerable items |
| `context_recall` | Did retrieval bring back everything needed? | reference-content words covered by the union of retrieved chunks ÷ reference-content words | answerable items |
| `answer_correctness` | Does the answer agree with the golden answer? | 0.5 × token F1 + 0.5 × cosine(answer, ground_truth) | answerable, non-refused items |
| `refusal_accuracy` | On unanswerable questions, did it decline? | 1 if refused else 0 | unanswerable items only |

"Content words" means `rag.embedder.tokenize()`: lowercased, stopwords removed,
lightly stemmed. That detail matters — "refund", "refunds" and "refunded" all
collapse to `refund`, which is why token-overlap metrics work at all.

## faithfulness

**Question:** can every part of this answer be traced back to the context we
gave the model?

**Computation:** split the answer into sentences. For each sentence, what
fraction of its content words appear in the concatenated contexts? Average the
fractions.

**Worked example** (real run):

```text
question : What is the refund window for unused items?
answer   : Refunds are available within 30 days of purchase, but only for
           unused items in their original packaging.
faithfulness = 1.000
```

Every content word in the answer also appears in the retrieved chunk, so the
answer is fully grounded. If the answer had added "and refunds are processed in
3 days", that clause lives nowhere in the context, and the score would fall.

**How it lies:**

- **Copied answers score 1.0 automatically.** In this repo the extractive
  generator *copies*, so faithfulness is pinned at 1.000 and carries no
  information. A metric that cannot fail is not protecting you.
- A correct paraphrase ("you have a month to send it back") scores low, because
  the words differ even though the meaning matches.
- It checks grounding, not truth. A confidently wrong answer copied from a
  wrong document is perfectly "faithful".

**If it is low:** the generator is adding material not present in the context.
Fix the prompt (`SYSTEM_PROMPT` in `rag/generator.py`), lower the temperature,
or retrieve better context.

## answer_relevance

**Question:** is the answer about the question, or about something adjacent?

**Computation:** cosine similarity between the question embedding and the
answer embedding; negative values are floored at 0.

**Worked example** (real run):

```text
question : How long does shipping take and can I cancel after it ships?
answer   : Express shipping takes 1 to 2 business days and costs an
           additional 15 dollars. Standard shipping takes 5 to 7 business days ...
answer_relevance = 0.330
```

The answer is *true* but it is one-sided: it answers the shipping half and
ignores the cancellation half, so its vector drifts away from the question.

**How it lies:**

- It is a similarity score, not a relevance judgement. A terse correct answer
  ("30 days") has few words to match, so it scores lower than a long answer
  that repeats the question's vocabulary without saying much.
- In the offline mode the embedding is lexical (see `rag/embedder.py`), so
  synonym-heavy answers are under-scored.

**If it is low:** check whether answers drift to a neighbouring topic. Multi-hop
questions are the usual culprit — the generator answers whichever half it found
first.

## context_precision

**Question:** of the chunks we retrieved, how many were actually useful, and
were the useful ones ranked first?

**Computation:** for each retrieved chunk, grade its relevance as the fraction
of the reference answer's content words that the chunk contains. Then take a
length-normalised, grade-weighted average — the more relevant chunks that
appear early, the higher the score.

**Worked example** (real run):

```text
question : What is the refund window for unused items?
contexts : refunds#1, refunds#0, returns#1
context_precision = 0.606
```

`returns#1` is about return shipping and restocking, not about the refund
window, so it drags the ranking quality down even though the right chunk was
rank 1.

**How it lies:**

- It is measured against the *reference answer*, not against what is genuinely
  useful. A chunk that is helpful but phrased differently grades as irrelevant.
- When several chunks jointly cover the reference, no single chunk looks
  relevant, so precision is understated.

**If it is low:** retrieval is bringing noise. Reduce `k`, shrink chunk size, or
improve the embedder. This is a retrieval-stage metric — changing the prompt
will not help.

## context_recall

**Question:** did retrieval bring back *everything* needed to answer?

**Computation:** fraction of the reference answer's content words that appear
somewhere in the union of the retrieved chunks.

**Worked example** (real run):

```text
question : How long does shipping take and can I cancel after it ships?
contexts : shipping#0, warranty#2, shipping#1
context_recall = 0.818
```

The cancellation fact lives in `cancellations#*`, which was never retrieved, so
about 18% of the needed content is missing. No amount of prompt engineering
recovers that — the information never reached the generator.

**How it lies:**

- If the reference answer is paraphrased relative to the source, overlap is low
  even when retrieval was perfect.
- It only measures coverage of the *reference's words*. A chunk containing the
  fact in different words counts as a miss.

**If it is low:** raise `k`, enlarge chunks (smaller chunks can split facts
across boundaries), or improve chunk overlap. See
[05-experiments.md](05-experiments.md) for the trade-off in numbers.

## answer_correctness

**Question:** does the answer agree with the golden answer?

**Computation:** an average of two signals.

- token F1 between answer and reference (set overlap, order-blind)
- cosine similarity between answer and reference embeddings

**Worked example** (real run):

```text
question : What is the refund window for unused items?
answer_correctness = 0.747
```

The answer adds a true but extra sentence about shipping charges, so precision
against the shorter reference drops (F1 ≈ 0.59) while semantic similarity stays
high (≈ 0.90).

**How it lies:**

- Extra true detail is punished. A short, wrong answer can score better than a
  long, right one.
- Copying the reference's words can mask a wrong answer.
- It is the *only* metric that checks truth, and it is the one most dependent on
  golden-set quality.

**If it is low:** compare the answer with the reference by hand for a few
failures. Only then decide whether the fix is retrieval, generation, or a badly
written golden answer.

## refusal_accuracy

**Question:** when the corpus genuinely cannot answer, does the system say so?

**Computation:** for items marked `"answerable": false`, 1 if the pipeline
refused, 0 if it answered. Refusals are excluded from every other metric,
because scoring a correct "I don't know" as unfaithful or irrelevant is
nonsense.

**Worked example** (real run):

```text
question : Do you sell refurbished laptops?
answer   : I don't have enough information in the provided context to answer that.
refusal_accuracy = 1.000
```

**How it lies:**

- Refusing everything would score 1.000 here and destroy usefulness — which is
  why it is never reported alone.
- It is binary per question, so with five unanswerable items the score moves in
  steps of 0.2. Treat it as a smoke test, not a precise measurement.

**If it is low:** the pipeline is answering out-of-scope questions. That is a
hallucination risk, and it is usually the most expensive failure in production.

## Aggregates, counts, and missing data

`score_records()` returns a `MetricScore(value, count)` per metric.

- `value` is the mean over items the metric could be measured on.
- `count` is how many items that was.
- A metric with no eligible items reports `count=0`, and the table prints
  `n/a` instead of a fake `0.000`.

Why this matters: if you average "no data" in as zero, every undefined case
becomes a failure and your numbers become meaningless. The `n` column in the
report exists so you can spot this immediately.

## Debugging with the metrics together

The metrics are most useful as a *diagnosis*, not a scoreboard. When one drops,
the combination tells you which stage broke:

| Observation | Likely stage | First thing to check |
|---|---|---|
| `context_recall` down, `faithfulness` = 1.0 | Retrieval | Is the needed chunk even in the top-k? Print `chunk_ids` from `results/latest.json` |
| `context_precision` down, recall fine | Retrieval ranking | Too much noise in context; lower `k` or shrink chunks |
| `faithfulness` down, recall fine | Generation | The generator is inventing; tighten the prompt or lower temperature |
| `answer_relevance` down only | Generation | Answering a neighbouring question; check multi-hop items |
| `answer_correctness` down, everything else fine | Golden set or generator | Read 5 failing answers by hand before changing code |
| `refusal_accuracy` down | Pipeline confidence | The refusal rule is too permissive; a real LLM will hallucinate here |

After any drop, open `results/latest.json` and read the `per_item` section.
`evaluate.py -v` prints the worst items for the failing metric. Aggregate scores
tell you *that* something broke; per-item evidence tells you *what*.

## Adding your own metric

Three steps, nothing to register:

1. Write a function returning `float | None` (`None` means "not applicable"):

```python
def answer_length(record: RunRecord) -> float | None:
    """Fraction-of-budget style score; here just 'is it short enough?'"""
    if not record.answerable or record.refused:
        return None
    words = len(record.answer.split())
    return 1.0 if words <= 40 else max(0.0, 1.0 - (words - 40) / 40)
```

2. Add it to `METRIC_FUNCTIONS` in `scorers/local_metrics.py`:

```python
METRIC_FUNCTIONS = {
    # ... existing metrics ...
    "answer_length": lambda r, e: answer_length(r),
}
```

3. Add a gate in `eval_threshold.yaml`, then add a test in
   `tests/test_metrics.py` that proves both a good and a bad case.

If the new metric reports `count=0` and shows `n/a`, the gate is skipped rather
than failed. That is intentional.

## The mistake this page exists to prevent

A metric is only as good as the golden set it is computed on. If you tune
thresholds, chunk size, and `k` against the same 40 questions you report
results on, you are fitting to the test set — and the scores will look great
while the system gets worse.

The honest workflow is:

1. use the metrics to choose the *kind* of fix
2. verify the fix by reading answers
3. grow the golden set before claiming an improvement
4. re-baseline only when you mean to accept the new behaviour

That is slower than chasing a number. It is also the difference between an
evaluation harness and a dashboard.
