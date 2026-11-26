# Experiments you can run

This is a lab, not a results page. Each experiment changes exactly **one** knob, runs the same 40 questions, and reports the numbers recorded for this repo. Run every command from the repository root (the folder containing `evaluate.py`), because the default paths are relative to it.

How to read a result: on 35 answerable questions a change of 0.01 is usually one or two questions. Treat anything under about 0.02 as "unchanged" until you have looked at the per-item table.

## 1. Setup and reading the table

The offline pipeline needs two third-party packages: `numpy` (vectors) and `PyYAML` (the threshold file). `requirements.txt` pins both, plus `pytest` for the test suite:

```bash
python -m pip install -r requirements.txt
```

Then run the suite with its defaults (vector retriever, k=3, chunk size 240, extractive generator, offline metrics):

```bash
python evaluate.py
```

```
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
```

What each part means:

| column | meaning |
| --- | --- |
| header line | the configuration actually used (`retriever \| mode \| k \| chunk`). If it does not match what you asked for, stop here. `mode=local (fallback)` means an LLM judge was requested, failed to import, and you are reading offline numbers |
| `questions: ...` | golden-set size, how many are deliberately unanswerable, how many the system refused |
| `metric` | one row per metric **with a gate** in `eval_threshold.yaml`; a computed metric with no gate never appears |
| `score` | mean over the items where that metric could be measured, three decimals |
| `gate` | the threshold from `eval_threshold.yaml`, two decimals |
| `n` | how many items went into the mean. This is the column beginners ignore and should not: `refusal_accuracy` has `n=5`, so one question flipping moves it by 0.200. `n=0` shows status `n/a` and is never gated |
| `status` | `PASS` when `score >= gate` (equality passes), `FAIL` below it, `n/a` when `n=0` |

After the table comes a summary line (`[OK] all gated metrics passed` or `[FAIL] n metric(s) below threshold`) and `results written to results/latest.json`. Every run overwrites that file, so pass `--json results/my-run.json` to keep an experiment. The exit code is what CI reads:

```bash
python evaluate.py; echo $LASTEXITCODE   # PowerShell
python evaluate.py; echo $?              # bash / zsh
```

`0` means every gate passed, `1` a breach or regression, `2` the configuration could not be loaded. The numbers above are the committed baseline in `baseline.json`; if your own run disagrees, something in the corpus, the chunker or the metrics changed after that baseline was recorded, and section 6 explains how to re-record it.

## 2. The retriever experiment

Same questions, k, chunking and generator — only the retrieval method changes.

```bash
python evaluate.py --compare
```

```
    metric                 keyword    vector     delta
    --------------------------------------------------
    faithfulness             1.000     1.000    +0.000
    answer_relevance         0.482     0.484    +0.002
    context_precision        0.769     0.807    +0.038
    context_recall           0.979     0.981    +0.002
    answer_correctness       0.814     0.832    +0.018
    refusal_accuracy         1.000     1.000    +0.000
```

The vector retriever wins or ties every row, but only one gap survives the noise floor: `context_precision` (+0.038), where hashing vectors put the useful chunk higher more often than raw word overlap does. Everything else is 0.000 to 0.018, which is one or two questions.

The reason is not mysterious: **both retrievers are lexical.** `HashingEmbedder` buckets words and word bigrams; it has no idea that "money back" and "refund" mean the same thing. It is a slightly better lexical matcher, not a semantic one. A real embedding model is expected to widen this gap on paraphrase questions — an experiment this offline repo cannot run, and the first item in section 9. Reporting +0.018 as evidence that "embeddings beat keywords" would be dishonest; the honest claim is that bigram hashing is a modest improvement and neither method understands synonyms. That is a limitation, not a win to brag about.

`--compare` gates only the vector run (it is the last retriever evaluated), so the keyword column is informational.

## 3. The k experiment

`k` is how many chunks reach the generator.

```bash
python evaluate.py --k 1 --json results/my-k1.json
python evaluate.py --k 3 --json results/my-k3.json
python evaluate.py --k 5 --json results/my-k5.json
```

| k | faithfulness | answer_relevance | context_precision | context_recall | answer_correctness |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.000 | 0.467 | 0.831 | 0.831 | 0.805 |
| 3 (default) | 1.000 | 0.484 | 0.807 | 0.981 | 0.832 |
| 5 | 1.000 | 0.487 | 0.744 | 1.000 | 0.815 |

This is the classic precision/recall trade-off in a toy system. As `k` rises, recall rises (0.831 → 0.981 → 1.000) because more chunks mean more chances that the one holding the answer is in the list; at k=5 recall is perfect. Precision falls (0.831 → 0.807 → 0.744) because the extra chunks are mostly filler, and average precision punishes relevant chunks that irrelevant ones are ranked above.

Retrieval is a ranked list and you are choosing where to stop reading. Stop early and the answer is precise but incomplete; stop late and it is complete but noisy. There is no free setting — which is why the default is 3.

Two honest footnotes: `answer_correctness` does not follow the trend (0.805, 0.832, 0.815) because the extractor only uses up to two sentences, so added chunks bring *competing* candidates that can displace the best one; and k=5 misses the shipped `context_precision` gate (0.744 vs 0.75) by 0.006, so a k=5 run fails the build even though recall is perfect.

## 4. The chunk-size experiment

Chunk size changes what retrieval has to choose from, not how it ranks.

```bash
python evaluate.py --chunk-size 120 --json results/my-c120.json
python evaluate.py --chunk-size 240 --json results/my-c240.json
python evaluate.py --chunk-size 480 --json results/my-c480.json
```

| chunk_size | faithfulness | answer_relevance | context_precision | context_recall | answer_correctness |
| --- | --- | --- | --- | --- | --- |
| 120 | 1.000 | 0.485 | 0.809 | 0.950 | 0.838 |
| 240 (default) | 1.000 | 0.484 | 0.807 | 0.981 | 0.832 |
| 480 | 1.000 | 0.486 | 0.864 | 1.000 | 0.834 |

Small chunks hurt recall. At 120 characters a chunk holds one or two short sentences, so a question needing two facts (the 30-day window *and* the proof-of-purchase rule) has its evidence split across chunks; retrieval returns one of them and recall drops to 0.950. Precision holds at 0.809, because a chunk covering only part of the reference answer earns a low graded relevance, but it also contains less unrelated text, so the two effects roughly cancel here.

Large chunks do well on this corpus, and it is worth being honest about why. At 480 characters recall is perfect — every gold word is inside the returned text — precision is the best of the three (0.864), and `answer_correctness` stays high (0.834). With only 21 chunks from eight short documents, a larger chunk is more likely to contain the whole answer, and the extractor still picks the right sentence out of it. This is a fact about this corpus, not a rule that bigger is always better: with long documents one chunk would cover several topics and precision would fall. The precision/recall trade-off is clearest in the `k` experiment above.

## 5. The faithfulness trap

Look at every table here: `faithfulness` is 1.000 everywhere. Not "high" — pinned.

That is not quality. `faithfulness` measures the share of the answer's content words that appear in the retrieved context, and the extractive generator *copies whole sentences out of that context*. By construction the answer cannot contain a word the context lacks. The metric is a tautology for this generator, so its gate at 0.90 has never been tested: it reports 1.000 every time and can never tell you something got worse. The answer traced in [the previous chapter](04-tour-of-the-code.md) is one or two retrieved sentences glued together, so every one of its content words is, by construction, a word the context already had.

A metric that cannot fail gives you no information. It looks like the strongest line in the report and is the emptiest one. To make it informative you need a generator capable of being unfaithful:

```bash
python -m pip install -r requirements-llm.txt
export OPENAI_API_KEY="sk-..."            # bash / zsh
$env:OPENAI_API_KEY = "sk-..."            # PowerShell
python evaluate.py --generator openai --json results/my-openai.json
```

`requirements-llm.txt` pulls in `openai`, `ragas`, `deepeval` and `python-dotenv` on top of the offline set. `--generator openai` uses `gpt-4o-mini` at temperature 0.0 (override it with the `OPENAI_MODEL` environment variable) with the retrieved chunks as the only allowed context. If you prefer a file over an exported variable, copy `.env.example` to `.env` and put the key there; `evaluate.py` loads it when `python-dotenv` is installed. Only now can `faithfulness` fall below 1.000, and only now does its gate mean anything. Expect other numbers to move as well: an LLM paraphrases, so token-F1-based `answer_correctness` often *drops* even when the answers get better.

## 6. Adding a question

`golden_set.json` is a JSON list. Append an item before the closing `]`:

```json
  {
    "question": "Can I change my delivery address after ordering?",
    "ground_truth": "There is no way to change the delivery address after an order is placed."
  },
```

Re-run `python evaluate.py`; nothing is cached across processes. `questions:` becomes 41, `answerable:` becomes 36, and the means shift — one question out of 35 can move a mean by up to about 0.03.

For a question the corpus **cannot** answer, leave `ground_truth` empty and add `"answerable": false`. Keep that flag on every unanswerable item; it is what protects the run:

```json
  {
    "question": "Do you offer gift wrapping?",
    "ground_truth": "",
    "answerable": false
  },
```

Items marked `answerable: false` are excluded from all five answer-quality metrics (there is no reference to compare against) and measured *only* by `refusal_accuracy`. Giving an unanswerable item a ground truth quietly creates a question gated by nothing except whether the system declined. Unanswerable items matter because they catch the worst RAG failure: a confident, fluent, wrong answer. A system that answers everything scores 1.000 on a dataset with no unanswerable items; add one and `refusal_accuracy` has `n=6`, where answering a single question instead of refusing gives 0.833.

One consequence: appending questions moves the means, so `--check-baseline` will then report regressions that are not regressions. Changing the dataset changes what the baseline means — regenerate it deliberately by running `python evaluate.py`, reading the table, and copying the numbers into `baseline.json`.

## 7. Adding a metric

The extension point is `METRIC_FUNCTIONS` in `scorers/local_metrics.py`: each entry is a callable taking `(record, embedder)` and returning a float in `[0, 1]` or `None`. Add the function above the dictionary, reusing the existing private helpers:

```python
def context_signal(record: RunRecord) -> float | None:
    """Share of the retrieved context's content words that the answer uses."""
    if not record.answerable or not record.contexts:
        return None
    answer_tokens = _content_tokens(record.answer)
    context_tokens = _content_tokens(" ".join(record.contexts))
    if not answer_tokens or not context_tokens:
        return None
    return len(answer_tokens & context_tokens) / len(context_tokens)
```

Then register it in the same dictionary, following the pattern of the six existing entries:

```python
    "context_signal": lambda r, e: context_signal(r),
```

It is called `context_signal`, not `context_noise`, on purpose. Noise would be "1 minus this", and noise is *bad when high* — but `evaluate.py` passes a gate when `score >= gate`, so a bad-when-high metric would pass trivially at 0.95. Express what you want more of, not what you want less of.

Somewhere in `eval_threshold.yaml` add the gate, run `python evaluate.py`, read the new row, then set the gate with a margin: roughly 0.05–0.10 below the observed value, so noise does not fail the build but a real drop does.

```yaml
context_signal: 0.30
```

Then run the tests. `test_every_metric_stays_in_range` iterates `METRIC_FUNCTIONS`, so your metric is checked against six awkward records (empty answer, no contexts, unanswerable, missing ground truth) the moment it exists:

```bash
python -m pytest
```

Two things the harness will not do for you: `--check-baseline` only compares metrics present in `baseline.json`, so a brand-new metric is unguarded until you refresh the baseline; and values outside `[0, 1]` break the table's assumptions, so clamp or normalise as the example does.

## 8. Breaking it on purpose

See the harness fail once on purpose, so you trust it when it fails for real.

**Break the absolute gate.** Drop to a single chunk:

```bash
python evaluate.py --k 1
```

`context_recall` comes out at 0.831 against a gate of 0.90, so the run ends with `[FAIL] 1 metric(s) below threshold: context_recall: 0.831 < 0.90` and exit code 1. Add `-v` to print the three lowest-scoring questions with their answers — that is where debugging starts.

Make the failure loud by raising the gate in `eval_threshold.yaml`:

```yaml
context_recall: 0.95
```

Now `python evaluate.py --k 1` misses by 0.119 instead of 0.069. That is the workflow for tightening quality: set a gate just above today's score and it fires the moment the next change loses recall. Undo both edits afterwards.

**Break it quietly, and let the baseline catch it.** Restore the gate to 0.90, then shrink the chunks:

```bash
python evaluate.py --chunk-size 120 --check-baseline
```

Every absolute gate passes (precision 0.809 ≥ 0.75, recall 0.950 ≥ 0.90), so a naive build goes green. `--check-baseline` still exits 1 and prints `context_recall: 0.950 < baseline 0.981`.

That is the difference between the two mechanisms. Absolute gates encode a floor you refuse to ship below and say nothing about whether you just got worse. The baseline encodes where you actually were, so it catches a 0.031 slide that is still above the floor. It ignores drops of 0.01 or less, and skips any metric where either side measured nothing.

## 9. What to try next

- **Real embeddings.** `VectorRetriever` accepts any embedder with `embed` and `embed_many`, so a `sentence-transformers` model can replace `HashingEmbedder` behind the same interface. Normalise the vectors on the way in or the `matrix @ q_vec` shortcut stops being cosine similarity. This is the change most likely to move `context_precision` on paraphrase questions.
- **A real LLM generator.** `--generator openai`, which turns `faithfulness` into a real metric (section 5).
- **RAGAS or DeepEval.** `--mode ragas` / `--mode deepeval` in a pinned environment with `OPENAI_API_KEY`. Both adapters fail loudly rather than invent scores, and both fall back to offline metrics — check `mode=` in the header before believing a table.
- **Tracing.** `RunRecord` already carries `chunk_ids` and `retrieval_scores` per question, which is the payload a tool like Langfuse wants. Log runs instead of only aggregating them and you can answer "when did this get worse" instead of "is it worse now".
