# LEARN.md — a 7-session study plan

This file is the path through the repository. Each session is 60–90 minutes:
read a little, change something, run it, and explain the result to yourself out
loud. If you cannot explain a number, you have not learned it yet.

Prerequisites: comfortable with Python functions, lists, dicts, and running
commands in a terminal. No machine learning background required.

Setup once:

```bash
python -m pip install -r requirements.txt
python evaluate.py
python -m pytest
```

If both commands succeed, you have a working baseline and can start.

---

## Session 1 — What is RAG, and what does this pipeline actually do?

**Read:** [docs/01-what-is-rag.md](docs/01-what-is-rag.md)

**Do:**

```bash
python -c "from rag.pipeline import build_chunks; c = build_chunks(); print(len(c)); print(c[0].text)"
```

Pick one chunk and one document. Answer in your own words: why is the chunk
smaller than the document, and what would break if it were the whole document?

**Checkpoint:** explain the difference between `Document`, `Chunk`, and
`RunRecord` without looking at the code.

**Exercise:** open `data/corpus/shipping.md` and add one sentence of your own
(e.g. a delivery cut-off). Run `python evaluate.py`. Did any number move? Why,
or why not?

---

## Session 2 — Why evaluation instead of eyeballing

**Read:** [docs/02-how-evaluation-works.md](docs/02-how-evaluation-works.md)

**Do:**

```bash
python evaluate.py --check-baseline
python evaluate.py --k 1
python evaluate.py --k 5
```

Compare the three tables. Notice that `context_recall` and `context_precision`
move in opposite directions.

**Checkpoint:** explain why `--check-baseline` catches something that an
absolute threshold does not.

**Exercise:** break the build on purpose. Set `context_recall: 0.99` in
`eval_threshold.yaml`, run `python evaluate.py`, and confirm the exit code is 1.
Then restore it with `git checkout eval_threshold.yaml` (or undo your edit).
In CI, that non-zero exit is the entire product.

---

## Session 3 — The metrics, one at a time

**Read:** [docs/03-metrics-explained.md](docs/03-metrics-explained.md)

**Do:** open `results/latest.json` and find the answer with the lowest
`answer_correctness`. Read the question, the golden answer, and the produced
answer. Decide which stage failed: retrieval or generation.

**Checkpoint:** for the same question, explain why `faithfulness` might be 1.0
while `answer_correctness` is low.

**Exercise:** in `scorers/local_metrics.py`, change `faithfulness` to return
`None` for every record. Run the evaluation. Observe the `n/a` in the table and
the skipped gate — this is the "no data is not zero" rule doing its job. Undo.

---

## Session 4 — A tour of the code

**Read:** [docs/04-tour-of-the-code.md](docs/04-tour-of-the-code.md)

**Do:** trace one question end to end in a debugger or with `print`:

```bash
python -c "
from rag.pipeline import build_chunks
from rag.retriever import VectorRetriever
ch = build_chunks()
r = VectorRetriever(ch)
for hit in r.retrieve('Can I cancel after my order ships?', k=3):
    print(round(hit.score, 4), hit.chunk_id, hit.text[:70])
"
```

**Checkpoint:** name the function that produced each of these: the list of
chunks, the ranked hits, the refusal decision, the aggregate score.

**Exercise:** add a unit test to `tests/test_metrics.py` for a case the suite
does not cover yet (for example: a correct answer that is a refusal-free
paraphrase). Run `python -m pytest`. Keep the test only if it tests something
real; delete it if it merely restates the implementation.

---

## Session 5 — Experiments you can defend

**Read:** [docs/05-experiments.md](docs/05-experiments.md)

**Do:** run the full sweep yourself.

```bash
python evaluate.py --compare
python evaluate.py --k 1
python evaluate.py --k 5
python evaluate.py --chunk-size 120
python evaluate.py --chunk-size 480
```

**Checkpoint:** pick the configuration you would ship, and write two sentences
justifying it using precision and recall. There is no single right answer; the
justification is the exercise.

**Exercise:** add three new questions to `golden_set.json` — one easy, one
paraphrased, and one that the corpus genuinely cannot answer
(`"answerable": false`). Run the evaluation. Did `refusal_accuracy` change?
What does that tell you about adding unanswerable items to a golden set?

---

## Session 6 — Real embeddings and a real generator

This session needs an API key or a local model. It is optional, and it is where
the offline proxies stop being enough.

**Read:** `rag/embedder.py` and `rag/generator.py` side by side.

**Do (choose one):**

- Local embeddings: `pip install sentence-transformers`, then write an
  `Embedder` class with the same `embed()` / `embed_many()` interface and pass
  it to `VectorRetriever(chunks, embedder=...)`.
- Real generation: `pip install -r requirements-llm.txt`, set
  `OPENAI_API_KEY`, and run `python evaluate.py --generator openai`.

**Checkpoint:** which metrics changed the most, and which stayed put? Explain
why `faithfulness` stops being pinned at 1.000 once a real LLM writes the
answer.

**Warning:** LLM modes cost money per run and are non-deterministic. Record the
model name alongside the scores, or the numbers are not comparable later.

---

## Session 7 — Make it yours

Choose one and finish it. A finished small thing beats an unfinished ambitious
one.

**Option A — Improve retrieval.** Add a reranker: retrieve `k=10`, then re-score
the candidates with a stronger signal (e.g. embedding similarity on the full
question plus the chunk title) and keep the top 3. Measure before and after.

**Option B — Improve the golden set.** Grow it to 100 questions with paraphrase
clusters, then check whether your thresholds still hold. A bigger golden set
usually *lowers* scores — that is honest, not a regression.

**Option C — Add a metric.** Implement a "citation" metric: does the answer's
source chunk actually support the specific claim? Encode it, gate it, document
when it lies.

**Option D — CI for a real repo.** Point the harness at an actual RAG
application (your own project), replace the pipeline behind the `query()`
interface, and keep everything else — golden set, metrics, gates.

Whatever you pick, update `README.md` results and `baseline.json`, and write one
paragraph about what surprised you. That paragraph is the most valuable thing
you will produce.

---

## Self-check: do you actually understand this?

If you can answer these without notes, you have the mental model:

1. Why does `faithfulness` sit at 1.000 in the default configuration, and what
   would make it informative?
2. `context_recall` dropped from 0.981 to 0.831 when `k` went from 3 to 1.
   Which stage broke, and what would you change?
3. Why does the report print `n/a` instead of `0.000` for a metric with
   `count=0`?
4. What is the difference between a threshold gate and a baseline check, and
   why do you want both?
5. Why is `keyword` vs `vector` a *small* difference in this repo, and why
   should you not conclude from it that embeddings do not matter?
6. Why are the five unanswerable questions in the golden set, and what would
   break if you removed them?

If any answer is fuzzy, the relevant document is one of
[01](docs/01-what-is-rag.md)–[05](docs/05-experiments.md). Re-read that section
and re-run the command that demonstrates it.
