# Tour of the code

This chapter walks the repository in reading order, from the knowledge base up to the CI gate. Every file is small enough to
read in full; the point here is that you know what each one is responsible for before you open it, and where to look when a
score moves.

## 0. The map

```
evaluate.py                      CLI, scoring, table, gates, exit code
├── rag/pipeline.py              question -> retrieve -> generate -> RunRecord
│   ├── rag/corpus.py            load data/corpus/*.md
│   ├── rag/chunker.py           sentences -> overlapping chunks
│   ├── rag/retriever.py         chunks -> top-k chunks
│   │   └── rag/embedder.py      text -> unit vector, cosine similarity
│   └── rag/generator.py         (question, top-k) -> answer or refusal
├── scorers/local_metrics.py     RunRecord -> numbers (offline, default)
├── scorers/__init__.py          ScorerUnavailable, canonical metric names
├── scorers/ragas_scorer.py      optional judge (lazy import)
└── scorers/deepeval_scorer.py   optional judge (lazy import)
```

Arrows mean "imports". Nothing imports `evaluate.py`, which is why every stage can be exercised from a test or a REPL without
the CLI.

## 1. `data/corpus/*.md` — the knowledge base

Eight files (`refunds`, `returns`, `exchanges`, `shipping`, `cancellations`, `support-hours`, `warranty`, `payments`), each a
title line plus one paragraph of policy prose.

It is deliberately small: you can hold all eight documents in your head, so when a score moves you can read the exact sentence
that caused it. On a 10,000-document corpus you could no longer tell a retrieval bug from a corpus you have never read. It
deliberately overlaps too: refunds and returns both discuss refunds, damaged items appear in both, exchanges mentions refunds
and store credit. That overlap is what makes the retrieval metrics distinguish anything — with a unique vocabulary per document,
finding the right chunk would be trivial and `context_precision` would sit near 1.000 forever. Two golden-set questions also
need evidence from two documents at once, and those are the questions that make `k` and chunk size matter.

**Question this file answers:** what is the system allowed to know?

## 2. `rag/corpus.py` — loading documents

`Document` is a frozen dataclass of `doc_id` (filename without `.md`), `title` and `text`. `_title_of` takes the first
`# Heading` line via the regex `^#\s+(.+)$` in multiline mode, falling back to the filename, and `_body_of` strips every
heading out of the searchable text — headings are document structure, not prose. Keeping them apart matters: a heading left
in the body gets chunked, embedded, and can be quoted back as part of an answer. `load_corpus` globs `*.md`, **sorts by
filename**, strips each file, skips empty ones, and raises `FileNotFoundError` with a "run from the project root" hint, or
`ValueError` if nothing was found. The sort is not cosmetic: ties are broken by `chunk_id` (section 5) and chunk ids are
built from document order, so sorted loading is what makes runs reproducible day to day.

## 3. `rag/chunker.py` — sentence-aware packing

`split_sentences` splits on `(?<=[.!?])\s+`. Good enough here, not a real splitter — abbreviations would break it.
`chunk_text(doc_id, text, *, title="", max_chars=240, overlap_chars=60)` packs whole sentences: append while the chunk fits,
`flush()` it, carry trailing sentences into the next chunk while they fit the overlap budget, and emit the final buffer with
no overlap after it.

Every `Chunk` also carries the document `title` and exposes an `embed_text` property — the title followed by the prose.
Retrieval matches on `embed_text`, which is a miniature version of "contextual chunk headers": the document title is a strong
topical signal, and three otherwise similar sentences from refunds, returns and exchanges become distinguishable. The
generator only ever sees `Chunk.text`, so the title can never leak into an answer.

Consequences worth knowing:

- `overlap_chars` is a *target*, not a guarantee — overlap is sentence-aligned, so a long last sentence carries nothing.
- `if len(carry) == len(current): carry = carry[-1:]` stops a two-sentence chunk from being copied whole into its successor, which would halt progress.
- One sentence longer than `max_chars` becomes an oversized chunk of its own (the tests allow exactly that case), and bad parameters (`max_chars <= 0`, or overlap outside `[0, max_chars)`) raise `ValueError` instead of producing nonsense quietly.

**The `docid#N` convention.** Ids are `f"{doc_id}#{index}"`: `refunds#0`, `refunds#1`, `refunds#2`. They are stable across runs,
and they travel all the way into the report — `RunRecord.chunk_ids`, the `contexts` list in `results/latest.json`, the per-item
table. When an answer looks wrong, the id is the shortest path back to the text: open the file, find chunk N, read the sentence
the generator copied.

**The size/overlap trade-off.** Chunks too large bury the answer in noise and the generator picks a worse sentence; chunks too
small split one piece of evidence across two chunks so retrieval returns half of it. Overlap buys tolerance at boundaries and
costs duplicate text (the same sentence can be retrieved twice, inflating how much context looks relevant). At the defaults this
corpus becomes 21 chunks — few enough to print if you ever wonder what retrieval had to choose from.

## 4. `rag/embedder.py` — hashing vectors

`tokenize` lowercases, extracts `[a-z0-9]+`, drops `STOPWORDS`, then applies `_stem`. The same tokenizer serves the embedder and
`KeywordRetriever`, so comparing them varies only the retrieval method, not the preprocessing.

`_bucket` hashes with `hashlib.blake2b(..., digest_size=8)` and takes the result modulo `dim` (512). This is why `hashlib` and
**not** Python's built-in `hash()`: string hashing is randomised per process, so `hash("refund")` differs between runs. Buckets
would differ, vectors would differ, scores would wobble, and `--check-baseline` would fail for no reason. A stable hash is what
makes these numbers reproducible at all.

`HashingEmbedder.embed` builds a bag of words (1.0 each) plus word bigrams (0.6), then L2-normalises. Bigrams are the one
concession to word order: `refund_window` is a different feature from `refund` and `window` apart. Vectors are cached per text;
`embed_many` stacks them into an `(n, dim)` matrix; `cosine_similarity` returns 0.0 for a zero vector instead of dividing by
zero.

Stopwords go because every chunk shares them — they would make everything look mildly similar to everything else. `_stem` is a
12-line heuristic (`ies -> y`, `sses -> ss`, trailing `s` but not `ss`/`us`/`is`, `ing`/`ed` with a doubled-consonant undo,
`shipp -> ship`) so that `refund`, `refunds` and `refunded` match. It is not linguistically correct and will mangle words this
corpus never uses; production uses a Snowball stemmer or real embeddings. The ceiling is worth stating plainly: this is still
lexical, so "money back" and "refund" land in unrelated buckets — which is the whole point of the next chapter.

## 5. `rag/retriever.py` — top-k

`RetrievedChunk` is a frozen dataclass wrapping a `Chunk` and the `score` that earned it a place, with `text` and `chunk_id`
pass-through properties. `Retriever` is a `Protocol`, not a base class: anything with a `name` and `retrieve(question, k)`
satisfies it.

`KeywordRetriever` is the deliberate naive baseline — Jaccard overlap `|q ∩ c| / |q ∪ c|` over token sets, built once per
chunk from `c.embed_text`. An empty question returns `[]` instead of every chunk at score 0. It fails in one specific way, no
shared words means no match, and that failure mode is why it exists as a comparison point.

`VectorRetriever` embeds every chunk once into `self._matrix` at construction (again from `embed_text`), so a query is a single
`matrix @ q_vec`. Rows are already unit-normalised, so the dot product *is* the cosine similarity — no separate normalising
step at query time. An empty chunk list raises `ValueError` rather than returning nothing forever. Both sort on
`(-score, chunk_id)`; the second element is the deterministic tie-break, so equal scores resolve alphabetically by id, never by
dictionary or filesystem order. One test builds two identical chunks (`a#0`, `b#0`) and checks the ranking twice.
`build_retriever(kind, chunks)` is the CLI factory and raises on unknown kinds.

## 6. `rag/generator.py` — the answer

`Answer` is `text` plus `refused`; `REFUSAL_TEXT` is a constant so tests and metrics refer to it instead of matching prose.

`ExtractiveGenerator` (default `max_sentences=2`) is offline and deterministic: tokenise the question (empty question or empty
retrieval means refuse); for every sentence in every retrieved chunk compute `score = overlap / sqrt(len(tokens))` and remember
the retrieval rank; skip zero-overlap sentences entirely; refuse if no candidate survived; sort by score, then better rank, then
sentence text; take the best, then add further sentences only while they score at least `0.75 * best_score`, deduplicating, up to
`max_sentences`.

The `sqrt` stops a long sentence winning just by containing more words. The `0.75 * best` cutoff is the interesting part: without
it the generator bolts on a nearby but off-topic second sentence and drags `answer_relevance` down — the source comment says the
evaluation caught that immediately, a good example of the harness earning its keep. The loop uses `continue`, not `break`: a
sentence under the cutoff is skipped, not treated as the end of the list.

Refusal is a feature with its own metric: if no retrieved sentence shares a meaningful word with the question, the system says
it does not know instead of guessing. That is the difference between a system with a known failure mode and one that
hallucinates confidently.

`OpenAIGenerator` is the optional real-LLM path: it imports `openai` lazily (offline use never needs it), raises `SystemExit`
with an install hint if absent, uses a system prompt that forbids outside knowledge and dictates the exact refusal string, and
detects refusals afterwards with `_REFUSAL_MARKERS` because a model will not reliably use the exact wording. Its model is
`OPENAI_MODEL` from the environment when set, otherwise `gpt-4o-mini`, at temperature 0.0. Both generators satisfy the same
duck-typed interface, which is what lets the harness stay unchanged when you swap them.

## 7. `rag/pipeline.py` — the evidence object

`RunRecord` is the centre of the project: per question it holds `question`, `answer`, `contexts` (chunk texts), `chunk_ids`
(stable ids), `ground_truth`, `answerable`, `refused`, `retrieval_scores` (rounded to 4 places). `to_dict()` uses `asdict`, so a
record dumps straight to JSON.

The scorers never see the retriever or the generator, only `RunRecord`. That means you can score a saved run without re-running
the pipeline, diff two runs field by field, or hand the same records to RAGAS later. Keep it a plain dataclass of primitives and
that stays true. `build_chunks` is the one place corpus loading and chunking are joined. `RAGPipeline.query` is three lines of
real work: `retrieve(question, k=self.k)`, `generate(question, retrieved)`, assemble the record pulling `ground_truth` and
`answerable` out of metadata. `load_dataset` validates a non-empty JSON list with a `question` on every item; `run_dataset` maps
it through the pipeline.

## 8. `scorers/local_metrics.py` — the six numbers

`MetricScore` is a frozen `(value, count)` pair. `count` is not decoration: `score_records` skips `None`, so a metric with
nothing to measure reports `count == 0`, and both `gate()` and the table read that as "n/a" instead of a failure. Averaging "no
data" in as zero would quietly poison every number in the report, so the rule is: return `None`, never `0`, when there is
nothing to measure.

| metric | what it computes | where it lies |
| --- | --- | --- |
| `faithfulness` | per answer sentence, the share of its content words present in the joined contexts, averaged | copied text always scores 1.0 even when it answers the wrong question; a correct paraphrase scores low |
| `answer_relevance` | cosine between question and answer, clamped at 0 | short answers score low: too few tokens to match |
| `context_precision` | average precision over retrieved chunks, each graded by `\|gt ∩ chunk\| / \|gt\|` | several chunks that *jointly* cover the reference make no chunk look relevant, so precision is understated |
| `context_recall` | `\|gt ∩ union(contexts)\| / \|gt\|`, order-blind | the golden answer paraphrasing the source lowers it even when retrieval was right |
| `answer_correctness` | half set-based token F1 against the golden answer, half embedding cosine | right answer worded differently loses F1; wrong answer reusing reference words keeps it; set F1 is repetition-blind |
| `refusal_accuracy` | only for `"answerable": false`: 1.0 if it refused, 0.0 if it answered | nothing bad, but it needs unanswerable items to exist at all |

In code, `faithfulness`, `answer_relevance` and `answer_correctness` return `None` for unanswerable or refused items,
`context_precision` and `context_recall` return `None` for unanswerable items or a missing `ground_truth`, and
`refusal_accuracy` returns `None` for answerable ones. `context_precision` scores `0.0` when not one retrieved chunk shares a
word with the reference, which is a genuine failure rather than missing data.

`METRIC_FUNCTIONS` maps names to `(record, embedder)` callables — the single extension point, and the exercise in the next
chapter. `per_item_scores` runs the same functions per question, so you can find *which* item broke, not just that something
did.

## 9. `scorers/__init__.py`, `ragas_scorer.py`, `deepeval_scorer.py` — optional judges

`ScorerUnavailable` is raised when an optional library cannot run. `CANONICAL_METRICS` is the metric-name contract; adapters
translate library names onto it. `_require_api_key` raises `ScorerUnavailable` with an actionable message when `OPENAI_API_KEY`
is unset.

Both adapters import their library *inside* the function, never at module import, so the offline path has no optional
dependency. Both exclude unanswerable records — judging a correct refusal as "unfaithful" is meaningless — so `refusal_accuracy`
stays local. The RAGAS adapter tries `ragas.metrics.collections` first (the 0.4+ layout), falls back to module-level singletons
(0.1/0.2), then raises `ScorerUnavailable` quoting the real import error. `_NAME_MAP` renames `answer_relevancy` and
`response_relevancy` to `answer_relevance`.

DeepEval maps four metrics — `FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`,
`ContextualRecallMetric` — instantiated once with `threshold=0.0, include_reason=False`. There is deliberately **no**
`AnswerCorrectnessMetric`: DeepEval 4.3 has no such class. An earlier version of this project imported a correctness-shaped
class, actually ran `AnswerRelevancyMetric`, and published it under the key `answer_correctness`. A metric with the wrong name
is worse than a missing one because it looks like it is working. That bug is why `answer_correctness` in `--mode deepeval`
simply keeps its offline value.

One consequence to be honest about: `evaluate.py` catches `ScorerUnavailable`, falls back to local metrics, prints the mode as
`local (fallback)` and still exits 0. That is the right call for a teaching repo (no fake scores, the run continues), but a
broken judge then looks like a passing build unless you read the header line.

## 10. `evaluate.py` — turning scores into a red or green build

Defaults sit in one block: `DEFAULT_K = 3`, `DEFAULT_CHUNK_SIZE = 240`, `DEFAULT_OVERLAP = 60`, retriever `vector`, generator
`extractive`, mode `local`. `parse_args` defines the flags (`--retriever`, `--generator`, `--mode`, `--k`, `--chunk-size`,
`--overlap`, `--dataset`, `--corpus`, `--thresholds`, `--json`, `--check-baseline`, `--compare`, `-v`). `load_thresholds` reads
the YAML into `{metric: float}`. At import time it also tries `from dotenv import load_dotenv` inside a `try/except
ImportError`, so a local `.env` can supply `OPENAI_API_KEY` without the offline path gaining a dependency.

`run_suite` builds a pipeline, runs the dataset, scores it locally, then optionally replaces individual values via
`scores.update(...)` from a judge. It returns `config`, `counts` (questions, answerable, unanswerable, refused), `scores` (each
`{value, count}`), `per_item` and `run_at`.

`render_table` prints the table and returns the breaches. `gate()` computes the same breaches independently — it is what the
CLI-contract tests exercise, while `main` uses `render_table`'s result. `regression_failures` compares against `baseline.json`
with a `0.01` tolerance and skips metrics where either side has `count == 0`. `print_worst` shows the three lowest-scoring
questions for the first breached metric, which is where debugging starts.

Exit codes: `0` everything gated passed (and no regression under `--check-baseline`); `1` a threshold breach or a regression;
`2` the config could not be loaded (missing dataset or a malformed threshold file — argparse's own "invalid choice" also exits
2). That mapping is the product: a number becomes a build status.

Every run writes `results/latest.json` unless you pass `--json`. A single run writes the run dict; `--compare` writes
`{"primary": ..., "runs": {...}}`. Note that `--compare` mutates `args.retriever` while looping and gates only the last run
(vector), so keyword numbers in a compare are informational, not gated.

## 11. `tests/` — 44 tests

`pytest.ini` sets `testpaths = tests` and `-q`. `conftest.py` puts the repo root on `sys.path` and provides `run_dir`: a scratch
folder under `tests/_scratch/<random>`. It deliberately avoids `tmp_path`, whose cleanup can raise `PermissionError` on
locked-down Windows setups and take the whole session down.

- `test_chunker.py` — sentence splitting, non-empty chunks, stable `docid#N` ids, the size budget (with the single-long-sentence exception), determinism, overlap across a boundary, rejection of bad parameters.
- `test_retriever.py` — determinism and unit length across embedder instances, cosine bounds and symmetry, zero similarity for empty text, the needed chunk present in the top-k for a shipping question, `k` respected, deterministic tie-breaking, empty question, non-negative scores, the factory's error path.
- `test_generator.py` — answer from the most relevant sentence, the off-topic second sentence actually filtered out, refusal on unrelated context / empty retrieval / empty question, `max_sentences`, determinism.
- `test_evaluate.py` — the CLI contract: gate breaches, gate ignoring unmeasured metrics, passing on equality, regression plus its tolerance, exit 0 with JSON written, per-item evidence present, exit 1 on an impossible threshold, exit 2 on a missing dataset, `--check-baseline` green against the baseline.
- `test_metrics.py` — the most important file. A perfect answer scores 1.0, an unsupported answer 0.0, refusals are excluded rather than penalised, recall falls when retrieval misses, precision rewards relevant chunks first and bottoms out at 0.0, refusal accuracy only applies to unanswerable items, every metric stays in `[0, 1]` across six awkward records, aggregates count only measured items, an empty run does not crash, per-item rows expose a failure.

Why metric tests matter most: a harness with a buggy metric is worse than no evaluation, because it hands you confidence you
have not earned. A broken pipeline is loud. A metric returning 0.84 when it means 0.48 is silent, and you will make decisions on
it.

## 12. Trace one question

Follow `"What proof of purchase do I need for a refund?"` through the default run.

1. **Corpus** — `load_corpus("data/corpus")` reads `refunds.md` first (alphabetical), giving `Document(doc_id="refunds", title="Refund Policy", text=<prose without the heading>)`.
2. **Chunking** — `chunk_text("refunds", text, title="Refund Policy", max_chars=240, overlap_chars=60)` yields three chunks: `refunds#0` (30-day window, proof of purchase), `refunds#1` (processing time, no shipping refund), `refunds#2` (non-refundable rule). Each carries the title for matching; none contains the heading in its prose.
3. **Embedding** — `tokenize(question)` drops the question words and keeps `need, proof, purchase, refund`; `HashingEmbedder.embed` buckets those four tokens plus three bigrams into a 512-dim unit vector.
4. **Retrieval** — `VectorRetriever.retrieve(question, k=3)` builds rows from `embed_text` (title plus prose), computes `matrix @ q_vec`, and returns `refunds#0` (0.3957), `refunds#1` (0.2883), `exchanges#2` (0.1417).
5. **Generation** — `ExtractiveGenerator.generate` splits each `item.text` into sentences, scores each by `overlap / sqrt(len(tokens))`, and picks "The customer must provide proof of purchase, such as an order confirmation email or receipt." The refund-window sentence also clears the `0.75 * best_score` bar, so it is appended: the answer is those two sentences.
6. **Record** — `RAGPipeline.query` returns a `RunRecord` with the three context texts, those chunk ids, `retrieval_scores=[0.3957, 0.2883, 0.1417]`, `answerable=True`.
7. **Metrics** — `score_records` gives, for this item: faithfulness `1.000`, answer_relevance `0.352`, context_precision `1.000`, context_recall `1.000`, answer_correctness `0.689`.

Read those last numbers together, because they disagree on purpose. Retrieval was perfect here: both refund chunks that matter
came back, the irrelevant one ranked last, so precision and recall are both 1.000. The answer is still only "quite good" — it
answers in the first sentence, then appends the refund window from the same chunk, extra material the reference answer does not
contain, so token F1 and embedding similarity both drop and `answer_correctness` lands at 0.689. `answer_relevance` is lower
still at 0.352, because a two-sentence answer is a diluted match for a six-word question.

That is the method of this repository: one question, one number, and the exact chunk where you can go and read why. The next
chapter, [Experiments you can run](05-experiments.md), is the lab — run the suite, change one knob at a time, watch these
numbers move.
