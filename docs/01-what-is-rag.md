# What is RAG?

RAG stands for **retrieval-augmented generation**. A language model on its own answers from what it learned during training. RAG adds one step: before the model answers, the system searches a document collection and puts the most relevant passages in front of the model. The model is then asked to answer from those passages instead of from memory.

This repository builds a tiny version of that idea so you can see every stage and measure it. The pipeline is four stages: **chunk, embed, retrieve, generate**.

## The problem: confident answers from nowhere

Ask a plain language model "How long does standard shipping take?" for a shop it has never seen. It may answer with a plausible number. The number might even be right for some other shop. That is **hallucination**: fluent, confident text that is not supported by any source you trust.

Hallucination has two common causes:

- The model does not have your private or current knowledge (shipping rules, refund windows, prices).
- The model is optimized to produce plausible continuations, not to say "I don't know".

RAG addresses both by changing what the model is allowed to see. Instead of answering from memory, it answers from a small set of retrieved passages. This is called **grounding**: the answer should be supported by the retrieved context, and the system should decline when that context is not enough.

## The four stages

```text
                 your documents (data/corpus/*.md)
                              |
                              v
                 [ 1. CHUNK ]  cut into small overlapping pieces
                              |
                              v
                 [ 2. EMBED ]  turn each piece into a vector
                              |
   question ------------------+
        |                     |
        v                     v
   [ 2. EMBED ]        stored chunk vectors
        |                     |
        +---------> [ 3. RETRIEVE ] rank chunks, keep the top k
                              |
                              v
                 [ 4. GENERATE ] answer from those chunks only
                              |
                              v
                    answer, or an honest refusal
```

Stages 1 and 2 run once over the corpus. At query time you embed the question, retrieve, then generate. The keyword retriever in this project skips the vector part of stage 2 and compares word sets directly.

## Stage 1: Chunk

The corpus is eight markdown files in `data/corpus`: refunds, returns, exchanges, shipping, cancellations, support hours, warranty, and payments. `rag/corpus.py` reads them, sorted by filename so runs are reproducible.

A retriever compares a question against **chunks**, not whole documents. `rag/chunker.py` splits each document into sentence-aware chunks. It packs whole sentences until the next one would exceed `max_chars` (240 by default), then starts a new chunk and repeats a few trailing sentences until `overlap_chars` (60 by default) is used up. At those defaults the eight documents become 21 chunks.

Both numbers are character budgets, not token counts, and both can be changed from the command line with `--chunk-size` and `--overlap`. Chunk size is a real trade-off:

| Choice | What goes wrong |
| --- | --- |
| Chunk too large | The answer is buried in unrelated text, and the top-k slots are spent on noise. |
| Chunk too small | The answer is split across two chunks, so neither one looks complete. |
| Overlap too small | An answer sitting on a boundary gets cut in half. |
| Overlap too large | Near-duplicate chunks crowd out other useful evidence in the top-k. |

The overlap here is a target, not a guarantee. The chunker repeats whole trailing sentences while they fit in the budget, so the actual overlap can be smaller than 60 characters. This is simpler than a production splitter, and it is enough to show why the setting matters.

## Stage 2: Embed

An **embedding** is a list of numbers (a vector) that represents a piece of text. Texts that mean similar things should get vectors that point in similar directions. A real embedding model is trained on huge text collections to learn that "refund" and "money back" are related.

This project deliberately does not call such a model. It uses `HashingEmbedder` in `rag/embedder.py`: a 512-dimensional vector built from the words and word pairs in the text. Each token is mapped to a dimension with a deterministic hash, and word bigrams get weight 0.6. The vector is then scaled to unit length. This is offline, needs only numpy, and gives the same result on every machine and every run.

Similarity is measured with **cosine similarity**. Cosine compares the angle between two vectors, not their length, so a short chunk and a long chunk can still be compared fairly:

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; 0 means "nothing in common"."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
```

**Important limitation:** the hashing embedder is still lexical. It matches words, lightly stemmed forms, and bigrams. It cannot match synonyms. Ask "can I get my money back" and it will not connect those words to a chunk that says "refunds are available" unless some words overlap. A real embedding model would. The project is honest about this ordering:

```text
keyword overlap  <  hashing embedder  <  real embedding model
```

That is why the vector retriever only modestly outperforms the keyword retriever here. Swapping in a real embedding model is left as an exercise; the evaluation harness does not need to change.

## Stage 3: Retrieve

**Top-k** means "keep the k highest-scoring chunks and throw the rest away". This project defaults to `k=3`. Retrieval returns a ranked list, and the generator only ever sees those three chunks.

There are two retrievers in `rag/retriever.py`:

- **KeywordRetriever** scores a chunk by the Jaccard overlap of question words and chunk words. It is the naive baseline.
- **VectorRetriever** embeds the question and compares it to every precomputed chunk vector by cosine similarity.

Both break ties by `chunk_id`, so equal scores produce the same order every run. The choice of `k` is another trade-off:

| Choice | What goes wrong |
| --- | --- |
| k too small | The one chunk with the answer never reaches the generator. |
| k too large | Irrelevant chunks dilute the context and can distract the generator. |

## Stage 4: Generate

The default `ExtractiveGenerator` is deterministic and offline. It splits each retrieved chunk into sentences, counts shared content words with the question, normalizes by sentence length, and keeps up to two sentences. It only adds a second sentence if that sentence scores at least 0.75 times as well as the best one, so most answers are a single sentence.

Most importantly, it can refuse. If no retrieved sentence shares a single meaningful word with the question, it returns a fixed refusal instead of guessing:

```python
if not candidates:
    # Nothing in the retrieved context relates to the question: refuse.
    return Answer(REFUSAL_TEXT, refused=True)
```

The refusal text is: `I don't have enough information in the provided context to answer that.` Refusing is a feature, not a failure. A system that always answers is a system that hallucinates. This is also why the golden set contains questions the corpus cannot answer: they test whether the pipeline declines.

The optional `OpenAIGenerator` uses the same interface. It sends a system prompt telling the model to answer **only** from the provided context and to reply with that exact refusal otherwise. It is only used with `--generator openai` and an `OPENAI_API_KEY`.

## Trace one question end-to-end

Take the golden-set question: `How long does standard shipping take?`

1. **Chunk.** `shipping.md` is cut into two chunks at the default settings. The first one, `shipping#0`, contains the standard, express and same-day shipping sentences.
2. **Embed.** The question becomes a 512-dimensional vector, and so does every chunk (at index time).
3. **Retrieve.** With `k=3`, the vector retriever returns:

   | Rank | Chunk | Score |
   | --- | --- | --- |
   | 1 | `warranty#2` | 0.4197 |
   | 2 | `shipping#0` | 0.3888 |
   | 3 | `warranty#1` | 0.3511 |

4. **Generate.** The extractor looks at sentences inside those three chunks. A short sentence in `warranty#2` ("Replacement units are shipped with standard shipping.") scores higher than the shipping sentence because its score is normalized by length. The shipping sentence is still close enough to be added as the second sentence, and the two-sentence limit stops there. The answer is:

   ```text
   Replacement units are shipped with standard shipping. Standard shipping takes 5 to 7 business days and is free on orders over 50 dollars.
   ```

The answer is grounded, but it mixes in a sentence from a different document. That is exactly the kind of imperfection the evaluation is supposed to expose: for this item the pipeline scores 1.0 faithfulness and 1.0 context recall, but only 0.698 answer correctness and 0.554 answer relevance. An earlier version of the loader kept markdown headings in the body, so a heading such as `# Shipping` could become part of an answer; `_body_of` now strips headings before chunking and keeps the heading in `Document.title`, used only for retrieval.

## What this project is not

- The corpus is eight short documents. Real collections are far larger and messier.
- The hashing embedder is a teaching stand-in, not a semantic model.
- The extractive generator is a stable stand-in for an LLM, not a language model.
- The golden set has 40 questions, which is enough to learn the workflow but not enough to certify a product.
- The scores are proxies produced by small formulas, not proof that answers are correct.

Those constraints are intentional: they make every stage readable and every run reproducible. The next step is learning how the harness turns these runs into numbers and pass/fail decisions: see [How evaluation works](02-how-evaluation-works.md).
