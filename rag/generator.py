"""Generators: turn (question, retrieved chunks) into an answer.

``ExtractiveGenerator``
    Offline and deterministic: it copies the sentence(s) from the retrieved
    context that best match the question. It is not clever, and it is not
    supposed to be - it is a *stable stand-in* for an LLM so that the
    evaluation scores do not change every time you run them.

    It also refuses: if no retrieved sentence shares a single meaningful word
    with the question, it answers "I don't have enough information" instead of
    guessing. Refusing is a feature. A RAG system that always answers is a
    system that hallucinates.

``OpenAIGenerator``
    The real thing, used only when you pass ``--generator openai`` and set
    ``OPENAI_API_KEY``. Same interface, so the evaluation code does not care
    which one it is talking to. That swap-without-changing-the-harness property
    is the reason the interface exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rag.chunker import split_sentences
from rag.embedder import tokenize
from rag.retriever import RetrievedChunk

REFUSAL_TEXT = "I don't have enough information in the provided context to answer that."

# Phrases an LLM uses when it is declining. Used to detect refusals when the
# generator is a real model rather than our deterministic extractor.
_REFUSAL_MARKERS = (
    "don't have enough information",
    "do not have enough information",
    "insufficient information",
    "i don't know",
    "i do not know",
    "cannot answer",
    "can't answer",
    "not in the provided context",
    "no information",
)


@dataclass
class Answer:
    """What a generator returns."""

    text: str
    refused: bool = False


class ExtractiveGenerator:
    """Deterministic, offline generator: best-matching sentences from context."""

    name = "extractive"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> Answer:
        q_tokens = set(tokenize(question))
        if not q_tokens or not retrieved:
            return Answer(REFUSAL_TEXT, refused=True)

        candidates: list[tuple[float, int, str]] = []
        for rank, item in enumerate(retrieved):
            for sentence in split_sentences(item.text):
                s_tokens = set(tokenize(sentence))
                overlap = len(q_tokens & s_tokens)
                if overlap == 0:
                    continue
                # Normalise by sentence length so a long sentence does not win
                # just by containing more words.
                score = overlap / (len(s_tokens) ** 0.5)
                # Break ties by retrieval rank.
                candidates.append((score, -rank, sentence))

        if not candidates:
            # Nothing in the retrieved context relates to the question: refuse.
            return Answer(REFUSAL_TEXT, refused=True)

        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        picked: list[str] = []
        best_score = candidates[0][0]
        for score, _, sentence in candidates:
            # Only add a second sentence if it is about as relevant as the best
            # one. Without this cutoff the generator happily bolts on a nearby
            # but off-topic sentence, which drags answer relevance down - a
            # small bug that the evaluation caught immediately.
            if score < 0.75 * best_score:
                continue
            if sentence in picked:
                continue
            picked.append(sentence)
            if len(picked) >= self.max_sentences:
                break

        return Answer(" ".join(picked), refused=False)


SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the answer, reply exactly with: "
    f'"{REFUSAL_TEXT}" '
    "Do not use outside knowledge. Answer in one or two sentences."
)


class OpenAIGenerator:
    """Optional generator backed by any OpenAI-compatible chat endpoint."""

    name = "openai"

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        try:
            from openai import OpenAI  # imported lazily: offline use must not need it
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise SystemExit(
                "The openai package is not installed. Run:\n"
                "    pip install -r requirements-llm.txt"
            ) from exc

        self._client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._temperature = temperature

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> Answer:
        context = "\n\n".join(f"- {item.text}" for item in retrieved) or "(no context)"
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        lowered = text.lower()
        refused = any(marker in lowered for marker in _REFUSAL_MARKERS)
        return Answer(text, refused=refused)
