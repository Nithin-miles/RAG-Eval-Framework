"""Glue the stages together: question -> retrieve -> generate -> record.

The output of this module is a :class:`RunRecord` per question. A run record is
the *evidence* the scorers work from, and it is deliberately a plain dataclass
of primitives: you can dump it to JSON, diff two runs, or hand it to a different
scorer library without touching the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag.chunker import chunk_text
from rag.corpus import load_corpus
from rag.generator import Answer, ExtractiveGenerator
from rag.retriever import Retriever, build_retriever


@dataclass
class RunRecord:
    """Everything that happened for one question, ready to be scored."""

    question: str
    answer: str
    contexts: list[str]  # the chunks the answer was grounded in
    chunk_ids: list[str]  # stable ids, so you can find the source chunk fast
    ground_truth: str = ""
    answerable: bool = True
    refused: bool = False
    retrieval_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_chunks(
    corpus_dir: str | Path = "data/corpus",
    *,
    max_chars: int = 240,
    overlap_chars: int = 60,
) -> list:
    """Load the corpus and chunk every document."""
    docs = load_corpus(corpus_dir)
    chunks = []
    for doc in docs:
        chunks.extend(
            chunk_text(
                doc.doc_id,
                doc.text,
                title=doc.title,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )
    return chunks


class RAGPipeline:
    """A minimal, readable RAG pipeline."""

    def __init__(self, retriever: Retriever, generator=None, k: int = 3) -> None:
        self.retriever = retriever
        self.generator = generator or ExtractiveGenerator()
        self.k = k

    def query(self, question: str, **metadata) -> RunRecord:
        retrieved = self.retriever.retrieve(question, k=self.k)
        answer: Answer = self.generator.generate(question, retrieved)
        return RunRecord(
            question=question,
            answer=answer.text,
            contexts=[r.text for r in retrieved],
            chunk_ids=[r.chunk_id for r in retrieved],
            ground_truth=str(metadata.get("ground_truth", "")),
            answerable=bool(metadata.get("answerable", True)),
            refused=answer.refused,
            retrieval_scores=[round(r.score, 4) for r in retrieved],
        )


def load_dataset(path: str | Path = "golden_set.json") -> list[dict]:
    """Read the golden set: a list of questions with reference answers."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must be a non-empty JSON list")
    for i, item in enumerate(data):
        if "question" not in item:
            raise ValueError(f"{path}: item {i} has no 'question' key")
    return data


def run_dataset(pipeline: RAGPipeline, dataset: list[dict]) -> list[RunRecord]:
    """Run every question in the golden set through the pipeline."""
    return [
        pipeline.query(
            item["question"],
            ground_truth=item.get("ground_truth", ""),
            answerable=item.get("answerable", True),
        )
        for item in dataset
    ]
