"""Load the knowledge base that our RAG system is allowed to answer from.

A real system would point this at Confluence, S3, Notion, a database...
For a teaching project a folder of markdown files is perfect: you can read
it, edit it, and see the effect on the evaluation scores immediately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """One source document, before chunking."""

    doc_id: str  # e.g. "refunds" (the filename, without .md)
    title: str  # first markdown heading, falling back to the doc_id
    text: str  # the full body text


def _title_of(text: str, fallback: str) -> str:
    """Use the first '# Heading' as the title if there is one."""
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", flags=re.MULTILINE)


def _body_of(text: str) -> str:
    """Strip markdown headings from the searchable body.

    Headings are document structure, not prose. If they stay in the text they
    end up inside chunks, get embedded, and - worse - can become part of an
    answer ("# Refund Policy Refunds are available..."). The title is kept
    separately in :attr:`Document.title`.
    """
    return _HEADING_RE.sub("", text).strip()


def load_corpus(corpus_dir: str | Path = "data/corpus") -> list[Document]:
    """Read every ``*.md`` file in ``corpus_dir``, sorted by filename.

    Sorting matters: it makes retrieval ties and evaluation scores
    reproducible from run to run.
    """
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Corpus folder not found: {root}. "
            "Run commands from the project root (the folder with evaluate.py)."
        )

    docs: list[Document] = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append(
            Document(
                doc_id=path.stem,
                title=_title_of(text, path.stem),
                text=_body_of(text),
            )
        )

    if not docs:
        raise ValueError(f"No markdown files found in {root}.")

    return docs
