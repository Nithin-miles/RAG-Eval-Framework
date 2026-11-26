"""The toy RAG (Retrieval-Augmented Generation) pipeline.

The four stages, in order:

    1. corpus.py    - load the documents we are allowed to answer from
    2. chunker.py   - cut documents into small "chunks" a retriever can search
    3. embedder.py  - turn text into vectors so we can measure similarity
    4. retriever.py - find the top-k most similar chunks for a question
    5. generator.py - write an answer using only those chunks
    6. pipeline.py  - glue: question -> chunks -> answer (+ the chunks used)

Everything here runs offline. No API key, no model download, no network.
That is deliberate: evaluation is the hard part, so the RAG under test
should be boring, deterministic and fast.
"""

from rag.chunker import Chunk, chunk_text, split_sentences
from rag.corpus import Document, load_corpus
from rag.embedder import HashingEmbedder, cosine_similarity
from rag.generator import ExtractiveGenerator, OpenAIGenerator
from rag.pipeline import RAGPipeline, RunRecord
from rag.retriever import KeywordRetriever, RetrievedChunk, VectorRetriever

__all__ = [
    "Chunk",
    "chunk_text",
    "split_sentences",
    "Document",
    "load_corpus",
    "HashingEmbedder",
    "cosine_similarity",
    "ExtractiveGenerator",
    "OpenAIGenerator",
    "RAGPipeline",
    "RunRecord",
    "KeywordRetriever",
    "RetrievedChunk",
    "VectorRetriever",
]
