import json
import os


class SimpleRAG:
    """Keyword-based retriever - no vector DB needed."""

    def __init__(self, corpus_path=None):
        if corpus_path is None:
            corpus_path = os.path.join(os.path.dirname(__file__), "corpus.json")
        with open(corpus_path) as f:
            self.corpus = json.load(f)

    def _retrieve(self, question, k=2):
        q_words = set(question.lower().split())
        scored = []
        for doc in self.corpus:
            d_words = set(doc["text"].lower().split())
            overlap = len(q_words & d_words)
            scored.append((overlap, doc))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [d["text"] for _, d in scored[:k]]

    def query(self, question):
        contexts = self._retrieve(question)
        answer = "Based on retrieved context: " + contexts[0] if contexts else "No context found."
        return answer, contexts
