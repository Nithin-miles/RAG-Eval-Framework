"""Turn text into vectors, and measure how similar two vectors are.

In a real system you would call an embedding model (OpenAI
``text-embedding-3-small``, BGE, E5, ...). Those models are *semantic*: they
put "refund" and "money back" close together.

Here we use a **hashing embedder**: a deterministic, offline stand-in that
still behaves like an embedding, because it is a vector and we rank by cosine
similarity. It is still lexical (it cannot understand synonyms), so it sits
between "keyword matching" and "real embeddings" - which is exactly what makes
it a good teaching tool:

    keyword overlap  <  hashing embedder  <  real embedding model

Why hashing instead of TF-IDF?
    * no vocabulary to fit, no external file, no sklearn dependency
    * stable ids: the same word always lands in the same dimension
    * about 20 lines you can read

The one detail that matters: we use :mod:`hashlib`, NOT Python's built-in
``hash()``. ``hash()`` is randomised per process, so scores would change
between runs and the evaluation would not be reproducible.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")

# Words that carry no topical meaning; they would just add noise.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "do", "does", "for", "from", "has", "have", "how", "i", "if", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "so", "that",
    "the", "their", "then", "there", "these", "they", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "offer", "with", "you", "your",
}


def _stem(word: str) -> str:
    """A deliberately tiny suffix stripper - not a real stemmer.

    It exists because "refund" vs "refunds" vs "refunded" should count as the
    same word for matching, and a 12-line heuristic gets you most of the way.
    A production system uses a Snowball stemmer or, better, real embeddings.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]

    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        base = word[:-2]
    else:
        return word

    # "shipp" -> "ship": undo a doubled final consonant.
    if len(base) > 3 and base[-1] == base[-2]:
        base = base[:-1]
    return base


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase, stopword-filtered, lightly stemmed word tokenizer.

    Shared by the embedder and the keyword retriever so that a fair
    comparison between them varies only the *retrieval method*.
    """
    words = _WORD_RE.findall(text.lower())
    if keep_stopwords:
        return words
    return [_stem(w) for w in words if w not in STOPWORDS]


def _bucket(token: str, dim: int) -> int:
    """Map a token to a dimension index, deterministically across processes."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


@dataclass
class HashingEmbedder:
    """Deterministic, offline text embedder (bag of words + word bigrams).

    Bigrams matter: "refund window" is a different signal from "refund" and
    "window" appearing separately, which is what lets this hand-written
    embedder do slightly better than plain keyword overlap.
    """

    dim: int = 512
    bigram_weight: float = 0.6
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def embed(self, text: str) -> np.ndarray:
        """Return an L2-normalised vector for one piece of text."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        vec = np.zeros(self.dim, dtype=np.float64)
        words = tokenize(text)
        for word in words:
            vec[_bucket(word, self.dim)] += 1.0
        for left, right in zip(words, words[1:]):
            vec[_bucket(f"{left}_{right}", self.dim)] += self.bigram_weight

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm

        self._cache[text] = vec
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Stack many embeddings into a matrix of shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self.embed(t) for t in texts])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; 0 means "nothing in common"."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
