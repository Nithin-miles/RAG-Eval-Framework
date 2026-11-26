"""Optional adapters to real LLM-judge evaluation libraries.

Two things to notice before you read the code:

1. **Optional means optional.** The offline metrics in ``local_metrics.py`` are
   the default because they cost nothing and never flake. These adapters only
   run when you explicitly ask for them (``--mode ragas`` / ``--mode deepeval``)
   and have an ``OPENAI_API_KEY`` set.

2. **Libraries break, and that is normal.** The original version of this project
   pinned ``ragas>=0.1.21``; a fresh install now pulls a version whose import
   path no longer exists, so the "working" code was silently unusable. These
   adapters therefore:
       * import lazily (so the offline path never depends on them),
       * raise :class:`ScorerUnavailable` with an actionable message,
       * map library metric names onto this project's canonical names.

If an adapter fails, the runner prints why and continues - it never silently
substitutes fake scores. That bug is the reason this rewrite exists.
"""

from __future__ import annotations


class ScorerUnavailable(RuntimeError):
    """Raised when an optional scorer cannot run in this environment."""


# Metric names are the project's contract; everything else maps onto these.
CANONICAL_METRICS = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


def _require_api_key() -> None:
    import os

    if not os.getenv("OPENAI_API_KEY"):
        raise ScorerUnavailable(
            "OPENAI_API_KEY is not set. LLM-judge scorers need a key because "
            "they call a model to grade each answer. Set it in .env, or use "
            "the default offline mode (--mode local)."
        )
