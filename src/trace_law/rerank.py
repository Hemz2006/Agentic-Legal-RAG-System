"""Cross-encoder reranking (TRACE-Law).

A second retrieval tier that rescores the top candidates with a query-document
cross-encoder. The real model (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2 or
a legal cross-encoder) needs weights from the model hub. To keep the system
runnable and testable offline, we expose a pluggable `score_fn` and a
deterministic lexical-overlap fallback when no model is available.
"""
from __future__ import annotations

import math
import re
from typing import Callable, List, Optional, Sequence, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


def _toks(s: str) -> set:
    return set(_TOKEN.findall(s.lower()))


def lexical_overlap_score(query: str, doc: str) -> float:
    """Deterministic offline stand-in for a cross-encoder relevance score.

    Jaccard-like overlap with a length-damping term. Bounded in [0, 1].
    """
    q, d = _toks(query), _toks(doc)
    if not q or not d:
        return 0.0
    inter = len(q & d)
    if inter == 0:
        return 0.0
    coverage = inter / len(q)                 # how much of the query is covered
    damp = 1.0 / (1.0 + math.log1p(len(d)))   # penalise very long docs slightly
    return coverage * (0.5 + 0.5 * damp)


def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Try to load a sentence-transformers CrossEncoder. Returns a score_fn or None."""
    try:  # pragma: no cover - needs network/model weights
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)

        def _score(query: str, docs: Sequence[str]) -> List[float]:
            pairs = [(query, d) for d in docs]
            return [float(s) for s in model.predict(pairs)]

        return _score
    except Exception:
        return None


def rerank(
    query: str,
    results: Sequence[Tuple[str, float]],
    top_k: Optional[int] = None,
    score_fn: Optional[Callable[[str, Sequence[str]], List[float]]] = None,
) -> List[Tuple[str, float]]:
    """Rerank (doc, score) candidates by cross-encoder relevance.

    score_fn(query, docs) -> list[float]. If None, uses the offline lexical
    fallback. Returns reranked [(doc, ce_score), ...].
    """
    if not results:
        return []
    docs = [d for d, _ in results]
    if score_fn is None:
        ce_scores = [lexical_overlap_score(query, d) for d in docs]
    else:
        ce_scores = _apply_score_fn(score_fn, query, docs)
    reranked = sorted(zip(docs, ce_scores), key=lambda x: x[1], reverse=True)
    return reranked[:top_k] if top_k else reranked


def _apply_score_fn(score_fn: Callable, query: str, docs: Sequence[str]) -> List[float]:
    """Call a user-supplied rerank scorer, accepting either calling convention.

    Two score-function shapes occur in this codebase and the docs:
      * batch:    score_fn(query, docs) -> list[float]   (load_cross_encoder)
      * per-doc:  score_fn(query, doc)  -> float         (lexical_overlap_score)
    We try the batch form first and transparently fall back to per-doc, so the
    public API and the Action-Guide snippets work with either callable.
    """
    try:
        scores = score_fn(query, docs)
        # A per-doc function handed a list returns a single float (or errors);
        # only accept a result that is genuinely one score per document.
        if isinstance(scores, (list, tuple)) and len(scores) == len(docs):
            return [float(s) for s in scores]
    except (AttributeError, TypeError):
        pass
    return [float(score_fn(query, d)) for d in docs]
