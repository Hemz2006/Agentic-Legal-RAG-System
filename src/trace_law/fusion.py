"""Rank-list fusion for multi-query retrieval (TRACE-Law).

Provides Reciprocal Rank Fusion (RRF) -- the principled fusion used by
RAG-Fusion -- alongside the original max-cosine merge so the two can be
compared head to head in the baseline ladder.

Each per-variant result list is [(doc_id, score), ...] sorted best-first.
Pure functions -> unit-testable offline.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Result = Tuple[str, float]


def max_merge(result_lists: Sequence[Sequence[Result]]) -> List[Result]:
    """Original LegalAssist behaviour: keep the best score per document."""
    best: Dict[str, float] = {}
    for results in result_lists:
        for doc_id, score in results:
            if doc_id not in best or score > best[doc_id]:
                best[doc_id] = score
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[Result]], k: int = 60
) -> List[Result]:
    """RRF: score(d) = sum over lists of 1 / (k + rank_in_list(d)).

    `k` is the standard RRF smoothing constant (60 in Cormack et al. 2009).
    Returns a single ranked list of (doc_id, rrf_score).
    """
    scores: Dict[str, float] = {}
    for results in result_lists:
        for rank, (doc_id, _score) in enumerate(results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def weighted_rrf(
    result_lists: Sequence[Sequence[Result]],
    weights: Sequence[float],
    k: int = 60,
) -> List[Result]:
    """RRF that weights some retrievers (e.g. dense) more than others (BM25)."""
    if len(weights) != len(result_lists):
        raise ValueError("weights and result_lists must be the same length")
    scores: Dict[str, float] = {}
    for w, results in zip(weights, result_lists):
        for rank, (doc_id, _score) in enumerate(results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
