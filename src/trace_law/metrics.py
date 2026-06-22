"""Information-retrieval metrics for gold-label evaluation (TRACE-Law).

These replace "top-1 cosine similarity" as the headline metric. All functions
operate on *ranked lists of document ids* and a set/dict of relevant ids, so
they work with IL-PCR style gold relevance judgments.

Pure functions, no external dependencies -> fully unit-testable offline.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set, Union

Relevance = Union[Set[str], Dict[str, float]]


def _rel_set(relevant: Relevance) -> Set[str]:
    return set(relevant.keys()) if isinstance(relevant, dict) else set(relevant)


def _gain(doc_id: str, relevant: Relevance) -> float:
    """Graded gain if a dict of {id: grade} is given, else binary 1.0/0.0."""
    if isinstance(relevant, dict):
        return float(relevant.get(doc_id, 0.0))
    return 1.0 if doc_id in relevant else 0.0


def precision_at_k(ranked: Sequence[str], relevant: Relevance, k: int = 5) -> float:
    if k <= 0:
        return 0.0
    rel = _rel_set(relevant)
    topk = list(ranked)[:k]
    if not topk:
        return 0.0
    hits = sum(1 for d in topk if d in rel)
    return hits / k


def recall_at_k(ranked: Sequence[str], relevant: Relevance, k: int = 10) -> float:
    rel = _rel_set(relevant)
    if not rel:
        return 0.0
    topk = set(list(ranked)[:k])
    return len(topk & rel) / len(rel)


def reciprocal_rank(ranked: Sequence[str], relevant: Relevance, k: int = 10) -> float:
    rel = _rel_set(relevant)
    for rank, d in enumerate(list(ranked)[:k], start=1):
        if d in rel:
            return 1.0 / rank
    return 0.0


def average_precision(ranked: Sequence[str], relevant: Relevance, k: int = 10) -> float:
    """Average Precision @ k (the per-query term inside MAP)."""
    rel = _rel_set(relevant)
    if not rel:
        return 0.0
    hits = 0
    score = 0.0
    for rank, d in enumerate(list(ranked)[:k], start=1):
        if d in rel:
            hits += 1
            score += hits / rank
    denom = min(len(rel), k)
    return score / denom if denom else 0.0


def dcg_at_k(ranked: Sequence[str], relevant: Relevance, k: int = 10) -> float:
    dcg = 0.0
    for rank, d in enumerate(list(ranked)[:k], start=1):
        g = _gain(d, relevant)
        if g:
            dcg += g / math.log2(rank + 1)
    return dcg


def ndcg_at_k(ranked: Sequence[str], relevant: Relevance, k: int = 10) -> float:
    actual = dcg_at_k(ranked, relevant, k)
    # ideal ranking: highest gains first
    if isinstance(relevant, dict):
        grades = sorted(relevant.values(), reverse=True)
    else:
        grades = [1.0] * len(_rel_set(relevant))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(grades[:k]) if g)
    return actual / ideal if ideal else 0.0


def evaluate(
    runs: Dict[str, Sequence[str]],
    qrels: Dict[str, Relevance],
    ks: Iterable[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Aggregate metrics over a set of queries.

    runs:  {query_id: ranked list of doc_ids}
    qrels: {query_id: relevant ids (set) or {id: grade} (dict)}
    """
    ks = list(ks)
    qids = [q for q in runs if q in qrels]
    out: Dict[str, float] = {"num_queries": float(len(qids))}
    if not qids:
        return out

    for k in ks:
        out[f"P@{k}"] = sum(precision_at_k(runs[q], qrels[q], k) for q in qids) / len(qids)
        out[f"Recall@{k}"] = sum(recall_at_k(runs[q], qrels[q], k) for q in qids) / len(qids)
        out[f"nDCG@{k}"] = sum(ndcg_at_k(runs[q], qrels[q], k) for q in qids) / len(qids)
    out["MRR@10"] = sum(reciprocal_rank(runs[q], qrels[q], 10) for q in qids) / len(qids)
    out["MAP@10"] = sum(average_precision(runs[q], qrels[q], 10) for q in qids) / len(qids)
    return out
