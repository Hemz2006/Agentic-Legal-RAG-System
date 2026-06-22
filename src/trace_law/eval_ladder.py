"""Baseline-ladder evaluation against gold relevance labels (TRACE-Law).

Implements the comparative evaluation the literature review asks for. Given a
corpus (id->text), a set of queries, and gold qrels, it runs the ladder:

    BM25
    Dense single-query
    Dense multi-query (max-merge)
    Dense multi-query (RRF)
    Dense multi-query (RRF) + cross-encoder rerank

and reports Recall@k / Precision@k / nDCG@k / MRR / MAP for each rung via
trace.metrics. Dense retrievers are injected as callables, so the same harness
runs offline (BM25 + TF-IDF) or on GPU (FAISS) unchanged.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from trace_law import fusion, metrics, rerank
from trace_law.trace_pipeline import default_expansions

Retriever = Callable[[str, int], List[Tuple[str, float]]]


def _ids(results: Sequence[Tuple[str, float]]) -> List[str]:
    return [d for d, _ in results]


def run_ladder(
    queries: Dict[str, str],                 # query_id -> query_text
    qrels: Dict[str, set],                   # query_id -> relevant doc ids
    id_to_text: Dict[str, str],
    dense_retriever: Optional[Retriever] = None,
    bm25_retriever: Optional[Retriever] = None,
    top_k: int = 10,
    rerank_score_fn: Optional[Callable] = None,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, Dict[str, float]]:
    runs: Dict[str, Dict[str, List[str]]] = {}

    def add(name: str, qid: str, ranked_ids: List[str]):
        runs.setdefault(name, {})[qid] = ranked_ids

    for qid, qtext in queries.items():
        variants = default_expansions(qtext)

        if bm25_retriever is not None:
            add("BM25", qid, _ids(bm25_retriever(qtext, top_k)))

        if dense_retriever is not None:
            single = dense_retriever(qtext, top_k)
            add("Dense-single", qid, _ids(single))

            per_variant = [dense_retriever(v, top_k) for v in variants]
            add("Dense-multi-max", qid, _ids(fusion.max_merge(per_variant)))

            rrf = fusion.reciprocal_rank_fusion(per_variant)
            add("Dense-multi-RRF", qid, _ids(rrf))

            # rerank top candidates of the RRF list
            cand = rrf[: max(2 * top_k, 20)]
            cand_text = [(id_to_text.get(d, ""), s) for d, s in cand]
            reranked = rerank.rerank(qtext, cand_text, score_fn=rerank_score_fn)
            t2id = {id_to_text.get(d, ""): d for d, _ in cand}
            add("Dense-multi-RRF+rerank", qid, [t2id.get(t, t) for t, _ in reranked])

    return {name: metrics.evaluate(run, qrels, ks=ks) for name, run in runs.items()}


def format_ladder(report: Dict[str, Dict[str, float]], ks: Sequence[int] = (1, 5, 10)) -> str:
    cols = [f"nDCG@{k}" for k in ks] + ["MRR@10", "MAP@10", f"Recall@{ks[-1]}"]
    width = 26
    header = "System".ljust(width) + "".join(c.rjust(11) for c in cols)
    lines = [header, "-" * len(header)]
    for name, m in report.items():
        row = name.ljust(width) + "".join(f"{m.get(c, 0.0):11.3f}" for c in cols)
        lines.append(row)
    return "\n".join(lines)
