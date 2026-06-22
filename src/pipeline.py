"""Unified LegalAssist + TRACE-Law pipeline (single source of truth).

This is the one entry point that ties the repo's retrieval *engine*
(embedder + FAISS via retriever.py, or the TF-IDF lexical fallback) to the
TRACE-Law *reasoning* stages (RRF -> rerank -> authority -> temporal ->
generate -> verify -> reliability/abstain) and to the gold-label evaluation.

It is import-safe: heavy deps load lazily. With no models/data it still runs
via the offline fallbacks, which is what the test-suite exercises.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import config
from trace_law.bm25_index import BM25Index
from trace_law.eval_ladder import run_ladder, format_ladder
from trace_law.integration import build_id_corpus
from trace_law.trace_pipeline import TraceConfig, TraceResult, run_trace_pipeline

Retriever = Callable[[str, int], List[Tuple[str, float]]]


def trace_config_from_settings(**overrides) -> TraceConfig:
    """Build a TraceConfig from config.py defaults, with optional overrides."""
    cfg = TraceConfig(
        top_k_retrieve=config.TOP_K,
        rrf_k=config.RRF_K,
        rerank_candidates=config.RERANK_CANDIDATES,
        final_evidence=config.FINAL_EVIDENCE,
        authority_alpha=config.AUTHORITY_ALPHA,
        abstain_threshold=config.ABSTAIN_THRESHOLD,
        use_rrf=config.USE_RRF,
        generation_backend=config.GENERATION_BACKEND,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def build_engine(texts: Sequence[str], doc_ids: Optional[Sequence[str]] = None,
                 dense_index=None) -> Tuple[Dict[str, str], Retriever, Retriever]:
    """Build (id_to_text, dense_retriever, bm25_retriever) from a corpus.

    If `dense_index` (a prebuilt FAISS index aligned to `texts`) is given, the
    dense retriever uses it; otherwise the dense retriever is None (BM25-only),
    so this works offline. `doc_ids` default to positional ids.
    """
    texts = list(texts)
    if doc_ids is None:
        id_to_text, text_to_id = build_id_corpus(texts)
        ids = list(id_to_text.keys())
    else:
        ids = list(doc_ids)
        id_to_text = {i: t for i, t in zip(ids, texts)}
        text_to_id = {t: i for i, t in zip(ids, texts)}

    bm = BM25Index(texts, doc_ids=ids)
    bm25: Retriever = lambda q, k: bm.search(q, k)

    dense: Optional[Retriever] = None
    if dense_index is not None:
        from retriever import retrieve  # lazy

        def dense(q, k):  # type: ignore
            out = []
            for text, score in retrieve(q, dense_index, texts, top_k=k):
                out.append((text_to_id.get(text, text), float(score)))
            return out

    return id_to_text, dense, bm25


def answer(query: str, retrievers: Sequence[Retriever], id_to_text: Dict[str, str],
           doc_meta: Optional[Dict[str, dict]] = None, **cfg_overrides) -> TraceResult:
    """Run the full TRACE-Law pipeline for one query."""
    cfg = trace_config_from_settings(**cfg_overrides)
    return run_trace_pipeline(query, [r for r in retrievers if r is not None],
                              id_to_text, config=cfg, doc_meta=doc_meta)


def evaluate(queries: Dict[str, str], qrels: Dict[str, set], id_to_text: Dict[str, str],
             dense: Optional[Retriever] = None, bm25: Optional[Retriever] = None,
             rerank_score_fn: Optional[Callable] = None,
             top_k: int = 10, ks=(1, 5, 10)) -> Dict[str, Dict[str, float]]:
    """Run the baseline ladder against gold labels and return the metric report."""
    return run_ladder(queries, qrels, id_to_text, dense_retriever=dense,
                      bm25_retriever=bm25, rerank_score_fn=rerank_score_fn, top_k=top_k, ks=ks)


__all__ = ["build_engine", "answer", "evaluate", "format_ladder", "trace_config_from_settings"]
