"""Adapters that connect TRACE-Law to the existing LegalAssist retriever.

The repo's `retrieve(query, index, texts, top_k)` returns [(doc_text, score)].
TRACE-Law's pipeline/ladder expect retrievers shaped as
`(query, k) -> [(doc_id, score)]` plus an `id_to_text` map. These helpers bridge
the two without importing faiss/torch at module load (lazy), so this file is
safe to import anywhere.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple


def build_id_corpus(texts: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Assign stable ids to corpus texts.

    Returns (id_to_text, text_to_id). Ids are positional: 'd0', 'd1', ...
    """
    id_to_text: Dict[str, str] = {}
    text_to_id: Dict[str, str] = {}
    for i, t in enumerate(texts):
        did = f"d{i}"
        id_to_text[did] = t
        text_to_id.setdefault(t, did)
    return id_to_text, text_to_id


def dense_retriever_adapter(index, texts: List[str], text_to_id: Dict[str, str]) -> Callable:
    """Wrap the repo's dense/lexical `retrieve` into a (query, k)->[(id,score)] fn."""
    from retriever import retrieve  # lazy: pulls faiss/torch only when used

    def _retr(query: str, k: int) -> List[Tuple[str, float]]:
        out = []
        for text, score in retrieve(query, index, texts, top_k=k):
            out.append((text_to_id.get(text, text), float(score)))
        return out

    return _retr


def bm25_retriever_adapter(texts: List[str], ids: List[str]) -> Callable:
    """Build a BM25 retriever over the same corpus (offline, no model)."""
    from trace_law.bm25_index import BM25Index

    idx = BM25Index(texts, doc_ids=ids)
    return lambda query, k: idx.search(query, top_k=k)


def build_real_retrievers():  # pragma: no cover - needs faiss/models/data
    """Convenience: build dense + BM25 retrievers from the repo's build_retriever().

    Requires the FAISS index/data to be present and faiss/torch installed.
    """
    from retriever import build_retriever

    index, texts = build_retriever()
    id_to_text, text_to_id = build_id_corpus(texts)
    ids = list(id_to_text.keys())
    dense = dense_retriever_adapter(index, texts, text_to_id)
    bm25 = bm25_retriever_adapter(texts, ids)
    return dense, bm25, id_to_text
