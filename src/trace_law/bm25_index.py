"""BM25 baseline retriever (TRACE-Law baseline ladder).

Uses rank_bm25 (pure Python, no model download) so it runs fully offline.
Returns (doc_id, score) tuples to match the dense/lexical retrievers.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

try:
    from rank_bm25 import BM25Okapi
    _HAVE_BM25 = True
except Exception:  # pragma: no cover
    _HAVE_BM25 = False

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """BM25 over a fixed corpus. doc_ids default to positional ids 'd{i}'."""

    def __init__(self, texts: Sequence[str], doc_ids: Sequence[str] | None = None):
        if not _HAVE_BM25:
            raise ImportError("rank_bm25 is required for BM25Index (pip install rank-bm25)")
        self.texts = list(texts)
        self.doc_ids = list(doc_ids) if doc_ids is not None else [f"d{i}" for i in range(len(self.texts))]
        if len(self.doc_ids) != len(self.texts):
            raise ValueError("doc_ids and texts length mismatch")
        self._tokenized = [tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_tokens = tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # Keep docs that share at least one query term. We deliberately do NOT
        # filter on `scores[i] > 0`: BM25Okapi assigns idf <= 0 to any term that
        # occurs in >= 50% of the corpus, so on small corpora a genuine match can
        # score exactly 0 and a strict `> 0` filter wrongly drops it (returning an
        # empty list and the "Not enough data" message). Term-overlap preserves the
        # filter's intent (drop totally irrelevant docs) without that pathology.
        q_set = set(q_tokens)
        return [(self.doc_ids[i], float(scores[i])) for i in order
                if q_set & set(self._tokenized[i])]
