"""End-to-end TRACE-Law tests on a tiny synthetic Indian-legal corpus.

Runs fully offline using BM25 + a TF-IDF cosine retriever as stand-ins for the
GPU dense retriever. Verifies the whole pipeline returns a coherent, structured
result and that the baseline ladder computes gold-label metrics.
"""
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from trace_law.bm25_index import BM25Index
from trace_law.trace_pipeline import run_trace_pipeline, TraceConfig
from trace_law.eval_ladder import run_ladder, format_ladder

CORPUS = {
    "d0": ("Supreme Court of India, Constitution Bench. Dishonour of cheque under "
           "Section 138 of the Negotiable Instruments Act, 1881. The complaint was held maintainable."),
    "d1": ("High Court, single judge. Anticipatory bail under Section 438 CrPC for an offence "
           "punishable under Section 420 IPC for cheating. Bail granted."),
    "d2": ("Supreme Court. Grounds of divorce under the Hindu Marriage Act, 1955; "
           "irretrievable breakdown of marriage considered."),
    "d3": ("Dowry death under Section 304B IPC and the presumption under Section 113B "
           "of the Indian Evidence Act. Conviction affirmed by the Supreme Court."),
    "d4": ("Motor accident compensation under the Motor Vehicles Act for negligence; "
           "just compensation awarded by the Tribunal."),
    "d5": "A note about cooking recipes and kitchen utensils, wholly unrelated to law.",
}
QUERIES = {
    "q_cheque": "section 138 cheque dishonour notice",
    "q_bail": "anticipatory bail cheating IPC 420",
    "q_dowry": "dowry death IPC 304B presumption",
}
QRELS = {
    "q_cheque": {"d0"},
    "q_bail": {"d1"},
    "q_dowry": {"d3"},
}


def _make_retrievers():
    ids = list(CORPUS.keys())
    texts = [CORPUS[i] for i in ids]
    bm25 = BM25Index(texts, doc_ids=ids)

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(texts)

    def dense(query, k):
        qv = vec.transform([query])
        sims = cosine_similarity(qv, matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [(ids[i], float(sims[i])) for i in order if sims[i] > 0]

    def bm25_retr(query, k):
        return bm25.search(query, top_k=k)

    return dense, bm25_retr


def test_pipeline_runs_and_flags_statute():
    dense, bm25_retr = _make_retrievers()
    cfg = TraceConfig(generation_backend="extractive", use_rrf=True, final_evidence=3)
    res = run_trace_pipeline("dowry death IPC 304B presumption", [dense, bm25_retr], CORPUS, cfg)
    d = res.as_dict()
    # d3 (the dowry-death judgment citing IPC 304B) should be retrieved
    assert "d3" in d["evidence_ids"]
    # IPC 304B is pre-2024 -> temporal layer must flag it and map to BNS 80
    assert d["temporal"]["stale"] is True
    assert "Statutory-transition note" in d["answer"]
    # verification + reliability present and well-formed
    assert 0.0 <= d["verification"]["support_rate"] <= 1.0
    assert d["reliability"]["decision"] in ("answer", "abstain")
    assert "F - Facts" in d["answer"] or "FIRAC" in d["answer"] or d["abstained"]


def test_pipeline_abstains_on_irrelevant_query():
    dense, bm25_retr = _make_retrievers()
    cfg = TraceConfig(generation_backend="extractive", abstain_threshold=0.55)
    res = run_trace_pipeline("quantum chromodynamics lattice gauge theory", [dense, bm25_retr], CORPUS, cfg)
    # nothing in the legal corpus matches -> low reliability -> abstain
    assert res.reliability["decision"] == "abstain"
    assert res.abstained is True


def test_baseline_ladder_orders_and_scores():
    dense, bm25_retr = _make_retrievers()
    report = run_ladder(QUERIES, QRELS, CORPUS, dense_retriever=dense,
                        bm25_retriever=bm25_retr, top_k=5, ks=(1, 5))
    # all five rungs present
    for rung in ["BM25", "Dense-single", "Dense-multi-max", "Dense-multi-RRF",
                 "Dense-multi-RRF+rerank"]:
        assert rung in report
        assert 0.0 <= report[rung]["MRR@10"] <= 1.0
    # the relevant doc is findable -> at least one system gets non-zero recall
    assert max(report[r]["Recall@5"] for r in report) > 0.0
    # format helper produces a table
    table = format_ladder(report, ks=(1, 5))
    assert "System" in table and "BM25" in table
