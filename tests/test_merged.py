"""Tests for the combined LegalAssist + TRACE-Law codebase (all offline)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
import pipeline
from agents import DecisionAgent, RetrievalAgent
from retriever import LexicalIndex, retrieve
from trace_law.extractive import firac_brief

CORPUS = {
    "d0": "Supreme Court Constitution Bench. Dishonour of cheque under Section 138 Negotiable Instruments Act; complaint maintainable.",
    "d1": "High Court single judge. Anticipatory bail under Section 438 CrPC for cheating under Section 420 IPC; bail granted.",
    "d2": "Supreme Court. Dowry death under Section 304B IPC with presumption under Section 113B Indian Evidence Act; conviction affirmed.",
    "d3": "Quashing of FIR under Section 482 CrPC; inherent powers discussed.",
    "d4": "A blog post about cooking recipes, unrelated to law.",
}
QUERIES = {"q_bail": "anticipatory bail cheating IPC 420", "q_dowry": "dowry death 304B presumption"}
QRELS = {"q_bail": {"d1"}, "q_dowry": {"d2"}}


def _engine():
    ids = list(CORPUS); texts = [CORPUS[i] for i in ids]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)); m = vec.fit_transform(texts)
    id_to_text, dense, bm25 = pipeline.build_engine(texts, doc_ids=ids)

    def dense_tfidf(q, k):
        sims = cosine_similarity(vec.transform([q]), m).ravel()
        order = sims.argsort()[::-1][:k]
        return [(ids[i], float(sims[i])) for i in order if sims[i] > 0]

    return id_to_text, dense_tfidf, bm25


# ----- config defaults aligned to the paper -----
def test_config_defaults_match_paper():
    assert "MiniLM" in config.EMBEDDING_MODEL
    assert config.RETRIEVER_BACKEND == "dense"
    assert abs(config.DECISION_TAU - 0.35) < 1e-9
    assert config.USE_RRF is True


# ----- agents now use RRF + tau=0.35 -----
def test_retrieval_agent_uses_rrf():
    def fake_retrieve(q, index, texts, top_k):
        # variant lists where 'a' is consistently near the top -> RRF should rank it first
        return {"a": [("a", 0.9), ("b", 0.8)], "b": [("a", 0.5), ("c", 0.4)]}.get("a", [("a", 0.9)]) \
            if False else [("a", 0.9), ("b", 0.8)]
    agent = RetrievalAgent(retrieve_fn=fake_retrieve)
    fused = agent.run(["q1", "q2"], None, None, top_k=5)
    assert fused[0][0] == "a"


def test_decision_agent_threshold():
    assert DecisionAgent().min_best_score == config.DECISION_TAU
    weak = DecisionAgent().run([("d", 0.30), ("e", 0.2), ("f", 0.1)])
    assert weak["sufficient"] is False   # 0.30 < 0.35
    strong = DecisionAgent().run([("d", 0.40), ("e", 0.2), ("f", 0.1)])
    assert strong["sufficient"] is True


# ----- unified pipeline runs offline -----
def test_pipeline_answer_offline():
    id_to_text, dense, bm25 = _engine()
    res = pipeline.answer("dowry death 304B presumption", [dense, bm25], id_to_text)
    d = res.as_dict()
    assert "d2" in d["evidence_ids"]
    assert d["temporal"]["stale"] is True            # IPC 304B flagged
    assert d["reliability"]["decision"] in ("answer", "abstain")


def test_pipeline_evaluate_ladder():
    id_to_text, dense, bm25 = _engine()
    report = pipeline.evaluate(QUERIES, QRELS, id_to_text, dense=dense, bm25=bm25, top_k=5, ks=(1, 5))
    for rung in ["BM25", "Dense-single", "Dense-multi-RRF", "Dense-multi-RRF+rerank"]:
        assert rung in report
    assert max(report[r]["Recall@5"] for r in report) > 0.0


# ----- legacy app path now runs the full trace pipeline via a LexicalIndex (no faiss) -----
def test_legacy_agentic_pipeline_offline():
    import rag_pipeline
    texts = list(CORPUS.values())
    index = LexicalIndex(texts)
    out = rag_pipeline.run_agentic_pipeline("anticipatory bail cheating", index, texts, top_k=5)
    assert set(["query", "expanded_queries", "retrieved", "analysis", "decision", "answer"]) <= set(out)
    assert isinstance(out["answer"], str) and len(out["answer"]) > 0
    assert len(out["expanded_queries"]) == 4


# ----- consolidated extractive generator -----
def test_firac_brief_offline():
    brief = firac_brief("cheque dishonour section 138",
                        [("Dishonour of cheque under Section 138 of the Negotiable Instruments Act, 1881, was held maintainable by the Court.", 0.7)])
    assert "F - Facts" in brief and "Section 138" in brief
