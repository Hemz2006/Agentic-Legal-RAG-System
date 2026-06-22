"""Unit tests for the TRACE-Law modules (all offline, no model weights)."""
from datetime import date

from trace_law import (
    metrics, fusion, authority, temporal_validity, verification,
    reliability, rerank, generation,
)
from trace_law.bm25_index import BM25Index


# ----------------------------- metrics -----------------------------
def test_precision_recall_basic():
    ranked = ["a", "b", "c", "d"]
    rel = {"a", "c"}
    assert metrics.precision_at_k(ranked, rel, 4) == 0.5
    assert metrics.recall_at_k(ranked, rel, 4) == 1.0
    assert metrics.recall_at_k(ranked, rel, 1) == 0.5


def test_mrr_and_perfect_ndcg():
    assert metrics.reciprocal_rank(["x", "a"], {"a"}, 10) == 0.5
    perfect = ["a", "b", "c"]
    assert abs(metrics.ndcg_at_k(perfect, {"a", "b", "c"}, 3) - 1.0) < 1e-9


def test_ndcg_graded_and_order_matters():
    qrel = {"a": 3.0, "b": 1.0}
    good = metrics.ndcg_at_k(["a", "b", "z"], qrel, 3)
    bad = metrics.ndcg_at_k(["b", "a", "z"], qrel, 3)
    assert good > bad
    assert 0.0 <= bad <= good <= 1.0


def test_evaluate_aggregates():
    runs = {"q1": ["a", "b"], "q2": ["x", "y"]}
    qrels = {"q1": {"a"}, "q2": {"y"}}
    out = metrics.evaluate(runs, qrels, ks=(1, 2))
    assert out["num_queries"] == 2
    assert 0.0 <= out["MRR@10"] <= 1.0
    assert out["Recall@2"] == 1.0


# ----------------------------- fusion -----------------------------
def test_rrf_rewards_consensus():
    # doc 'a' is top in both lists -> should win RRF over 'b' (top in one only)
    l1 = [("a", 0.9), ("b", 0.8)]
    l2 = [("a", 0.7), ("c", 0.6)]
    fused = fusion.reciprocal_rank_fusion([l1, l2], k=60)
    assert fused[0][0] == "a"


def test_max_merge_keeps_best_score():
    l1 = [("a", 0.4)]
    l2 = [("a", 0.9), ("b", 0.5)]
    merged = dict(fusion.max_merge([l1, l2]))
    assert merged["a"] == 0.9
    assert merged["b"] == 0.5


def test_weighted_rrf_weighting():
    l_dense = [("a", 1.0)]
    l_bm25 = [("b", 1.0)]
    fused = dict(fusion.weighted_rrf([l_dense, l_bm25], weights=[5.0, 1.0]))
    assert fused["a"] > fused["b"]


# ----------------------- temporal validity -----------------------
def test_ipc_section_mapped_to_bns():
    rep = temporal_validity.check_text("The accused was convicted under Section 302 IPC.")
    assert rep.stale is True
    flags = {(f.code, f.section): f.new_section for f in rep.flags}
    assert flags.get(("IPC", "302")) == "103"


def test_crpc_and_evidence_act_detected():
    rep = temporal_validity.check_text(
        "Anticipatory bail under Section 438 CrPC; reliance on Section 65B of the Indian Evidence Act."
    )
    codes = {f.code for f in rep.flags}
    assert {"CRPC", "IEA"} <= codes
    by = {(f.code, f.section): f.new_section for f in rep.flags}
    assert by.get(("CRPC", "438")) == "482"
    assert by.get(("IEA", "65B")) == "63"


def test_no_false_positive_on_modern_text():
    rep = temporal_validity.check_text("The matter was decided under the Bharatiya Nyaya Sanhita, 2023.")
    assert rep.stale is False
    assert rep.flags == []


def test_nearest_code_attribution_no_cross_flags():
    # Two codes in one sentence: each section must bind to its OWN code only.
    text = ("Dowry death under Section 304B IPC with the presumption under "
            "Section 113B of the Indian Evidence Act, 1872.")
    rep = temporal_validity.check_text(text)
    pairs = {(f.code, f.section) for f in rep.flags}
    assert ("IPC", "304B") in pairs
    assert ("IEA", "113B") in pairs
    # the bug we fixed: 113B must NOT be attributed to IPC, nor 304B to IEA
    assert ("IPC", "113B") not in pairs
    assert ("IEA", "304B") not in pairs


# ----------------------------- authority -----------------------------
def test_supreme_constitution_bench_outranks_single_high_court():
    sc = "Supreme Court of India, Constitution Bench, 2022. The appeal is allowed."
    hc = "High Court, single judge, 1981."
    assert authority.score_authority(sc).authority > authority.score_authority(hc).authority


def test_overruled_downweights():
    base = "Supreme Court division bench held the principle applies."
    overruled = base + " This decision was later overruled."
    assert authority.score_authority(overruled).authority < authority.score_authority(base).authority


def test_rerank_by_authority_changes_order():
    results = [
        ("District court single judge 1975 matter.", 0.81),     # high sim, low authority
        ("Supreme Court Constitution Bench 2021 landmark.", 0.80),  # slightly lower sim, top authority
    ]
    ranked = authority.rerank_by_authority(results, alpha=0.6)
    assert ranked[0][0].startswith("Supreme Court")


# ----------------------------- rerank -----------------------------
def test_lexical_rerank_prefers_overlap():
    q = "section 138 cheque dishonour"
    results = [("a story about cooking recipes", 0.9),
               ("dishonour of cheque under section 138 negotiable instruments", 0.4)]
    out = rerank.rerank(q, results)
    assert "cheque" in out[0][0]


# ----------------------------- verification -----------------------------
def test_supported_vs_unsupported_claim():
    sources = ["The court held that bail was granted under Section 438 considering custodial antecedents."]
    answer = ("Bail was granted under Section 438 [Judgment 1]. "
              "The accused was sentenced to life imprisonment for murder [Judgment 1].")
    res = verification.verify_answer(answer, sources, support_threshold=0.5)
    labels = {v["claim"][:20]: v["label"] for v in res["verdicts"]}
    # first claim is supported by the source; second (sentenced/murder) is not
    assert res["num_cited_claims"] == 2
    assert res["num_unsupported"] >= 1
    assert 0.0 <= res["support_rate"] <= 1.0


def test_uncited_substantive_claim_flagged():
    res = verification.verify_answer("The penalty is always seven years imprisonment.", ["irrelevant text"])
    assert any(v["label"] == "uncited" for v in res["verdicts"])


# ----------------------------- reliability -----------------------------
def test_high_signals_answer_low_signals_abstain():
    good = reliability.reliability_score(retrieval_score=0.8, support_rate=0.9, authority=0.8, margin=0.3)
    bad = reliability.reliability_score(retrieval_score=0.1, support_rate=0.1, authority=0.2, margin=0.0)
    assert good.decision == "answer"
    assert bad.decision == "abstain"
    assert reliability.abstention_message(bad) is not None
    assert reliability.abstention_message(good) is None


def test_temporal_penalty_lowers_reliability():
    flagged = reliability.reliability_score(0.6, 0.6, 0.6, 0.1, temporal_stale=True, temporal_flagged=True)
    unflagged = reliability.reliability_score(0.6, 0.6, 0.6, 0.1, temporal_stale=True, temporal_flagged=False)
    clean = reliability.reliability_score(0.6, 0.6, 0.6, 0.1, temporal_stale=False)
    # unflagged stale is worst; flagged stale still costs a residual vs clean
    assert unflagged.reliability < flagged.reliability < clean.reliability


def test_authority_metadata_overrides_regex():
    # text alone looks like a single-judge district matter; metadata says SC Constitution Bench
    text = "District court single judge 1975 matter."
    low = authority.score_authority(text)
    high = authority.score_authority(
        text, meta={"court_level": 1.0, "bench": 1.0, "recency": 0.95}
    )
    assert high.authority > low.authority


def test_negation_aware_entailment():
    ev = "The court held that bail was granted to the accused."
    supported = verification.lexical_entailment(ev, "Bail was granted to the accused [Judgment 1].")
    negated = verification.lexical_entailment(ev, "Bail was not granted to the accused [Judgment 1].")
    assert negated < supported


# ----------------------------- generation -----------------------------
def test_extractive_generator_offline():
    gen = generation.get_generator("extractive")
    out = gen.generate("cheque dishonour section 138",
                        [("Dishonour of cheque under Section 138 of the Negotiable Instruments Act, 1881.", 0.7)])
    assert "FIRAC" in out or "F - Facts" in out
    assert "Section 138" in out


def test_backend_fallbacks_without_keys_or_weights():
    # openai with no key and local with no weights both degrade to extractive output
    ev = [("Supreme Court held appeal allowed.", 0.6)]
    assert generation.get_generator("openai", api_key=None).generate("q", ev)
    assert generation.get_generator("local").generate("q", ev)


# ----------------------------- BM25 -----------------------------
def test_bm25_ranks_relevant_first():
    texts = [
        "Dishonour of cheque under Section 138 Negotiable Instruments Act.",
        "Grounds of divorce under the Hindu Marriage Act.",
        "Anticipatory bail guidelines for the Supreme Court.",
    ]
    idx = BM25Index(texts)
    res = idx.search("cheque dishonour section 138", top_k=3)
    assert res and res[0][0] == "d0"


# --- v3 regression: rerank accepts both batch and per-doc score functions ---
def test_rerank_accepts_perdoc_and_batch_score_fns():
    from trace_law import rerank
    cands = [("alpha bravo charlie", 0.1), ("delta echo", 0.2), ("alpha bravo", 0.3)]
    # per-doc scorer: (query, doc) -> float  (the lexical_overlap_score shape)
    per_doc = rerank.rerank("alpha bravo", cands,
                            score_fn=rerank.lexical_overlap_score)
    # batch scorer: (query, docs) -> list[float]  (the load_cross_encoder shape)
    def batch_fn(q, docs):
        return [rerank.lexical_overlap_score(q, d) for d in docs]
    batch = rerank.rerank("alpha bravo", cands, score_fn=batch_fn)
    assert [d for d, _ in per_doc] == [d for d, _ in batch]
    assert per_doc[0][0] == "alpha bravo"   # exact query match ranks first


# --- v3 regression: authority polarity inversion fix ---
def test_authority_no_penalty_when_lower_court_overruled():
    from trace_law import authority as au
    # Supreme Court overruling a High Court -> should NOT be penalised as "overruled"
    sc = au.score_authority(
        "The Supreme Court allowed the appeal; the High Court judgment was overruled.")
    # A case whose OWN holding was overruled (no lower-court cue) -> penalised
    own = au.score_authority("This decision was subsequently overruled and is no longer good law.")
    assert sc.authority > own.authority
    assert any("lower court" in t for t in sc.treatment)
