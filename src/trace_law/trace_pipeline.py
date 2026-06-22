"""TRACE-Law pipeline orchestrator.

Wires the components into one explainable flow:

  expand -> retrieve (>=1 retriever) -> RRF fuse -> cross-encoder rerank
         -> authority reweight -> temporal-validity flag -> generate
         -> citation verification (NLI) -> reliability + abstention

It is retriever-agnostic: pass any callables `query -> [(doc_id, score)]`
(BM25, dense FAISS, TF-IDF...). doc_id -> text mapping lets every downstream
stage work on text. This makes the whole pipeline runnable & testable offline
with BM25 + TF-IDF, and identical in shape to the GPU/dense deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from trace_law import authority, fusion, rerank, reliability, temporal_validity, verification
from trace_law.generation import get_generator

Retriever = Callable[[str, int], List[Tuple[str, float]]]  # (query, k) -> [(doc_id, score)]


def default_expansions(query: str) -> List[str]:
    q = query.strip()
    if not q:
        return []
    return [
        q,
        f"Supreme Court judgment on {q}",
        f"Indian case law related to {q}",
        f"judgment reasoning provisions sections for {q}",
    ]


@dataclass
class TraceConfig:
    top_k_retrieve: int = 10        # per-variant retrieval depth
    rrf_k: int = 60
    rerank_candidates: int = 20     # how many fused docs to cross-encode
    final_evidence: int = 3         # docs passed to the generator
    authority_alpha: float = 0.3
    abstain_threshold: float = 0.45
    use_rrf: bool = True            # else max-merge
    generation_backend: str = "extractive"
    rerank_score_fn: Optional[Callable] = None
    entail_fn: Optional[Callable] = None


@dataclass
class TraceResult:
    query: str
    expansions: List[str]
    fused: List[Tuple[str, float]]
    reranked: List[Tuple[str, float]]
    evidence: List[Tuple[str, float]] = field(default_factory=list)
    temporal: dict = field(default_factory=dict)
    answer: str = ""
    verification: dict = field(default_factory=dict)
    reliability: dict = field(default_factory=dict)
    abstained: bool = False

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "expansions": self.expansions,
            "evidence_ids": [d for d, _ in self.evidence],
            "temporal": self.temporal,
            "verification": self.verification,
            "reliability": self.reliability,
            "abstained": self.abstained,
            "answer": self.answer,
        }


def run_trace_pipeline(
    query: str,
    retrievers: Sequence[Retriever],
    id_to_text: Dict[str, str],
    config: Optional[TraceConfig] = None,
    expansions_fn: Callable[[str], List[str]] = default_expansions,
    doc_meta: Optional[Dict[str, dict]] = None,
) -> TraceResult:
    cfg = config or TraceConfig()
    variants = expansions_fn(query) or [query]

    # 1) retrieve each variant with each retriever -> one list per (retriever, variant)
    result_lists: List[List[Tuple[str, float]]] = []
    for retr in retrievers:
        for v in variants:
            result_lists.append(retr(v, cfg.top_k_retrieve))

    # 2) fuse
    fused = (
        fusion.reciprocal_rank_fusion(result_lists, k=cfg.rrf_k)
        if cfg.use_rrf
        else fusion.max_merge(result_lists)
    )

    # 3) cross-encoder rerank top candidates (on text)
    cand = fused[: cfg.rerank_candidates]
    cand_text = [(id_to_text.get(d, ""), s) for d, s in cand]
    reranked_text = rerank.rerank(query, cand_text, score_fn=cfg.rerank_score_fn)
    # map reranked text back to ids (preserve order)
    text_to_id = {id_to_text.get(d, ""): d for d, _ in cand}
    reranked = [(text_to_id.get(t, t), s) for t, s in reranked_text]

    # pre-authority retrieval signal: normalise cross-encoder scores to [0,1].
    # Kept separate from the authority-blended score so reliability does NOT
    # double-count authority.
    ce_scores = [s for _, s in reranked]
    ce_hi = max(ce_scores) if ce_scores else 0.0
    if ce_hi > 0:
        ce_norm = {d: max(0.0, s) / ce_hi for d, s in reranked}
    else:
        ce_lo = min(ce_scores) if ce_scores else 0.0
        rng = (ce_hi - ce_lo) or 1.0
        ce_norm = {d: (s - ce_lo) / rng for d, s in reranked}

    # 4) authority reweighting on the reranked top set (metadata overrides regex)
    top_for_auth = reranked[: max(cfg.final_evidence * 2, cfg.final_evidence)]
    auth_input = [(id_to_text.get(d, ""), s) for d, s in top_for_auth]
    metas = (
        [(doc_meta.get(d, {}) or {}).get("authority_meta") for d, _ in top_for_auth]
        if doc_meta else None
    )
    auth_ranked = authority.rerank_by_authority(auth_input, alpha=cfg.authority_alpha, metas=metas)
    text_to_id_auth = {id_to_text.get(d, ""): d for d, _ in top_for_auth}
    evidence_pairs = []  # (doc_id, final_blended, authority, retrieval_norm)
    for t, final, detail in auth_ranked[: cfg.final_evidence]:
        did = text_to_id_auth.get(t, t)
        evidence_pairs.append((did, final, detail["authority"], ce_norm.get(did, 0.0)))
    evidence_ids = [(did, final) for did, final, _a, _r in evidence_pairs]
    evidence_text = [(id_to_text.get(d, ""), s) for d, s in evidence_ids]

    # 5) temporal-statutory validity (+ optional date awareness from metadata)
    temporal_reports = temporal_validity.check_documents([t for t, _ in evidence_text])
    temporal_stale = any(r.stale for r in temporal_reports)
    date_notes = []
    if doc_meta:
        for did, _ in evidence_ids:
            decided = (doc_meta.get(did, {}) or {}).get("decided")
            note = temporal_validity.annotate_query_decision(query, decided) if decided else None
            if note:
                date_notes.append(f"{did}: {note}")
    temporal = {
        "stale": temporal_stale,
        "reports": [r.as_dict() for r in temporal_reports],
        "date_notes": date_notes,
    }

    # 6) generate
    gen = get_generator(cfg.generation_backend)
    answer = gen.generate(query, evidence_text)
    # prepend an explicit statutory-transition warning into the answer if stale
    temporal_flagged = False
    if temporal_stale:
        warn = "; ".join(f.message for r in temporal_reports for f in r.flags)[:600]
        if date_notes:
            warn += " | " + "; ".join(date_notes)
        answer = f"[Statutory-transition note] {warn}\n\n{answer}"
        temporal_flagged = True  # user is warned, but staleness is not resolved

    # 7) citation verification
    verify = verification.verify_answer(
        answer, [t for t, _ in evidence_text], entail_fn=cfg.entail_fn
    )

    # 8) reliability + abstention (retrieval signal is PRE-authority; no double count)
    rets = [r for _, _, _, r in evidence_pairs]
    best_ret = rets[0] if rets else 0.0
    margin = max(0.0, best_ret - min(rets)) if rets else 0.0
    mean_auth = (
        sum(a for _, _, a, _ in evidence_pairs) / len(evidence_pairs)
        if evidence_pairs else 0.5
    )
    rel = reliability.reliability_score(
        retrieval_score=best_ret,
        support_rate=verify["support_rate"],
        authority=mean_auth,
        margin=margin,
        temporal_stale=temporal_stale,
        temporal_flagged=temporal_flagged,
        abstain_threshold=cfg.abstain_threshold,
    )
    abstained = rel.decision == "abstain"
    if abstained:
        msg = reliability.abstention_message(rel)
        answer = msg or answer

    return TraceResult(
        query=query,
        expansions=variants,
        fused=fused,
        reranked=reranked,
        evidence=evidence_ids,
        temporal=temporal,
        answer=answer,
        verification=verify,
        reliability=rel.as_dict(),
        abstained=abstained,
    )
