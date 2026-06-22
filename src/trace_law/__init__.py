"""TRACE-Law: reliability-oriented additions to LegalAssist.

Modules:
  metrics            gold-label IR metrics (P/R/MRR/nDCG/MAP)
  fusion             RRF and max-merge for multi-query retrieval
  bm25_index         BM25 baseline retriever (offline)
  rerank             cross-encoder reranking (+ offline fallback)
  authority          authority-weighted evidence ranking
  temporal_validity  IPC/CrPC/IEA -> BNS/BNSS/BSA flagging
  verification       NLI citation verification (+ offline fallback)
  reliability        reliability calibration + abstention
  generation         OpenAI / local-Qwen / extractive backends
  trace_pipeline     end-to-end orchestrator
  eval_ladder        baseline-ladder evaluation against gold labels
"""
__all__ = [
    "metrics", "fusion", "bm25_index", "rerank", "authority",
    "temporal_validity", "verification", "reliability", "generation", "extractive",
    "trace_pipeline", "eval_ladder", "integration",
]
