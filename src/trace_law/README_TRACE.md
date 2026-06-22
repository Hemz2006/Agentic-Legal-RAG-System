# TRACE-Law additions to LegalAssist

This package (`src/trace_law/`) implements the reliability-oriented upgrades
proposed in the literature review, on top of the existing four-agent pipeline.
Everything here is **additive** — the original `agents.py` / `retriever.py` /
`rag_pipeline.py` still work unchanged.

## What was added (maps 1:1 to the TRACE-Law proposal)

| Proposal | Module | Runs offline? |
|---|---|---|
| Gold-label evaluation (Recall/Precision/MRR/nDCG/MAP) | `metrics.py` | ✅ |
| Baseline ladder (BM25 → dense-single → multi-max → multi-RRF → +rerank) | `eval_ladder.py`, `bm25_index.py`, `fusion.py` | ✅ |
| Reciprocal Rank Fusion (replaces max-merge) | `fusion.py` | ✅ |
| Cross-encoder reranking | `rerank.py` (+ lexical fallback) | ✅ (fallback) |
| Temporal-statutory validity (IPC→BNS, CrPC→BNSS, IEA→BSA) | `temporal_validity.py` | ✅ |
| Authority-weighted ranking (court / bench / treatment / recency) | `authority.py` | ✅ |
| Local LLM (Qwen) / OpenAI / extractive generation | `generation.py` | ✅ (extractive) |
| Structural citation verification (NLI) | `verification.py` (+ lexical fallback) | ✅ (fallback) |
| Reliability calibration + abstention | `reliability.py` | ✅ |
| End-to-end orchestrator | `trace_pipeline.py` | ✅ |
| Adapters to the repo's FAISS retriever | `integration.py` | needs faiss |

> Heavy models (FAISS dense vectors, a real cross-encoder, an NLI model, Qwen)
> need weights from the model hub and a GPU. Each of those modules has a
> deterministic offline fallback so the pipeline and the whole test-suite run
> with **no GPU, no API key, and no downloads**. Swap the fallbacks for the real
> `score_fn` / `entail_fn` / generation backend in production.

## Run the tests and the demo
```bash
pip install -r requirements.txt          # adds rank-bm25
python -m pytest tests/ -q                # 25 tests, all offline
python scripts/demo_trace.py              # ladder + one full pipeline query
```

## Wire it into the real (dense/FAISS) system
```python
from trace_law.integration import build_real_retrievers
from trace_law.trace_pipeline import run_trace_pipeline, TraceConfig
from trace_law.rerank import load_cross_encoder
from trace_law.verification import load_nli

dense, bm25, id_to_text = build_real_retrievers()      # uses repo build_retriever()
cfg = TraceConfig(
    use_rrf=True,
    generation_backend="local",                        # local Qwen; "openai" or "extractive" too
    rerank_score_fn=load_cross_encoder(),              # None -> offline fallback
    entail_fn=load_nli(),                              # None -> offline fallback
)
result = run_trace_pipeline("anticipatory bail cheating IPC 420", [dense, bm25], id_to_text, cfg)
print(result.as_dict())
```

## Evaluate against IL-PCR gold labels
`eval_ladder.run_ladder(queries, qrels, id_to_text, dense_retriever=dense, bm25_retriever=bm25)`
returns Recall@k / Precision@k / nDCG@k / MRR / MAP for every rung. Build `qrels`
(`{query_id: {relevant_doc_id, ...}}`) from IL-TUR's IL-PCR split instead of the
old `query_id == relevant_id` proxy in `data/eval_queries.csv`.

## Known config notes (worth fixing before submission)
- `config.py` defaults `EMBEDDING_MODEL=law-ai/InLegalBERT` and
  `RETRIEVER_BACKEND=lexical`, but the paper reports `all-MiniLM-L6-v2` dense.
  Set `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` and
  `RETRIEVER_BACKEND=dense` to match the paper.
- `DecisionAgent` default `min_best_score=0.28`; the paper's main τ is `0.35`.

## Statute mapping coverage
`temporal_validity.py` ships a curated, frequently-litigated subset of the
IPC→BNS / CrPC→BNSS / IEA→BSA correspondences. Unknown sections are still flagged
at the code level. Extend the dictionaries with the official correspondence
tables for full coverage.

## Revision notes (bug fixes applied)
1. **Reliability no longer double-counts authority.** The pipeline now feeds the
   reliability scorer the *pre-authority* (normalised cross-encoder) retrieval
   signal, separate from the authority term.
2. **Temporal penalty is graduated, not dead.** Flagging the user reduces the
   penalty (0.15 → 0.05) but does not zero it; staleness always costs something.
   Parameter renamed `temporal_addressed` → `temporal_flagged`.
3. **Margin edge case fixed.** Computed over available evidence only (no phantom
   third doc inflating the confidence margin).
4. **Authority accepts trusted metadata** (`court_level`, `bench`, `recency`,
   `treatment_delta`) overriding brittle regex — pass IL-TUR/court metadata to
   avoid treatment-polarity and recency-from-text errors.
5. **Negation-aware entailment fallback.** Polarity mismatch ("bail was *not*
   granted" vs "bail was granted") is damped instead of scoring as full support.
6. **Date-aware temporal layer wired in.** Pass `doc_meta={doc_id: {"decided": date,
   "authority_meta": {...}}}` to `run_trace_pipeline` to activate decided-date notes.
