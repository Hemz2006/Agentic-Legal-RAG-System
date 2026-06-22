# Architecture

## Unified pipeline (`src/pipeline.py`)
`run_trace_pipeline` (in `trace_law/trace_pipeline.py`) is the single flow; `pipeline.py`
wires it to the repo engine and config.

| # | Stage | Module | Notes |
|---|-------|--------|-------|
| 1 | Query expansion | `agents.QueryExpansionAgent` | 4 fixed framings |
| 2 | Retrieve + fuse | `retriever` (dense) + `trace_law.bm25_index` + `trace_law.fusion` | RRF (k=60) replaces max-merge |
| 3 | Cross-encoder rerank | `trace_law.rerank` | real model or offline lexical fallback; gives the **pre-authority** retrieval signal |
| 4 | Authority reweight | `trace_law.authority` | court/bench/treatment/recency; metadata override |
| 5 | Temporal validity | `trace_law.temporal_validity` | IPC→BNS, CrPC→BNSS, IEA→BSA; date-aware via `doc_meta` |
| 6 | Generation | `trace_law.generation` + `trace_law.extractive` | Qwen / OpenAI / extractive |
| 7 | Citation verification | `trace_law.verification` | NLI or negation-aware lexical fallback |
| 8 | Reliability + abstain | `trace_law.reliability` | weighted blend − temporal penalty |

## Data flow
```
texts ──build_engine──► (id_to_text, dense, bm25)
query ──answer──► run_trace_pipeline ──► TraceResult{evidence, temporal, verification, reliability, answer}
queries+qrels ──evaluate──► run_ladder ──► metrics per system
```

## Import-safety
Heavy deps (faiss, torch, transformers, openai, sentence-transformers) are imported
**lazily inside functions**, so every module imports offline and the whole test-suite
runs with no GPU, no API key and no downloads (via deterministic fallbacks).

## What the merge removed/replaced
- `agents.RetrievalAgent` max-merge → **RRF** (`fusion='rrf'` default).
- `rag_pipeline` duplicate extractive generator → consolidated in `trace_law.extractive`.
- `evaluation.py` pseudo-relevance (60% token overlap) + top-1-cosine ablation → **removed**;
  replaced by gold-label metrics + the baseline ladder.
- `config.py` defaults InLegalBERT/lexical/τ=0.28 → **all-MiniLM-L6-v2 / dense / τ=0.35** (paper-aligned).
