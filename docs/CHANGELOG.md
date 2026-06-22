# Changelog — LegalAssist + TRACE-Law merge

## Combined release
### Added
- `src/pipeline.py` — unified entry (`build_engine` / `answer` / `evaluate`).
- `trace_law/` package — metrics, fusion (RRF), bm25, rerank, authority,
  temporal_validity, verification, reliability, generation, extractive,
  trace_pipeline, eval_ladder, integration.
- Gold-label evaluation + baseline ladder; `scripts/make_demo_eval.py`.
- Docs: README, SETUP, ARCHITECTURE, EVALUATION, CHANGELOG.

### Changed
- `config.py` defaults are now paper-aligned: `all-MiniLM-L6-v2`, `dense`, τ=0.35.
- `agents.RetrievalAgent` fuses with **RRF** (was max-cosine merge).
- `embedder.py`, `retriever.py`, `rag_pipeline.py` are **import-safe** (lazy heavy deps).
- `rag_pipeline.run_agentic_pipeline` now runs the full TRACE-Law pipeline.

### Removed
- `evaluation.py` pseudo-relevance rule (60% token overlap) and the top-1-cosine
  single-vs-multi ablation — they did not measure relevance.
- Duplicate extractive generator in `rag_pipeline` (consolidated in `trace_law.extractive`).
- `results/ablation_single_vs_multi.csv` (artifact of the removed ablation).

### Fixed (carried from TRACE-Law v2)
- Reliability no longer double-counts authority (pre-authority retrieval signal).
- Temporal penalty is graduated, not dead code.
- Margin edge case; negation-aware entailment fallback; authority metadata override;
  date-aware temporal layer wired in.

### Tests
- 36 offline tests (individual + end-to-end), all passing with no GPU/keys/network.
