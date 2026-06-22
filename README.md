# LegalAssist + TRACE-Law

An open-source **agentic Retrieval-Augmented Generation (RAG)** system for searching
Indian court judgments, with a **reliability layer (TRACE-Law)** on top. This is the
**combined codebase**: the original LegalAssist engine (embeddings + FAISS + agents)
and the TRACE-Law upgrades (RRF fusion, cross-encoder reranking, authority weighting,
temporal-statutory validity, citation verification, reliability + abstention, and
gold-label evaluation) merged into one coherent project.

```
your question ──► expand (4 ways) ──► retrieve (BM25 + dense) ──► RRF fuse
       ──► cross-encoder rerank ──► authority reweight ──► temporal-validity flag
       ──► FIRAC generation (Qwen / OpenAI / extractive) ──► citation verification
       ──► reliability score ──► answer  OR  abstain
```

## Why two layers (and why the old engine stays)
TRACE-Law is **additive**. The original repo provides the *engine* (data loading,
embeddings, the FAISS dense index). TRACE-Law provides the *reasoning and evaluation*
around it. They live in one project; `src/pipeline.py` ties them together.

## Quickstart (offline, no GPU/keys needed)
```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 36 tests, fully offline
python scripts/demo_trace.py        # ladder + one full pipeline query
```

## Run the real system
```python
import sys; sys.path.insert(0, "src")
import pipeline

# texts = your corpus (list[str]); dense_index = a FAISS index built on those texts (optional)
id_to_text, dense, bm25 = pipeline.build_engine(texts, dense_index=my_faiss_index)
res = pipeline.answer("anticipatory bail cheating IPC 420", [dense, bm25], id_to_text)
print(res.answer)
print(res.as_dict())     # evidence, temporal flags, verification, reliability
```

## Evaluate (gold labels)
```python
report = pipeline.evaluate(queries, qrels, id_to_text, dense=dense, bm25=bm25)
print(pipeline.format_ladder(report))
```
Or via CLI: `python src/evaluation.py --corpus-csv c.csv --queries-csv q.csv --qrels-csv r.csv`.
See [`docs/EVALUATION.md`](docs/EVALUATION.md) for datasets and CSV formats.

## Docs
- [`docs/SETUP.md`](docs/SETUP.md) — step-by-step Colab / VS Code setup for beginners
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline & module map
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — datasets, qrels, the baseline ladder
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — what the merge changed
- [`src/trace_law/README_TRACE.md`](src/trace_law/README_TRACE.md) — TRACE-Law internals

## Project layout
```
src/
  config.py          paper-aligned settings (MiniLM dense, tau=0.35) + TRACE knobs
  embedder.py        HF encoder (lazy/import-safe)
  retriever.py       dense FAISS engine + TF-IDF lexical fallback
  agents.py          query expansion, RRF retrieval, decision, analyzer
  pipeline.py        UNIFIED entry: build_engine / answer / evaluate
  rag_pipeline.py    legacy app-compatible wrapper (delegates to pipeline)
  evaluation.py      gold-label metrics + ladder (old pseudo-relevance removed)
  download_data.py   fetch a corpus from Hugging Face
  app.py             Streamlit UI
  trace_law/         metrics, fusion, bm25, rerank, authority, temporal_validity,
                     verification, reliability, generation, extractive,
                     trace_pipeline, eval_ladder, integration
tests/               36 offline tests
scripts/             demo_trace.py, make_demo_eval.py
docs/                setup, architecture, evaluation, changelog
```

## Hardware notes
- Everything except the LLM runs on CPU/4 GB GPU.
- Local LLM on an RTX 3050 (4 GB): use **Qwen2.5-3B-Instruct** via Ollama.
- Embedding a large corpus: do it once on a Colab/Kaggle T4 and cache the FAISS index.

*Research assistance, not legal advice.*
