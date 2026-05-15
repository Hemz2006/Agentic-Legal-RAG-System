# Agentic Legal RAG System

An AI-powered legal research tool for the Indian judiciary domain. The system helps lawyers, law students, legal aid workers, and researchers search Indian court judgments using natural language and receive source-grounded legal analysis.

## Problem

Indian advocates and law students often lack affordable access to semantic search over large judgment corpora. Proprietary tools such as SCC Online and Manupatra are paid, Google remains mostly keyword-based, and general chatbots can hallucinate legal citations.

## Approach

- Domain-specific embeddings with `law-ai/InLegalBERT`
- FAISS vector search for fast semantic retrieval
- Agentic multi-query expansion to improve recall
- Deduplication and ranking before generation
- Decision-agent sufficiency check with automatic query refinement
- Analyzer-agent selection of the top three cases
- LLM answer generation grounded only in retrieved sources
- Streamlit UI with source transparency and similarity scores

## Datasets

The project is configured for Indian legal datasets from Hugging Face.

Default dataset:

```text
rishiai/indian-court-judgements-and-its-summaries
```

Optional gated benchmark dataset for separate research/evaluation:

```text
Exploration-Lab/IL-TUR
```

IL-TUR requires accepting the Hugging Face dataset terms and setting `HF_TOKEN` in `.env`.

## Agent Workflow

1. Query Agent: expands the user question into multiple legal search intents.
2. Retrieval Agent: searches FAISS using InLegalBERT embeddings.
3. Decision Agent: checks whether retrieved evidence is sufficient.
4. Refinement: expands retrieval again if the first pass is weak.
5. Analyzer Agent: filters and structures the top three cases.
6. LLM Reasoning Layer: explains the top cases, recommends the best match, and cites only retrieved sources.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
HF_TOKEN=optional_huggingface_token_for_gated_datasets
EMBEDDING_MODEL=law-ai/InLegalBERT
LEGAL_DATASET_NAME=rishiai/indian-court-judgements-and-its-summaries
LEGAL_DATASET_SPLIT=train
SAMPLE_SIZE=2000
```

Download and normalize Indian judgments:

```bash
python3 src/download_data.py
```

For a quick schema/download smoke test:

```bash
python3 src/download_data.py --limit 25
```

If you change the dataset or embedding model, rebuild the FAISS cache:

```bash
rm data/index/faiss.index data/index/texts.pkl
```

Run the app:

```bash
streamlit run src/app.py
```

## Evaluation

The evaluation scaffold supports:

- Precision@5
- MRR@10
- Answer faithfulness review prompt generation
- Generic embedding vs InLegalBERT ablation by switching `EMBEDDING_MODEL`

Example:

```bash
python3 src/evaluation.py --relevance-csv data/eval_queries.csv
```

Run the single-query vs multi-query ablation:

```bash
python3 src/evaluation.py --relevance-csv data/eval_queries.csv --ablation
```

If local dense InLegalBERT embedding is too slow or unavailable, run the evaluation fallback:

```bash
python3 src/evaluation.py --relevance-csv data/eval_queries.csv --ablation --backend lexical
```

This writes:

```text
results/ablation_single_vs_multi.csv
```

Expected relevance CSV columns:

```text
query_id,relevant_id
```

The repository includes `data/eval_queries.csv` as a starter smoke-test set. For rigorous evaluation, replace or extend it with judged query-document relevance pairs from IL-TUR/PCR or another annotated retrieval benchmark.

## Notes

- The retriever caches FAISS data in `data/index/faiss.index` and `data/index/texts.pkl`.
- This project disables Streamlit's file watcher in `.streamlit/config.toml` because the app is text-only and Streamlit can otherwise import optional `transformers` image modules.
- The generated answers are research assistance only and are not legal advice.
