# Evaluation

## Datasets
| Role | Dataset | Size | Labels |
|------|---------|------|--------|
| **Primary eval** | IL-PCR (`github.com/Exploration-Lab/IL-PCR`, or HF `Exploration-Lab/IL-TUR` task `pcr`) | 7,070 docs / 1,182 queries | citation links = gold qrels |
| Stretch (statutes) | IL-PCSR (HF `Exploration-Lab/IL-PCSR`, **gated**) | + statute labels | precedent + statute qrels |
| Larger corpus pool | NyayaAnumana | ~2.28 M cases | none for retrieval (judgment prediction) — use as distractor pool only |

IL-PCR is the right benchmark: it is ungated, already cited by the paper, and ships gold
labels. NyayaAnumana is the largest Indian corpus but has **no retrieval relevance labels**,
so use a sample only to enlarge the candidate pool, never as the eval set.

## The three inputs the evaluator needs
```python
id_to_text = {"d0": "judgment text ...", ...}   # every candidate document
queries    = {"q0": "query case text ...", ...}
qrels      = {"q0": {"d7", "d42"}, ...}          # gold relevant doc ids (citations)
```

## Run it
```python
import pipeline
report = pipeline.evaluate(queries, qrels, id_to_text, dense=dense, bm25=bm25,
                           rerank_score_fn=load_cross_encoder())  # None -> offline fallback
print(pipeline.format_ladder(report))
```
CLI (CSV inputs `doc_id,text` / `query_id,text` / `query_id,relevant_id`):
```bash
python src/evaluation.py --corpus-csv corpus.csv --queries-csv queries.csv --qrels-csv qrels.csv
```

## The baseline ladder (what gets reported)
BM25 · Dense-single · Dense-multi (max-merge) · Dense-multi (RRF) · Dense-multi (RRF)+rerank,
each scored with Precision@k, Recall@k, nDCG@k, MRR@10, MAP@10.

## For the paper
- Report the ladder with confidence intervals + a paired significance test between rungs.
- Add a leave-one-out ablation (drop RRF / rerank / authority / temporal).
- Validate the temporal detector (precision/recall) against the official IPC→BNS table.
- Plot a risk–coverage curve for the abstention threshold.
