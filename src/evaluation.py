"""Gold-label retrieval evaluation (replaces the old pseudo-relevance ablation).

The previous top-1-cosine / 60%-token-overlap "ablation" has been removed: it
did not measure relevance. Evaluation now uses real gold qrels (e.g. IL-PCR
citation links) and standard IR metrics via trace_law.
"""
import argparse
from collections import defaultdict
from typing import Dict, Set

# Re-export the canonical gold-label metrics.
from trace_law.metrics import (  # noqa: F401
    evaluate, precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k, average_precision,
)
from trace_law.eval_ladder import format_ladder, run_ladder  # noqa: F401


def load_relevance_csv(path: str) -> Dict[str, Set[str]]:
    """Load qrels from a CSV with columns query_id, relevant_id."""
    import pandas as pd
    df = pd.read_csv(path)
    missing = {"query_id", "relevant_id"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    grouped: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        grouped[str(row["query_id"])].add(str(row["relevant_id"]))
    return grouped


def run_gold_eval(queries, qrels, id_to_text, dense=None, bm25=None,
                  rerank_score_fn=None, top_k=10, ks=(1, 5, 10)):
    """Convenience wrapper around the baseline ladder."""
    report = run_ladder(queries, qrels, id_to_text, dense_retriever=dense,
                        bm25_retriever=bm25, rerank_score_fn=rerank_score_fn, top_k=top_k, ks=ks)
    print(format_ladder(report, ks=ks))
    return report


def main():
    parser = argparse.ArgumentParser(description="Gold-label retrieval evaluation (baseline ladder).")
    parser.add_argument("--corpus-csv", help="CSV with columns: doc_id,text")
    parser.add_argument("--queries-csv", help="CSV with columns: query_id,text")
    parser.add_argument("--qrels-csv", help="CSV with columns: query_id,relevant_id")
    args = parser.parse_args()
    if not (args.corpus_csv and args.queries_csv and args.qrels_csv):
        parser.error("provide --corpus-csv, --queries-csv and --qrels-csv (see docs/EVALUATION.md)")

    import pandas as pd
    corpus = pd.read_csv(args.corpus_csv)
    id_to_text = {str(r.doc_id): str(r.text) for r in corpus.itertuples()}
    q = pd.read_csv(args.queries_csv)
    queries = {str(r.query_id): str(r.text) for r in q.itertuples()}
    qrels = load_relevance_csv(args.qrels_csv)

    from trace_law.bm25_index import BM25Index
    ids = list(id_to_text); texts = [id_to_text[i] for i in ids]
    bm = BM25Index(texts, doc_ids=ids)
    run_gold_eval(queries, qrels, id_to_text, dense=None, bm25=lambda x, k: bm.search(x, k))


if __name__ == "__main__":
    main()
