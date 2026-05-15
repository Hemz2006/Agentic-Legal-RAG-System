"""Evaluation helpers for retrieval and answer-grounding experiments."""
import argparse
from collections import defaultdict
from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DATASET_PATH, TOP_K
from rag_pipeline import expand_query, multi_query_retrieve
from retriever import build_retriever


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", value.lower())
        if len(token) > 2
    }


def is_relevant(retrieved_text: str, relevant_items: set[str]) -> bool:
    """Match exact IDs or keyword-style relevance labels against retrieved text."""
    normalized_text = retrieved_text.lower()
    retrieved_tokens = _tokens(retrieved_text)

    for item in relevant_items:
        normalized_item = item.lower()
        if normalized_item == normalized_text or normalized_item in normalized_text:
            return True

        item_tokens = _tokens(item)
        if item_tokens and len(item_tokens & retrieved_tokens) / len(item_tokens) >= 0.6:
            return True

    return False


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item_id in top_k if is_relevant(item_id, relevant_ids))
    return hits / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    for rank, item_id in enumerate(retrieved_ids[:k], start=1):
        if is_relevant(item_id, relevant_ids):
            return 1.0 / rank
    return 0.0


def evaluate_rankings(rows: list[dict], k_precision: int = 5, k_mrr: int = 10) -> dict:
    precision_scores = []
    rr_scores = []

    for row in rows:
        retrieved_ids = row["retrieved_ids"]
        relevant_ids = set(row["relevant_ids"])
        precision_scores.append(precision_at_k(retrieved_ids, relevant_ids, k=k_precision))
        rr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids, k=k_mrr))

    return {
        f"precision@{k_precision}": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
        f"mrr@{k_mrr}": sum(rr_scores) / len(rr_scores) if rr_scores else 0.0,
        "queries": len(rows),
    }


def load_relevance_csv(path: str) -> dict[str, set[str]]:
    """Load a CSV with columns query_id, relevant_id."""
    df = pd.read_csv(path)
    required_columns = {"query_id", "relevant_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    grouped = defaultdict(set)
    for _, row in df.iterrows():
        grouped[str(row["query_id"])].add(str(row["relevant_id"]))
    return grouped


def run_ablation(queries: list[str], index, texts):
    """Compare single-query retrieval with agentic multi-query retrieval."""
    from retriever import retrieve

    results = []
    for query in queries:
        single = retrieve(query, index, texts, top_k=5)
        multi = multi_query_retrieve(query, index, texts, top_k=5)
        single_top_score = single[0][1] if single else 0.0
        multi_top_score = multi[0][1] if multi else 0.0
        if single_top_score:
            improvement_pct = ((multi_top_score - single_top_score) / single_top_score) * 100
        else:
            improvement_pct = 0.0

        results.append({
            "query": query,
            "single_top_score": single_top_score,
            "multi_top_score": multi_top_score,
            "improvement_pct": improvement_pct,
            "single_count": len(single),
            "multi_count": len(multi),
        })

    df = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    output_path = Path("results/ablation_single_vs_multi.csv")
    df.to_csv(output_path, index=False)

    average_improvement = df["improvement_pct"].mean() if not df.empty else 0.0
    print(df.to_string(index=False))
    print(f"\nSaved ablation results to {output_path}")
    print(f"Average top-score improvement: {average_improvement:.2f}%")
    return df


def _lexical_retrieve(query: str, vectorizer, matrix, texts: list[str], top_k: int = 5):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    top_indices = scores.argsort()[::-1][:top_k]
    return [(texts[index], float(scores[index])) for index in top_indices if scores[index] > 0]


def run_lexical_ablation(queries: list[str], top_k: int = 5):
    """Fallback ablation using TF-IDF when dense embedding index cannot run locally."""
    df = pd.read_csv(DATASET_PATH)
    texts = df["text"].dropna().astype(str).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)

    results = []
    for query in queries:
        single = _lexical_retrieve(query, vectorizer, matrix, texts, top_k=top_k)

        best_matches: dict[str, float] = {}
        for expanded in expand_query(query):
            for document, score in _lexical_retrieve(expanded, vectorizer, matrix, texts, top_k=top_k):
                if document not in best_matches or score > best_matches[document]:
                    best_matches[document] = score
        multi = sorted(best_matches.items(), key=lambda item: item[1], reverse=True)[:top_k]

        single_top_score = single[0][1] if single else 0.0
        multi_top_score = multi[0][1] if multi else 0.0
        if single_top_score:
            improvement_pct = ((multi_top_score - single_top_score) / single_top_score) * 100
        else:
            improvement_pct = 0.0

        results.append({
            "query": query,
            "backend": "tfidf_fallback",
            "single_top_score": single_top_score,
            "multi_top_score": multi_top_score,
            "improvement_pct": improvement_pct,
            "single_count": len(single),
            "multi_count": len(multi),
        })

    output = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    output_path = Path("results/ablation_single_vs_multi.csv")
    output.to_csv(output_path, index=False)

    average_improvement = output["improvement_pct"].mean() if not output.empty else 0.0
    print(output.to_string(index=False))
    print(f"\nSaved ablation results to {output_path}")
    print(f"Average top-score improvement: {average_improvement:.2f}%")
    return output


def faithfulness_review_prompt(answer: str, sources: list[str]) -> str:
    """Create a human/LLM review prompt for answer faithfulness checks."""
    source_text = "\n\n".join(f"[Source {idx + 1}]\n{text}" for idx, text in enumerate(sources))
    return f"""Evaluate whether the answer is fully supported by the sources.

Sources:
{source_text}

Answer:
{answer}

Return one of: Faithful, Partially Faithful, Not Faithful. Explain unsupported claims."""


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics from a relevance CSV.")
    parser.add_argument("--relevance-csv", required=True, help="CSV with query_id and relevant_id columns.")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run single-query vs multi-query retrieval ablation and save a CSV report.",
    )
    parser.add_argument(
        "--backend",
        choices=["dense", "lexical"],
        default="dense",
        help="Use dense FAISS retrieval or lexical TF-IDF fallback for ablation.",
    )
    args = parser.parse_args()

    relevance = load_relevance_csv(args.relevance_csv)

    if args.ablation:
        if args.backend == "lexical":
            run_lexical_ablation(list(relevance.keys()))
        else:
            index, texts = build_retriever()
            run_ablation(list(relevance.keys()), index, texts)
        return

    index, texts = build_retriever()

    rows = []
    for query_id, relevant_ids in relevance.items():
        retrieved = multi_query_retrieve(query_id, index, texts, top_k=max(TOP_K, 10))
        retrieved_ids = [doc for doc, _ in retrieved]
        rows.append({"retrieved_ids": retrieved_ids, "relevant_ids": relevant_ids})

    print(evaluate_rankings(rows))


if __name__ == "__main__":
    main()
