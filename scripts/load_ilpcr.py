"""Load the IL-PCR (Indian Legal Prior Case Retrieval) gold benchmark.

IMPORTANT: the GitHub repo `Exploration-Lab/IL-PCR` is the *U-CREAT code*; its
data folders are empty `.gitkeep` placeholders. The actual corpus + gold
citation links live on Hugging Face inside the IL-TUR benchmark under the
`pcr` config. This loader builds the three objects the evaluator needs:

    id_to_text : {candidate_id -> text}      # the searchable corpus
    queries    : {query_id     -> text}      # the query cases
    qrels      : {query_id      -> {relevant_candidate_id, ...}}  # gold answers

Usage (Colab or local), after `pip install datasets`:

    from scripts.load_ilpcr import load_ilpcr
    id_to_text, queries, qrels = load_ilpcr(split="test")          # full test set
    id_to_text, queries, qrels = load_ilpcr(split="dev", max_queries=30)  # quick

Then continue with Action-Guide steps 2.3 (build dense index) and 2.4 (evaluate).
"""
from __future__ import annotations

from typing import Dict, Optional, Set, Tuple


def _join(text) -> str:
    """IL-TUR stores each document's text as a list of sentences; join to a string."""
    if isinstance(text, list):
        return " ".join(str(s) for s in text)
    return str(text)


def load_ilpcr(
    split: str = "test",
    dataset_name: str = "Exploration-Lab/IL-TUR",
    config: str = "pcr",
    max_queries: Optional[int] = None,
    max_chars: Optional[int] = 60_000,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """Return (id_to_text, queries, qrels) for one IL-PCR split.

    split      : "train" | "dev" | "test"  (uses <split>_queries / <split>_candidates)
    max_queries: cap the number of queries (handy for a fast smoke run)
    max_chars  : truncate very long judgments so embedding stays in memory
    """
    from datasets import load_dataset  # lazy: only needed when actually loading

    ds = load_dataset(dataset_name, config)

    q_key, c_key = f"{split}_queries", f"{split}_candidates"
    if q_key not in ds or c_key not in ds:
        raise KeyError(
            f"Expected splits '{q_key}' and '{c_key}'. Available: {list(ds.keys())}"
        )

    def clip(s: str) -> str:
        return s[:max_chars] if max_chars else s

    # 1) the searchable corpus = candidate pool
    id_to_text: Dict[str, str] = {
        str(ex["id"]): clip(_join(ex["text"])) for ex in ds[c_key]
    }

    # 2) the query cases + 3) gold relevant candidates
    queries: Dict[str, str] = {}
    qrels: Dict[str, Set[str]] = {}
    for i, ex in enumerate(ds[q_key]):
        if max_queries is not None and i >= max_queries:
            break
        qid = str(ex["id"])
        queries[qid] = clip(_join(ex["text"]))
        # field is `relevant_candidates` in IL-TUR pcr; fall back to common names
        rel = ex.get("relevant_candidates") or ex.get("relevant_documents") or []
        qrels[qid] = {str(r) for r in rel}

    # keep only queries whose gold answers are actually in the candidate pool
    pool = set(id_to_text)
    for qid in list(qrels):
        qrels[qid] &= pool
        if not qrels[qid]:
            queries.pop(qid, None)
            qrels.pop(qid, None)

    print(
        f"IL-PCR[{split}]: {len(id_to_text)} candidates, "
        f"{len(queries)} queries with gold labels."
    )
    return id_to_text, queries, qrels


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test the IL-PCR loader.")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-queries", type=int, default=5)
    args = ap.parse_args()
    i2t, q, r = load_ilpcr(split=args.split, max_queries=args.max_queries)
    qid = next(iter(q))
    print("example query id :", qid)
    print("query text (200) :", q[qid][:200], "...")
    print("gold answers     :", r[qid])
