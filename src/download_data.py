"""Download an Indian court-judgment corpus from Hugging Face into data/legal_cases.csv.

Flexible: point --dataset at any HF dataset. Defaults to the generic judgment+
summary corpus; pass a NyayaAnumana / IL-TUR id with --limit to use a larger pool.
"""
import argparse
from pathlib import Path

OUTPUT_PATH = Path("data/legal_cases.csv")
TEXT_CANDIDATES = ["text", "Judgment", "judgment", "judgement", "full_text",
                   "case_text", "content", "Summary", "summary", "facts"]


def choose_text_column(columns):
    for c in TEXT_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(f"No judgment text column found. Available: {columns}")


def main():
    ap = argparse.ArgumentParser(description="Download an Indian legal corpus from Hugging Face.")
    ap.add_argument("--dataset", default="rishiai/indian-court-judgements-and-its-summaries")
    ap.add_argument("--config", default=None, help="HF dataset config name (optional)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="Row cap (e.g. 100000 for a NyayaAnumana sample)")
    ap.add_argument("--text-column", default=None, help="Force a specific text column")
    args = ap.parse_args()

    import pandas as pd
    from datasets import load_dataset

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    print(f"Downloading {args.dataset} (split={args.split})...")
    ds = load_dataset(args.dataset, args.config, split=args.split) if args.config \
        else load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    df = pd.DataFrame(ds)
    col = args.text_column or choose_text_column(df.columns.tolist())
    df = df.rename(columns={col: "text"})[["text"]].dropna()
    df = df[df["text"].astype(str).str.strip() != ""].drop_duplicates()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} documents (text column: {col}) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
