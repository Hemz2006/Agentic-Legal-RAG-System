"""Download Indian court judgments from Hugging Face for the retriever."""
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "rishiai/indian-court-judgements-and-its-summaries"
DEFAULT_SPLIT = "train"
OUTPUT_PATH = Path("data/legal_cases.csv")
TEXT_COLUMN_CANDIDATES = [
    "Judgment",
    "judgment",
    "judgement",
    "text",
    "full_text",
    "case_text",
    "content",
    "Summary",
    "summary",
]


def choose_text_column(columns: list[str]) -> str:
    for column in TEXT_COLUMN_CANDIDATES:
        if column in columns:
            return column
    raise ValueError(
        "Could not find a judgment text column. "
        f"Available columns: {columns}. "
        f"Expected one of: {TEXT_COLUMN_CANDIDATES}"
    )


def main():
    parser = argparse.ArgumentParser(description="Download Indian court judgments from Hugging Face.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for local testing. Omit to save the full split.",
    )
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    print(f"Downloading from HuggingFace: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME, split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    df = pd.DataFrame(dataset)
    print("Columns available:", df.columns.tolist())

    text_column = choose_text_column(df.columns.tolist())
    df = df.rename(columns={text_column: "text"})
    df = df[["text"]].dropna()
    df = df[df["text"].astype(str).str.strip() != ""].drop_duplicates()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Using text column: {text_column}")
    print(f"Saved {len(df)} cases -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
