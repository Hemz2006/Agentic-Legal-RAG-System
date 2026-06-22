import json
import logging
import pickle
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DATASET_PATH, FAISS_INDEX_PATH, INDEX_METADATA_PATH, TEXTS_PATH,
    EMBEDDING_MODEL, RETRIEVER_BACKEND, SAMPLE_SIZE, RANDOM_SEED, TOP_K,
    LEGAL_DATASET_NAME,
)
from embedder import embed_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class LexicalIndex:
    """Small local-safe retriever backend for demos on memory-constrained machines."""

    def __init__(self, texts: List[str]):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [(int(index), float(scores[index])) for index in indices if scores[index] > 0]


def _index_metadata_matches() -> bool:
    if not INDEX_METADATA_PATH.exists():
        return False
    with open(INDEX_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return (
        metadata.get("embedding_model") == EMBEDDING_MODEL
        and metadata.get("dataset_name") == LEGAL_DATASET_NAME
        and metadata.get("sample_size") == SAMPLE_SIZE
    )


def _embed(texts: List[str]):
    """Encode texts to L2-normalized float32 embeddings."""
    return embed_texts(texts)


def _load_texts() -> List[str]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run `python3 src/download_data.py` first."
        )

    logger.info("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

    return df["text"].astype(str).tolist()


def build_retriever(force_rebuild: bool = False):
    """Build a FAISS index (dense) or a TF-IDF LexicalIndex, returning (index, texts)."""
    if RETRIEVER_BACKEND == "lexical":
        texts = _load_texts()
        logger.info("Building lexical TF-IDF retriever over %s documents.", len(texts))
        return LexicalIndex(texts), texts

    import faiss  # lazy heavy import (dense path only)
    if (
        not force_rebuild
        and FAISS_INDEX_PATH.exists()
        and TEXTS_PATH.exists()
        and _index_metadata_matches()
    ):
        logger.info("Loading cached FAISS index from disk...")
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(TEXTS_PATH, "rb") as f:
            texts = pickle.load(f)
        logger.info(f"Loaded {len(texts)} documents from cache.")
        return index, texts

    texts = _load_texts()
    logger.info(f"Indexing {len(texts)} documents.")

    embeddings = _embed(texts)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim on normalized vectors
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
    with open(INDEX_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": EMBEDDING_MODEL,
                "dataset_name": LEGAL_DATASET_NAME,
                "sample_size": SAMPLE_SIZE,
                "document_count": len(texts),
                "embedding_dimension": int(embeddings.shape[1]),
            },
            f,
            indent=2,
        )
    logger.info(f"Index cached to {FAISS_INDEX_PATH}")

    return index, texts


def retrieve(
    query: str,
    index,
    texts: List[str],
    top_k: int = TOP_K,
) -> List[Tuple[str, float]]:
    """Retrieve top_k documents for a query, returning (text, similarity_score) tuples."""
    if not query or not query.strip():
        return []

    if isinstance(index, LexicalIndex):
        return [
            (texts[idx], score)
            for idx, score in index.search(query, top_k)
        ]

    query_emb = _embed([query])
    scores, indices = index.search(query_emb, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        results.append((texts[idx], float(score)))
    return results


if __name__ == "__main__":
    index, texts = build_retriever()
    query = "property dispute fraud case"
    print(f"\nQuery: {query}\n" + "=" * 60)
    for i, (doc, score) in enumerate(retrieve(query, index, texts), 1):
        print(f"\n[{i}] score={score:.4f}\n{doc[:400]}")
