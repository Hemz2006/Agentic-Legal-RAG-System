"""Central configuration for the Legal RAG system."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

# Load root .env first, then src/.env as a fallback/override for local IDE setups.
load_dotenv(ROOT_DIR / ".env")
load_dotenv(SRC_DIR / ".env", override=True)

# Paths
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "data" / "index"
RESULTS_DIR = ROOT_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Files
DATASET_PATH = DATA_DIR / "legal_cases.csv"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
TEXTS_PATH = INDEX_DIR / "texts.pkl"
INDEX_METADATA_PATH = INDEX_DIR / "metadata.json"

# Model config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "law-ai/InLegalBERT")
GENERIC_EMBEDDING_MODEL = os.getenv("GENERIC_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0.2

# Retrieval config
RETRIEVER_BACKEND = os.getenv("RETRIEVER_BACKEND", "lexical").lower()
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "5000"))
RANDOM_SEED = 42
TOP_K = 7
EMBEDDING_BATCH_SIZE = 128
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))

# Dataset config
LEGAL_DATASET_NAME = os.getenv("LEGAL_DATASET_NAME", "rishiai/indian-court-judgements-and-its-summaries")
LEGAL_DATASET_CONFIG = os.getenv("LEGAL_DATASET_CONFIG") or None
LEGAL_DATASET_SPLIT = os.getenv("LEGAL_DATASET_SPLIT", "train")
LEGAL_TEXT_COLUMNS = [
    column.strip()
    for column in os.getenv(
        "LEGAL_TEXT_COLUMNS",
        "text,judgment,judgement,full_text,content,summary,facts,legal_issues,judgment_reason",
    ).split(",")
    if column.strip()
]

# Optional gated benchmark dataset for evaluation/research.
ILTUR_DATASET_NAME = os.getenv("ILTUR_DATASET_NAME", "Exploration-Lab/IL-TUR")
ILTUR_DATASET_CONFIG = os.getenv("ILTUR_DATASET_CONFIG", "pcr")
HF_TOKEN = os.getenv("HF_TOKEN")

# API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
