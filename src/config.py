"""Central configuration for the combined LegalAssist + TRACE-Law system.

Defaults are aligned with the paper: dense all-MiniLM-L6-v2 retrieval and
decision threshold tau = 0.35. Everything is overridable via environment
variables / a .env file.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_ROOT / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
except Exception:  # dotenv optional
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

# Paths
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "data" / "index"
RESULTS_DIR = ROOT_DIR / "results"
for _d in (DATA_DIR, INDEX_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "legal_cases.csv"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
TEXTS_PATH = INDEX_DIR / "texts.pkl"
INDEX_METADATA_PATH = INDEX_DIR / "metadata.json"

# --- Models (paper-aligned defaults) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")  # RTX 3050-friendly
GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "extractive")  # extractive | openai | local
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# --- Retrieval (paper-aligned defaults) ---
RETRIEVER_BACKEND = os.getenv("RETRIEVER_BACKEND", "dense").lower()  # dense | lexical
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "5000"))
RANDOM_SEED = 42
TOP_K = int(os.getenv("TOP_K", "10"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))

# --- TRACE-Law pipeline knobs ---
USE_RRF = os.getenv("USE_RRF", "1") not in ("0", "false", "False")
RRF_K = int(os.getenv("RRF_K", "60"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
FINAL_EVIDENCE = int(os.getenv("FINAL_EVIDENCE", "3"))
AUTHORITY_ALPHA = float(os.getenv("AUTHORITY_ALPHA", "0.3"))
ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.45"))
DECISION_TAU = float(os.getenv("DECISION_TAU", "0.35"))  # paper main threshold
DECISION_MIN_CASES = int(os.getenv("DECISION_MIN_CASES", "3"))

# --- Datasets ---
# Generic corpus used by the demo app (judgment + summary HF dataset).
LEGAL_DATASET_NAME = os.getenv("LEGAL_DATASET_NAME", "rishiai/indian-court-judgements-and-its-summaries")
LEGAL_DATASET_SPLIT = os.getenv("LEGAL_DATASET_SPLIT", "train")
# Gold-label retrieval benchmark (citation links as qrels).
ILPCR_REPO = os.getenv("ILPCR_REPO", "https://github.com/Exploration-Lab/IL-PCR.git")
ILTUR_DATASET_NAME = os.getenv("ILTUR_DATASET_NAME", "Exploration-Lab/IL-TUR")
ILTUR_DATASET_CONFIG = os.getenv("ILTUR_DATASET_CONFIG", "pcr")
# Largest Indian corpus (judgment prediction; no retrieval labels) - optional larger pool.
NYAYA_DATASET_NAME = os.getenv("NYAYA_DATASET_NAME", "")  # set to a NyayaAnumana HF id to use a sample

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
