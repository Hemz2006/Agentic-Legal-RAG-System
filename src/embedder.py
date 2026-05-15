"""Embedding utilities for Indian legal semantic search."""
import logging
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, MAX_SEQUENCE_LENGTH

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None
_device = None


def get_embedding_model():
    """Lazy-load InLegalBERT or any compatible Hugging Face encoder."""
    global _tokenizer, _model, _device
    if _model is None or _tokenizer is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        _model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(_device)
        _model.eval()
    return _tokenizer, _model, _device


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    """Encode texts into L2-normalized float32 vectors for FAISS cosine search."""
    tokenizer, model, device = get_embedding_model()
    vectors = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())

    if not vectors:
        return np.empty((0, 0), dtype="float32")
    return np.vstack(vectors).astype("float32")
