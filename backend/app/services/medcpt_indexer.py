from typing import List, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from app.config import settings

article_tokenizer = AutoTokenizer.from_pretrained(settings.MEDCPT_ARTICLE_ENCODER)
article_model = AutoModel.from_pretrained(settings.MEDCPT_ARTICLE_ENCODER)
article_model.eval()


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    return masked.sum(1) / mask.sum(1).clamp(min=1e-9)


def embed_chunks(texts: List[str], batch_size: int = 8) -> np.ndarray:
    vectors = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = article_tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            outputs = article_model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors).astype("float32")


def build_faiss_index(chunks) -> Tuple[faiss.IndexFlatIP, List[dict]]:
    texts = [c["text"] for c in chunks]
    vectors = embed_chunks(texts)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index, chunks
