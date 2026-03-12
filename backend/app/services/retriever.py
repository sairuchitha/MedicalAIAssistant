from typing import List

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from app.config import settings

query_tokenizer = AutoTokenizer.from_pretrained(settings.MEDCPT_QUERY_ENCODER)
query_model = AutoModel.from_pretrained(settings.MEDCPT_QUERY_ENCODER)
query_model.eval()

cross_tokenizer = AutoTokenizer.from_pretrained(settings.MEDCPT_CROSS_ENCODER)
cross_model = AutoModelForSequenceClassification.from_pretrained(settings.MEDCPT_CROSS_ENCODER)
cross_model.eval()


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    return masked.sum(1) / mask.sum(1).clamp(min=1e-9)


def embed_query(question: str) -> np.ndarray:
    with torch.no_grad():
        inputs = query_tokenizer(question, return_tensors="pt", truncation=True, max_length=256)
        outputs = query_model(**inputs)
        pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        vector = pooled.cpu().numpy().astype("float32")
        faiss.normalize_L2(vector)
        return vector


def rerank(question: str, candidate_chunks: List[dict]) -> List[dict]:
    if not candidate_chunks:
        return []
    pairs = [(question, c["text"]) for c in candidate_chunks]
    inputs = cross_tokenizer(pairs, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = cross_model(**inputs).logits.squeeze(-1).cpu().numpy()
    ranked = sorted(zip(candidate_chunks, logits), key=lambda x: x[1], reverse=True)
    return [item[0] for item in ranked]


def retrieve(question: str, index, chunks, top_k: int = 10) -> List[dict]:
    q = embed_query(question)
    scores, ids = index.search(q, top_k)
    candidates = [chunks[i] for i in ids[0] if i != -1]
    return rerank(question, candidates)
