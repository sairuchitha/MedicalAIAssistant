import re
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from app.config import settings

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(settings.BIOCLINICALBERT_MODEL)
model = AutoModel.from_pretrained(settings.BIOCLINICALBERT_MODEL)
model = model.to(DEVICE)
model.eval()

ALWAYS_KEEP_PATTERNS = [
    r"\ballerg(y|ies)\b",
    r"\bmetformin\b",
    r"\binsulin\b",
    r"\bhypertension\b",
    r"\bdiabetes\b",
    r"\badmission\b",
    r"\bdischarge\b",
    r"\bchronic\b",
    r"\bmedication\b",
    r"\bdose\b",
    r"\bdiagnos",
]


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    return masked.sum(1) / mask.sum(1).clamp(min=1e-9)


def embed_sentences(sentences: List[str], batch_size: int = 32) -> np.ndarray:
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_embeds.append(pooled.cpu().numpy())
    return np.vstack(all_embeds)


def is_always_keep(sentence: str) -> bool:
    s = sentence.lower()
    return any(re.search(p, s) for p in ALWAYS_KEEP_PATTERNS)


def extract_relevant_sentences(notes: List[Dict], limit: int = 150) -> List[Dict]:
    """
    Accept a list of note dicts: {masked_text, row_id, chart_date, category, description}
    Return a list of sentence dicts: {sentence, note_id, note_date, note_type}
    preserving source provenance for citation generation.
    """
    all_items = []  # list of (sentence_text, note_id, note_date, note_type)
    for note in notes:
        text = note.get("masked_text", "")
        note_id = note.get("row_id", "")
        note_date = note.get("chart_date", "")
        note_type = note.get("category", "")
        for sent in split_sentences(text):
            all_items.append((sent, note_id, note_date, note_type))

    if not all_items:
        return []

    sentences = [item[0] for item in all_items]
    embeddings = embed_sentences(sentences)
    centroid = embeddings.mean(axis=0, keepdims=True)
    scores = cosine_similarity(embeddings, centroid).flatten()
    threshold = 0.60 * scores.max()

    selected = []
    for (sent, note_id, note_date, note_type), score in zip(all_items, scores):
        if is_always_keep(sent) or score >= threshold:
            selected.append({
                "sentence": sent,
                "note_id": note_id,
                "note_date": note_date,
                "note_type": note_type,
            })

    return selected[:limit]
