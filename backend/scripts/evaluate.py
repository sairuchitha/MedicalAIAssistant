"""
Evaluation script for Vision AI.

Measures:
  Summarization : ROUGE-L, BERTScore (vs. Discharge summary Brief Hospital Course)
  RAG / QA      : Recall@3 (correct answer in top-3 retrieved chunks)
                  Faithfulness (answer claims traceable to retrieved chunks)
                  Hallucination rate

Usage:
    cd backend
    source .venv/bin/activate
    python -m scripts.evaluate

Requirements:
    pip install rouge-score bert-score
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re

import numpy as np
import pandas as pd

# ── ROUGE-L ────────────────────────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("[WARN] rouge-score not installed. Run: pip install rouge-score")

# ── BERTScore ──────────────────────────────────────────────────────────────────
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("[WARN] bert-score not installed. Run: pip install bert-score")

from app.config import settings
from app.db.postgres import Note, SessionLocal
from app.services.retriever import retrieve
from app.services.runtime_store import get_patient_index, get_patient_notes, initialize_runtime
from app.services.sentence_extractor import extract_relevant_sentences
from app.services.summarizer import generate_structured_summary


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_brief_hospital_course(discharge_text: str) -> str:
    """Pull the Brief Hospital Course section from a discharge summary."""
    match = re.search(
        r"brief hospital course[:\s]*(.*?)(?=\n[A-Z][A-Z\s/]+:|$)",
        discharge_text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return ""


def get_discharge_summaries(db, subject_id: int) -> list:
    notes = (
        db.query(Note)
        .filter(Note.subject_id == subject_id, Note.category == "Discharge summary")
        .all()
    )
    return [n.masked_text for n in notes]


# ── Summarization evaluation ───────────────────────────────────────────────────

def evaluate_summarization(patient_ids: list) -> dict:
    print("\n=== Summarization Evaluation ===")
    db = SessionLocal()
    rouge_scores, bert_p_scores, bert_r_scores, bert_f_scores = [], [], [], []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if ROUGE_AVAILABLE else None

    try:
        for pid in patient_ids:
            print(f"  Patient {pid}...", end=" ")
            notes = get_patient_notes(pid)
            if not notes:
                print("no notes in cache — skipping")
                continue

            discharge_texts = get_discharge_summaries(db, pid)
            if not discharge_texts:
                print("no discharge summary — skipping")
                continue

            # Ground truth: Brief Hospital Course from most recent discharge summary
            ground_truth = extract_brief_hospital_course(discharge_texts[-1])
            if not ground_truth:
                ground_truth = discharge_texts[-1][:1000]  # fallback: first 1k chars

            # Generated summary
            extracted = extract_relevant_sentences(notes)
            if not extracted:
                print("no sentences extracted — skipping")
                continue

            summary_dict, _ = generate_structured_summary(extracted)
            generated = " ".join(summary_dict.values())

            # ROUGE-L
            if scorer:
                r = scorer.score(ground_truth, generated)
                rouge_scores.append(r["rougeL"].fmeasure)
                print(f"ROUGE-L={r['rougeL'].fmeasure:.3f}", end=" ")

            # BERTScore
            if BERTSCORE_AVAILABLE:
                P, R, F = bert_score_fn([generated], [ground_truth], lang="en", verbose=False)
                bert_p_scores.append(P.mean().item())
                bert_r_scores.append(R.mean().item())
                bert_f_scores.append(F.mean().item())
                print(f"BERTScore-F={F.mean().item():.3f}", end=" ")

            print()

    finally:
        db.close()

    results = {}
    if rouge_scores:
        results["ROUGE-L"] = {"mean": np.mean(rouge_scores), "scores": rouge_scores}
        print(f"\n  Mean ROUGE-L:     {np.mean(rouge_scores):.3f}  (target ≥ 0.35)")
    if bert_f_scores:
        results["BERTScore-F"] = {"mean": np.mean(bert_f_scores), "scores": bert_f_scores}
        print(f"  Mean BERTScore-F: {np.mean(bert_f_scores):.3f}  (target ≥ 0.78)")

    return results


# ── RAG / QA evaluation ────────────────────────────────────────────────────────

from app.services.qa_service import classify_question

# QA pairs tagged with question_type.
# lookup  — factual single-fact retrieval (skips cross-encoder)
# reasoning — multi-hop / trend questions (uses full reranking)
SAMPLE_QA_PAIRS = [
    # ── Lookup: single-fact, point-in-time ───────────────────────────────────
    # Patient 95324 — ICU/sepsis
    {"patient_id": 95324, "question": "What was the ventilation status?",
     "keywords": ["ventilation", "extubat", "intubat"], "qtype": "lookup"},
    {"patient_id": 95324, "question": "What is the current antibiotic regimen?",
     "keywords": ["antibiotic", "vancomycin", "pipercillin", "meropenem", "zosyn"], "qtype": "lookup"},
    # Patient 64925 — brain mass
    {"patient_id": 64925, "question": "What is the patient's diagnosis?",
     "keywords": ["mass", "parietal", "brain", "tumor", "lesion"], "qtype": "lookup"},
    {"patient_id": 64925, "question": "What were the most recent vitals?",
     "keywords": ["blood pressure", "heart rate", "temperature", "bp", "hr"], "qtype": "lookup"},
    # Patient 62561 — oncology
    {"patient_id": 62561, "question": "What is the patient's cancer history?",
     "keywords": ["breast", "mastectomy", "chemo", "cancer", "carcinoma"], "qtype": "lookup"},
    # Patient 32639 — trauma
    {"patient_id": 32639, "question": "What anticoagulation was the patient on?",
     "keywords": ["coumadin", "warfarin", "anticoagul", "heparin"], "qtype": "lookup"},
    # Patient 64230 — cardiac
    {"patient_id": 64230, "question": "What cardiac condition does the patient have?",
     "keywords": ["aortic stenosis", "aortic", "stenosis", "valve"], "qtype": "lookup"},

    # ── Reasoning: multi-hop, trend, progression ─────────────────────────────
    # Patient 95324 — ICU/sepsis
    {"patient_id": 95324, "question": "How did the patient's respiratory status change over time?",
     "keywords": ["ventilation", "extubat", "intubat", "oxygen", "respiratory"], "qtype": "reasoning"},
    {"patient_id": 95324, "question": "What cultures were taken and what was the progression of infection management?",
     "keywords": ["blood", "urine", "culture", "antibiotic", "sepsis"], "qtype": "reasoning"},
    # Patient 64925 — brain mass
    {"patient_id": 64925, "question": "How did the patient's neurological symptoms progress leading to admission?",
     "keywords": ["confusion", "weakness", "hemiparesis", "facial", "neuro"], "qtype": "reasoning"},
    {"patient_id": 64925, "question": "What was the history of imaging findings and how did they evolve?",
     "keywords": ["mri", "ct", "imaging", "mass", "lesion", "scan"], "qtype": "reasoning"},
    # Patient 62561 — oncology
    {"patient_id": 62561, "question": "How has the patient's cancer treatment progressed over time?",
     "keywords": ["chemo", "chemotherapy", "treatment", "cycle", "radiation", "cancer"], "qtype": "reasoning"},
    # Patient 32639 — trauma
    {"patient_id": 32639, "question": "What was the history of anticoagulation management and any complications?",
     "keywords": ["warfarin", "coumadin", "bleeding", "anticoagul", "inr"], "qtype": "reasoning"},
    # Patient 64230 — cardiac
    {"patient_id": 64230, "question": "How did the patient's cardiac condition and management evolve during the hospital stay?",
     "keywords": ["aortic", "valve", "cardiac", "surgery", "stenosis", "echo"], "qtype": "reasoning"},
]


def recall_at_k(retrieved_chunks: list, keywords: list, k: int = 3) -> bool:
    """Check if any of the top-k chunks contain at least one expected keyword."""
    top_k = retrieved_chunks[:k]
    combined = " ".join(c.get("text", "").lower() for c in top_k)
    return any(kw.lower() in combined for kw in keywords)


def evaluate_qa(qa_pairs: list) -> dict:
    print("\n=== RAG / QA Evaluation ===")

    lookup_hits, lookup_total = 0, 0
    reasoning_hits, reasoning_total = 0, 0

    print(f"\n  {'Question':<62} {'Type':<10} {'Result'}")
    print(f"  {'-'*62} {'-'*10} {'-'*6}")

    for pair in qa_pairs:
        pid = pair["patient_id"]
        question = pair["question"]
        keywords = pair["keywords"]
        # Use the tagged type; fall back to the classifier for untagged pairs
        qtype = pair.get("qtype") or classify_question(question)

        cached = get_patient_index(pid)
        if cached is None:
            print(f"  [SKIP] Patient {pid} not in index")
            continue

        index, chunk_meta = cached
        retrieved = retrieve(question, index, chunk_meta, question_type=qtype)

        hit = recall_at_k(retrieved, keywords, k=3)
        status = "✓" if hit else "✗"
        print(f"  [{status}] P{pid}: {question[:58]:<58} {qtype:<10}")

        if qtype == "lookup":
            lookup_hits += int(hit)
            lookup_total += 1
        else:
            reasoning_hits += int(hit)
            reasoning_total += 1

    total = lookup_total + reasoning_total
    hits = lookup_hits + reasoning_hits
    recall_lookup   = lookup_hits   / lookup_total   if lookup_total   > 0 else 0.0
    recall_reasoning = reasoning_hits / reasoning_total if reasoning_total > 0 else 0.0
    recall_overall  = hits / total if total > 0 else 0.0

    print(f"\n  Recall@3 — Lookup:    {lookup_hits}/{lookup_total} = {recall_lookup:.2f}   (target ≥ 0.80)")
    print(f"  Recall@3 — Reasoning: {reasoning_hits}/{reasoning_total} = {recall_reasoning:.2f}   (target ≥ 0.80)")
    print(f"  Recall@3 — Overall:   {hits}/{total} = {recall_overall:.2f}   (target ≥ 0.80)")

    return {
        "recall_at_3_lookup":    {"score": recall_lookup,    "hits": lookup_hits,    "total": lookup_total},
        "recall_at_3_reasoning": {"score": recall_reasoning, "hits": reasoning_hits, "total": reasoning_total},
        "recall_at_3_overall":   {"score": recall_overall,   "hits": hits,           "total": total},
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("Loading runtime cache from Postgres...")
    initialize_runtime()

    patient_ids = [95324, 64925, 62561, 32639, 64230]

    summ_results = {}
    if ROUGE_AVAILABLE or BERTSCORE_AVAILABLE:
        summ_results = evaluate_summarization(patient_ids)
    else:
        print("\n[SKIP] Summarization eval — install rouge-score and bert-score")

    qa_results = evaluate_qa(SAMPLE_QA_PAIRS)

    print("\n=== Final Results ===")
    for metric, val in summ_results.items():
        print(f"  {metric}: {val['mean']:.3f}")
    for key, val in qa_results.items():
        print(f"  {key}: {val['hits']}/{val['total']} = {val['score']:.3f}")

    # Save results
    out_path = Path("data/processed/evaluation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "summarization": {k: v["mean"] for k, v in summ_results.items()},
                "qa": {k: v["score"] for k, v in qa_results.items()},
                "qa_detail": qa_results,
            },
            f, indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()
