"""
One-time preprocessing script.

Selects 5 clinically diverse patients from MIMIC-III NOTEEVENTS.csv,
applies PHI masking, chunks notes, builds per-patient FAISS indexes,
and stores everything in PostgreSQL.

Run once before starting the server:
    cd backend
    source .venv/bin/activate
    python -m scripts.preprocess

Why these 5 patients:
  - All have Physician progress notes (richest for QA) AND Discharge summaries (ground truth for evaluation)
  - 50-60 clinical notes each — enough temporal depth for trend/reasoning questions
  - Diverse clinical profiles: ICU/sepsis, neuro-oncology, oncology, trauma, cardiac
  - Selected from 7,442 candidates with both Physician + Discharge summary notes
"""

import sys
from pathlib import Path

# Ensure backend/ is on the path regardless of how the script is invoked
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import faiss
import numpy as np
import pandas as pd

from app.config import settings
from app.db.postgres import (
    Base, CachedSummary, FaissIndex, Note, Patient, SessionLocal, engine
)
from app.services.chunker import chunk_note
from app.services.dedup import deduplicate_notes
from app.services.medcpt_indexer import build_faiss_index
from app.services.phi_masking import mask_phi
from app.utils.constants import IMPORTANT_NOTE_CATEGORIES

# ── Patient selection ──────────────────────────────────────────────────────────
# These 5 patients were hand-picked via EDA from 7,442 candidates that have BOTH
# Physician notes (avg 7k chars, rich for QA) and Discharge summaries (ground
# truth for ROUGE-L / BERTScore evaluation).  Each has 50-60 clinical notes
# covering at least 4 note categories, ensuring the RAG index contains varied
# temporal evidence for both lookup and reasoning questions.
DEMO_PATIENT_IDS = [95324, 64925, 62561, 32639, 64230]

# Clinical profiles (for documentation / demo narrative):
# 95324 — ICU/sepsis: blood cultures, ventilation weaning, liver involvement
# 64925 — Neuro: 48M newly-diagnosed posterior brain mass, hemiparesis
# 62561 — Oncology: 29F breast cancer with cerebellar metastasis
# 32639 — Trauma: 87F Afib on coumadin, hip fracture, retroperitoneal bleed
# 64230 — Cardiac: aortic stenosis s/p femoral bypass graft

MAX_NOTES_PER_PATIENT = 60  # median for this cohort is 40; 60 gives temporal depth


def load_patient_notes(csv_path: str) -> pd.DataFrame:
    print(f"Reading CSV: {csv_path}")
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=50_000, low_memory=False):
        chunk["CATEGORY"] = chunk["CATEGORY"].str.strip()
        chunk = chunk[chunk["SUBJECT_ID"].isin(DEMO_PATIENT_IDS)]
        chunk = chunk[chunk["ISERROR"].isna()]
        chunk = chunk[chunk["CATEGORY"].isin(IMPORTANT_NOTE_CATEGORIES)]
        chunk = chunk.dropna(subset=["SUBJECT_ID", "TEXT"])
        chunk["TEXT"] = chunk["TEXT"].astype(str).str.strip()
        chunk = chunk[chunk["TEXT"].str.split().str.len() >= settings.MIN_NOTE_WORDS]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df["CHARTDATE"] = pd.to_datetime(df["CHARTDATE"], errors="coerce")
    df = df.sort_values(["SUBJECT_ID", "CHARTDATE"])

    # Keep most recent MAX_NOTES_PER_PATIENT notes per patient
    df = df.groupby("SUBJECT_ID").tail(MAX_NOTES_PER_PATIENT).reset_index(drop=True)
    return df


def serialize_faiss(index) -> bytes:
    buf = faiss.serialize_index(index)
    return buf.tobytes()


def run():
    print("=== Vision AI Preprocessing ===\n")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        csv_path = settings.MIMIC_NOTEEVENTS_CSV
        df = load_patient_notes(csv_path)

        # Deduplicate within each patient
        df = deduplicate_notes(df)

        print(f"\nLoaded {len(df)} notes across {df['SUBJECT_ID'].nunique()} patients")
        print(df.groupby("SUBJECT_ID")["CATEGORY"].value_counts().to_string())

        for pid in DEMO_PATIENT_IDS:
            patient_df = df[df["SUBJECT_ID"] == pid]
            if patient_df.empty:
                print(f"\n[WARN] Patient {pid} not found — skipping")
                continue

            print(f"\n── Patient {pid}: {len(patient_df)} notes ──")

            # Clear existing data for this patient (idempotent re-runs)
            db.query(Note).filter(Note.subject_id == pid).delete()
            db.query(FaissIndex).filter(FaissIndex.subject_id == pid).delete()
            db.query(CachedSummary).filter(CachedSummary.subject_id == pid).delete()
            db.query(Patient).filter(Patient.subject_id == pid).delete()

            # ── Store patient metadata ─────────────────────────────────────
            categories = patient_df["CATEGORY"].unique().tolist()
            db.add(Patient(
                subject_id=pid,
                note_count=len(patient_df),
                categories=categories,
            ))

            # ── Mask PHI and store notes ───────────────────────────────────
            all_chunks = []
            for _, row in patient_df.iterrows():
                print(f"  Masking note {row['ROW_ID']} ({row['CATEGORY']})...", end="\r")
                masked = mask_phi(str(row["TEXT"]))

                db.add(Note(
                    subject_id=pid,
                    row_id=str(row.get("ROW_ID", "")),
                    chart_date=str(row["CHARTDATE"].date()) if pd.notna(row.get("CHARTDATE")) else None,
                    category=str(row.get("CATEGORY", "")),
                    description=str(row.get("DESCRIPTION", "")),
                    masked_text=masked,
                    word_count=len(masked.split()),
                ))

                note_chunks = chunk_note(
                    text=masked,
                    patient_id=pid,
                    note_id=str(row.get("ROW_ID", "")),
                    note_date=str(row["CHARTDATE"].date()) if pd.notna(row.get("CHARTDATE")) else "",
                    note_type=str(row.get("CATEGORY", "")),
                )
                all_chunks.extend(note_chunks)

            print(f"  PHI masking done. {len(all_chunks)} chunks generated.          ")

            # ── Build FAISS index ──────────────────────────────────────────
            print(f"  Building FAISS index with MedCPT embeddings...")
            index, chunk_meta = build_faiss_index(all_chunks)
            index_bytes = serialize_faiss(index)

            db.add(FaissIndex(
                subject_id=pid,
                index_data=index_bytes,
                chunk_meta=chunk_meta,
            ))

            db.commit()
            print(f"  Stored in Postgres. Index size: {len(index_bytes):,} bytes")

        db.commit()

        # ── Pre-warm summary cache ─────────────────────────────────────────────
        print("\n── Pre-warming summary cache ──")
        from app.services.phi_masking import mask_phi
        from app.services.sentence_extractor import extract_relevant_sentences
        from app.services.summarizer import generate_structured_summary
        from app.services.verification import verify_summary

        for pid in DEMO_PATIENT_IDS:
            notes = db.query(Note).filter(Note.subject_id == pid).all()
            if not notes:
                continue
            note_dicts = [
                {
                    "masked_text": n.masked_text,
                    "row_id": n.row_id or "",
                    "chart_date": n.chart_date or "",
                    "category": n.category or "",
                }
                for n in notes
            ]
            print(f"  Patient {pid}: generating summary...")
            extracted = extract_relevant_sentences(note_dicts)
            if not extracted:
                print(f"  [WARN] No sentences extracted — skipping")
                continue
            summary, raw_citations = generate_structured_summary(extracted)
            summary = {k: mask_phi(v) for k, v in summary.items()}
            sentences_only = [item["sentence"] for item in extracted]
            warnings = verify_summary(summary, sentences_only)
            db.add(CachedSummary(
                subject_id=pid,
                chief_complaint=summary.get("Chief Complaint"),
                active_diagnoses=summary.get("Active Diagnoses"),
                current_medications=summary.get("Current Medications"),
                recent_history=summary.get("Recent History and Care Plan"),
                citations=raw_citations,
                warnings=warnings,
            ))
            db.commit()
            print(f"  Patient {pid}: cached ({len(raw_citations)} citations)")

        print("\n=== Preprocessing complete ===")
        print(f"Patients stored:  {db.query(Patient).count()}")
        print(f"Notes stored:     {db.query(Note).count()}")
        print(f"FAISS indexes:    {db.query(FaissIndex).count()}")
        print(f"Cached summaries: {db.query(CachedSummary).count()}")

    finally:
        db.close()


if __name__ == "__main__":
    run()
