"""
Patient preprocessing pipeline — production-ready, config-driven.

Selects N clinically diverse patients from MIMIC-III NOTEEVENTS.csv,
applies PHI masking in parallel, chunks notes, builds per-patient FAISS
indexes, and stores everything in PostgreSQL.

Patient count is controlled entirely by MAX_PATIENTS in .env (default 5).
No patient IDs are hardcoded — selection is automatic based on quality
criteria (see select_patients below).

Usage:
    cd backend
    source .venv/bin/activate

    # Use MAX_PATIENTS from .env (default):
    python -m scripts.preprocess

    # Override patient count at runtime:
    MAX_PATIENTS=20 python -m scripts.preprocess

    # Process specific patient IDs only:
    python -m scripts.preprocess --patient-ids 95324 64925 62561
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

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

def load_noteevents(csv_path: str) -> pd.DataFrame:
    """Stream NOTEEVENTS.csv in 50k-row chunks, keeping only rows we care about."""
    print(f"  Reading {csv_path} ...")
    parts = []
    for chunk in pd.read_csv(csv_path, chunksize=50_000, low_memory=False):
        chunk["CATEGORY"] = chunk["CATEGORY"].fillna("").str.strip()
        chunk = chunk[chunk["ISERROR"].isna()]
        chunk = chunk[chunk["CATEGORY"].isin(IMPORTANT_NOTE_CATEGORIES)]
        chunk = chunk.dropna(subset=["SUBJECT_ID", "TEXT"])
        chunk["TEXT"] = chunk["TEXT"].astype(str).str.strip()
        chunk = chunk[chunk["TEXT"].str.split().str.len() >= settings.MIN_NOTE_WORDS]
        parts.append(chunk)

    df = pd.concat(parts, ignore_index=True)
    df["CHARTDATE"] = pd.to_datetime(df["CHARTDATE"], errors="coerce")
    return df.sort_values(["SUBJECT_ID", "CHARTDATE"])


def select_patients(df: pd.DataFrame, n: int) -> List[int]:
    """Auto-select the top-N patients using quality criteria.

    Criteria (applied in order):
      1. Must have a Discharge summary (enables ROUGE-L evaluation)
      2. Must have notes in >= 2 distinct categories (clinical diversity)
      3. Must have >= 20 qualifying notes (enough temporal depth for RAG)
      4. Ranked by note count descending; top-N selected

    This makes the pipeline fully config-driven — just set MAX_PATIENTS in .env.
    """
    has_discharge = set(
        df[df["CATEGORY"] == "Discharge summary"]["SUBJECT_ID"].unique()
    )
    category_diversity = df.groupby("SUBJECT_ID")["CATEGORY"].nunique()
    note_counts = df.groupby("SUBJECT_ID").size()

    qualified_ids = [
        pid for pid in df["SUBJECT_ID"].unique()
        if pid in has_discharge
        and category_diversity.get(pid, 0) >= 2
        and note_counts.get(pid, 0) >= 20
    ]

    if not qualified_ids:
        # Fallback: relax criteria — any patient with >= 2 notes
        print("  [WARN] No patients met strict criteria — relaxing to note_count >= 2")
        qualified_ids = note_counts[note_counts >= 2].index.tolist()

    ranked = note_counts[qualified_ids].sort_values(ascending=False)
    selected = ranked.head(n).index.astype(int).tolist()

    print(f"  Selected {len(selected)} patients from "
          f"{len(qualified_ids)} qualified candidates (pool: {df['SUBJECT_ID'].nunique()} total)")
    for pid in selected:
        cats = df[df["SUBJECT_ID"] == pid]["CATEGORY"].value_counts().to_dict()
        print(f"    Patient {pid}: {note_counts[pid]} notes — {cats}")

    return selected


# ── Per-note PHI masking (thread-safe) ────────────────────────────────────────

def _mask_and_chunk(row_data: dict) -> dict:
    """Mask PHI and produce chunks for a single note.  Thread-safe — Presidio
    and the chunker hold no mutable shared state across calls."""
    masked = mask_phi(str(row_data["text"]))
    chunks = chunk_note(
        text=masked,
        patient_id=row_data["patient_id"],
        note_id=str(row_data["row_id"]),
        note_date=str(row_data["note_date"]),
        note_type=str(row_data["note_type"]),
    )
    return {
        "row_id":    row_data["row_id"],
        "chart_date": row_data["note_date"],
        "category":  row_data["note_type"],
        "description": row_data["description"],
        "masked":    masked,
        "chunks":    chunks,
    }


# ── Per-patient processing ─────────────────────────────────────────────────────

def process_patient(pid: int, patient_df: pd.DataFrame, db) -> dict:
    """Full pipeline for one patient: mask → chunk → index → store."""
    t0 = time.time()

    # Clear existing data (idempotent re-runs)
    for model in (Note, FaissIndex, CachedSummary, Patient):
        db.query(model).filter(model.subject_id == pid).delete()

    categories = patient_df["CATEGORY"].unique().tolist()
    db.add(Patient(subject_id=pid, note_count=len(patient_df), categories=categories))

    # Build row_data list for parallel masking
    rows = [
        {
            "patient_id":  pid,
            "row_id":      str(row.get("ROW_ID", "")),
            "note_date":   str(row["CHARTDATE"].date()) if pd.notna(row.get("CHARTDATE")) else "",
            "note_type":   str(row.get("CATEGORY", "")),
            "description": str(row.get("DESCRIPTION", "")),
            "text":        str(row["TEXT"]),
        }
        for _, row in patient_df.iterrows()
    ]

    # Parallel PHI masking — notes are independent, Presidio is thread-safe
    all_chunks = []
    mask_workers = min(4, len(rows))
    with ThreadPoolExecutor(max_workers=mask_workers) as ex:
        futures = {ex.submit(_mask_and_chunk, r): r["row_id"] for r in rows}
        done = 0
        for future in as_completed(futures):
            res = future.result()
            done += 1
            print(f"    [{pid}] Masked {done}/{len(rows)} notes\r", end="", flush=True)
            db.add(Note(
                subject_id=pid,
                row_id=res["row_id"],
                chart_date=res["chart_date"],
                category=res["category"],
                description=res["description"],
                masked_text=res["masked"],
                word_count=len(res["masked"].split()),
            ))
            all_chunks.extend(res["chunks"])

    print(f"    [{pid}] PHI masking done — {len(all_chunks)} chunks            ")

    # Build FAISS index
    print(f"    [{pid}] Building FAISS index ...")
    index, chunk_meta = build_faiss_index(all_chunks)
    index_bytes = faiss.serialize_index(index).tobytes()

    db.add(FaissIndex(subject_id=pid, index_data=index_bytes, chunk_meta=chunk_meta))
    db.commit()

    elapsed = time.time() - t0
    return {"pid": pid, "notes": len(rows), "chunks": len(all_chunks), "elapsed": elapsed}


# ── Summary cache warm-up (reuses existing warm_cache logic) ──────────────────

def warm_summaries(patient_ids: List[int], db) -> None:
    from app.services.phi_masking import mask_phi as _mask
    from app.services.sentence_extractor import extract_relevant_sentences
    from app.services.summarizer import generate_structured_summary
    from app.services.verification import verify_summary

    print("\n── Pre-warming summary cache ──")
    for pid in patient_ids:
        existing = db.query(CachedSummary).filter(CachedSummary.subject_id == pid).first()
        if existing:
            print(f"  Patient {pid}: summary already cached — skipping")
            continue

        notes = db.query(Note).filter(Note.subject_id == pid).all()
        if not notes:
            continue

        note_dicts = [
            {"masked_text": n.masked_text, "row_id": n.row_id or "",
             "chart_date": n.chart_date or "", "category": n.category or ""}
            for n in notes
        ]

        print(f"  Patient {pid}: extracting sentences ...")
        extracted = extract_relevant_sentences(note_dicts)
        if not extracted:
            print(f"  [WARN] Patient {pid}: no sentences extracted — skipping")
            continue

        print(f"  Patient {pid}: generating summary ({len(extracted)} sentences) ...")
        summary, raw_citations = generate_structured_summary(extracted)
        summary = {k: _mask(v) for k, v in summary.items()}
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
        print(f"  Patient {pid}: cached ({len(raw_citations)} citations, {len(warnings)} warnings)")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MedMind AI preprocessing pipeline")
    parser.add_argument(
        "--patient-ids", nargs="+", type=int, default=None,
        help="Process specific patient IDs. If omitted, auto-selects MAX_PATIENTS from CSV."
    )
    parser.add_argument(
        "--skip-cache", action="store_true",
        help="Skip summary cache warm-up after indexing."
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to NOTEEVENTS.csv (overrides settings.MIMIC_NOTEEVENTS_CSV)"
    )
    return parser.parse_args()


def run(
    patient_ids: Optional[List[int]] = None,
    skip_cache: bool = False,
    csv_path: Optional[str] = None,
):
    print("=== MedMind AI Preprocessing ===\n")
    t_start = time.time()

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        csv = csv_path or settings.MIMIC_NOTEEVENTS_CSV
        df = load_noteevents(csv)

        # Apply per-patient note cap (most recent MAX_NOTES_PER_PATIENT notes)
        df = df.groupby("SUBJECT_ID").tail(settings.MAX_NOTES_PER_PATIENT).reset_index(drop=True)
        df = deduplicate_notes(df)

        # Determine which patients to process
        if patient_ids:
            pids = [p for p in patient_ids if p in df["SUBJECT_ID"].values]
            missing = set(patient_ids) - set(pids)
            if missing:
                print(f"  [WARN] Patient IDs not found in CSV: {missing}")
        else:
            pids = select_patients(df, n=settings.MAX_PATIENTS)

        if not pids:
            print("No patients to process. Check your CSV path and filters.")
            return

        print(f"\nProcessing {len(pids)} patients ...\n")

        results = []
        for pid in pids:
            patient_df = df[df["SUBJECT_ID"] == pid]
            print(f"── Patient {pid} ({len(patient_df)} notes) ──")
            r = process_patient(pid, patient_df, db)
            results.append(r)
            print(f"   Done in {r['elapsed']:.1f}s  "
                  f"({r['notes']} notes, {r['chunks']} chunks)\n")

        # Summary
        print("── Indexing summary ──")
        print(f"  Patients:  {db.query(Patient).count()}")
        print(f"  Notes:     {db.query(Note).count()}")
        print(f"  Indexes:   {db.query(FaissIndex).count()}")

        if not skip_cache:
            warm_summaries(pids, db)

        print(f"\n  Cached summaries: {db.query(CachedSummary).count()}")
        print(f"\n=== Done in {time.time() - t_start:.1f}s ===")

    finally:
        db.close()


if __name__ == "__main__":
    args = parse_args()
    run(
        patient_ids=args.patient_ids,
        skip_cache=args.skip_cache,
        csv_path=args.csv,
    )
