"""
Pre-generates and caches summaries for all preprocessed patients in parallel.

Each patient runs in its own thread with its own DB session.
BioClinicalBERT and Ollama are shared resources, so we cap concurrency at 2
(enough to pipeline DB I/O with model inference, without OOM risk).

Usage:
    cd backend
    source .venv/bin/activate
    python -m scripts.warm_cache

    # Force-regenerate even for already-cached patients:
    python -m scripts.warm_cache --force

    # Process specific patients only:
    python -m scripts.warm_cache --patient-ids 95324 64925
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

from app.db.postgres import CachedSummary, Note, Patient, SessionLocal
from app.services.phi_masking import mask_phi
from app.services.sentence_extractor import extract_relevant_sentences
from app.services.summarizer import generate_structured_summary
from app.services.verification import verify_summary


def warm_patient(pid: int, force: bool = False) -> dict:
    """Generate and cache a summary for one patient.

    Uses its own DB session — safe to run concurrently with other patients.
    Returns a status dict for reporting.
    """
    db = SessionLocal()
    try:
        # Skip if already cached (unless --force)
        if not force:
            existing = db.query(CachedSummary).filter(
                CachedSummary.subject_id == pid
            ).first()
            if existing:
                return {"pid": pid, "status": "skipped", "reason": "already cached"}

        notes = db.query(Note).filter(Note.subject_id == pid).all()
        if not notes:
            return {"pid": pid, "status": "skipped", "reason": "no notes in DB"}

        note_dicts = [
            {
                "masked_text": n.masked_text,
                "row_id":      n.row_id or "",
                "chart_date":  n.chart_date or "",
                "category":    n.category or "",
            }
            for n in notes
        ]

        extracted = extract_relevant_sentences(note_dicts)
        if not extracted:
            return {"pid": pid, "status": "warn", "reason": "no sentences extracted"}

        summary, raw_citations = generate_structured_summary(extracted)
        summary = {k: mask_phi(v) for k, v in summary.items()}

        sentences_only = [item["sentence"] for item in extracted]
        warnings = verify_summary(summary, sentences_only)

        # Delete stale entry if --force
        if force:
            db.query(CachedSummary).filter(CachedSummary.subject_id == pid).delete()

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

        return {
            "pid": pid,
            "status": "ok",
            "sentences": len(extracted),
            "citations": len(raw_citations),
            "warnings":  len(warnings),
        }

    except Exception as exc:
        db.rollback()
        return {"pid": pid, "status": "error", "reason": str(exc)}

    finally:
        db.close()


def run(patient_ids: Optional[List[int]] = None, force: bool = False) -> None:
    print("=== MedMind AI Cache Warmer ===\n")
    t_start = time.time()

    # Fetch patient list from a short-lived session (not passed to threads)
    db = SessionLocal()
    try:
        if patient_ids:
            patients = db.query(Patient).filter(Patient.subject_id.in_(patient_ids)).all()
        else:
            patients = db.query(Patient).all()
    finally:
        db.close()

    if not patients:
        print("No patients found. Run scripts.preprocess first.")
        return

    pids = [p.subject_id for p in patients]
    print(f"Patients to process: {pids}\n")

    # Concurrency=2: pipelines DB I/O with model inference.
    # Higher values fight over BioClinicalBERT/GPU and offer diminishing returns.
    max_concurrent = min(2, len(pids))
    results = []

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(warm_patient, pid, force): pid for pid in pids}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            pid = r["pid"]
            if r["status"] == "ok":
                print(f"  ✓ Patient {pid}: cached "
                      f"({r['sentences']} sentences, {r['citations']} citations, "
                      f"{r['warnings']} warnings)")
            elif r["status"] == "skipped":
                print(f"  — Patient {pid}: {r['reason']}")
            else:
                print(f"  ✗ Patient {pid}: {r['status']} — {r.get('reason', '')}")

    ok    = sum(1 for r in results if r["status"] == "ok")
    skip  = sum(1 for r in results if r["status"] == "skipped")
    err   = sum(1 for r in results if r["status"] == "error")

    print(f"\n=== Cache warming complete in {time.time() - t_start:.1f}s ===")
    print(f"  Generated: {ok}  |  Skipped: {skip}  |  Errors: {err}")

    # Final count from a fresh session
    db = SessionLocal()
    try:
        print(f"  Total cached summaries in DB: {db.query(CachedSummary).count()}")
    finally:
        db.close()


def parse_args():
    parser = argparse.ArgumentParser(description="MedMind AI summary cache warmer")
    parser.add_argument(
        "--patient-ids", nargs="+", type=int, default=None,
        help="Only warm specific patient IDs. Defaults to all patients in DB."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate summaries even for already-cached patients."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(patient_ids=args.patient_ids, force=args.force)
