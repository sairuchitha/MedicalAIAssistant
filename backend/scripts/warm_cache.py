"""
Pre-generates and caches summaries for all preprocessed patients.

Run once after preprocessing:
    cd backend
    source .venv/bin/activate
    python -m scripts.warm_cache
"""

import sys
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app.db.postgres import CachedSummary, Note, Patient, SessionLocal
from app.services.phi_masking import mask_phi
from app.services.sentence_extractor import extract_relevant_sentences
from app.services.summarizer import generate_structured_summary
from app.services.verification import verify_summary


def run():
    print("=== Vision AI Cache Warmer ===\n")
    db = SessionLocal()

    try:
        patients = db.query(Patient).all()
        if not patients:
            print("No patients found. Run scripts.preprocess first.")
            return

        for patient in patients:
            pid = patient.subject_id

            # Skip if already cached
            existing = db.query(CachedSummary).filter(
                CachedSummary.subject_id == pid
            ).first()
            if existing:
                print(f"Patient {pid}: already cached — skipping")
                continue

            print(f"Patient {pid}: generating summary...")

            notes = db.query(Note).filter(Note.subject_id == pid).all()
            note_dicts = [
                {
                    "masked_text": n.masked_text,
                    "row_id": n.row_id or "",
                    "chart_date": n.chart_date or "",
                    "category": n.category or "",
                }
                for n in notes
            ]

            extracted = extract_relevant_sentences(note_dicts)
            if not extracted:
                print(f"  [WARN] No sentences extracted — skipping")
                continue

            print(f"  Extracted {len(extracted)} sentences. Running LLM (parallel)...")
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

            print(f"  Cached. {len(raw_citations)} citations, {len(warnings)} warnings.")

        print("\n=== Cache warming complete ===")
        print(f"Cached summaries: {db.query(CachedSummary).count()}")

    finally:
        db.close()


if __name__ == "__main__":
    run()
