from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.schemas import SummaryRequest
from app.services.audit_logger import log_event
from app.services.mimic_loader import load_mvp_patient_notes
from app.services.phi_masking import mask_phi
from app.services.sentence_extractor import extract_relevant_sentences
from app.services.summarizer import generate_structured_summary
from app.services.verification import verify_summary

router = APIRouter()


@router.post("/summary")
def summarize_patient(req: SummaryRequest, db: Session = Depends(get_db)):
    df = load_mvp_patient_notes(max_patients=1, max_notes_per_patient=10)
    patient_df = df[df["SUBJECT_ID"].astype(int) == req.patient_id].copy()

    if patient_df.empty:
        raise HTTPException(status_code=404, detail="Patient not found in loaded dataset")

    notes = patient_df["TEXT"].astype(str).tolist()
    notes = [mask_phi(note) for note in notes]

    extracted = extract_relevant_sentences(notes)
    summary = generate_structured_summary(extracted)
    summary = {k: mask_phi(v) for k, v in summary.items()}
    warnings = verify_summary(summary, extracted)

    log_event(
        db,
        event_type="summary_generated",
        patient_id=req.patient_id,
        payload={"warning_count": len(warnings), "sections": list(summary.keys())},
    )

    return {
        "patient_id": req.patient_id,
        "summary": summary,
        "warnings": warnings,
    }