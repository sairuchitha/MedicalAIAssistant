from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.schemas import QARequest
from app.services.audit_logger import log_event
from app.services.chunker import chunk_note
from app.services.mimic_loader import load_mvp_patient_notes
from app.services.phi_masking import mask_phi
from app.services.qa_service import answer_question
from app.services.retriever import retrieve
from app.services.medcpt_indexer import build_faiss_index
from app.services.security import check_prompt_injection

router = APIRouter()


@router.post("/qa")
def ask_question(req: QARequest, db: Session = Depends(get_db)):
    blocked, hits = check_prompt_injection(req.question)
    if blocked:
        log_event(
            db,
            event_type="blocked_query",
            patient_id=req.patient_id,
            payload={"question": req.question, "hits": hits},
        )
        raise HTTPException(status_code=400, detail=f"Suspicious query blocked: {hits}")

    df = load_mvp_patient_notes(max_patients=1, max_notes_per_patient=10)
    patient_df = df[df["SUBJECT_ID"].astype(int) == req.patient_id].copy()

    if patient_df.empty:
        raise HTTPException(status_code=404, detail="Patient not found in loaded dataset")

    chunks = []
    for _, row in patient_df.iterrows():
        text = mask_phi(str(row["TEXT"]))
        note_chunks = chunk_note(
            text=text,
            patient_id=int(row["SUBJECT_ID"]),
            note_id=str(row.get("ROW_ID", "")),
            note_date=str(row.get("CHARTDATE", "")),
            note_type=str(row.get("CATEGORY", "")),
        )
        chunks.extend(note_chunks)

    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks available for this patient")

    index, chunk_meta = build_faiss_index(chunks)
    retrieved = retrieve(req.question, index, chunk_meta)
    result = answer_question(req.question, retrieved)

    result["answer"] = mask_phi(result["answer"])

    log_event(
        db,
        event_type="qa_generated",
        patient_id=req.patient_id,
        payload={
            "question": req.question,
            "question_type": result["question_type"],
            "citation_count": len(result["citations"]),
        },
    )

    return result