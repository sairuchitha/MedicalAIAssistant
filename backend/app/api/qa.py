from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.schemas import QARequest
from app.services.audit_logger import log_event
from app.services.phi_masking import mask_phi
from app.services.qa_service import answer_question
from app.services.retriever import retrieve
from app.services.runtime_store import get_patient_index
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

    cached = get_patient_index(req.patient_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="Patient not found. Run preprocessing first.")

    index, chunk_meta = cached
    retrieved = retrieve(req.question, index, chunk_meta)

    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant notes found for this question.")

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
