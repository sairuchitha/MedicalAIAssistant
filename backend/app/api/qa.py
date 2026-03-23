from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.schemas import QARequest
from app.services.audit_logger import log_event
from app.services.phi_masking import mask_phi
from app.services.qa_service import answer_question, classify_question
from app.services.retriever import retrieve
from app.services.runtime_store import get_cached_qa, get_patient_index, set_cached_qa
from app.services.security import check_prompt_injection

router = APIRouter()


@router.post("/qa")
def ask_question(req: QARequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    blocked, hits = check_prompt_injection(req.question)
    if blocked:
        background_tasks.add_task(
            log_event,
            db,
            event_type="blocked_query",
            patient_id=req.patient_id,
            payload={"question": req.question, "hits": hits},
        )
        raise HTTPException(status_code=400, detail=f"Suspicious query blocked: {hits}")

    if get_patient_index(req.patient_id) is None:
        raise HTTPException(status_code=404, detail="Patient not found. Run preprocessing first.")

    # ── Return cached answer for identical questions ───────────────────────────
    cached_result = get_cached_qa(req.patient_id, req.question)
    if cached_result is not None:
        return cached_result

    # Classify once — reused by both retrieve() and answer_question()
    # so we never run the keyword classifier twice per request
    qtype = classify_question(req.question)

    index, chunk_meta = get_patient_index(req.patient_id)
    retrieved = retrieve(req.question, index, chunk_meta, question_type=qtype)

    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant notes found for this question.")

    result = answer_question(req.question, retrieved, question_type=qtype)
    result["answer"] = mask_phi(result["answer"])

    # ── Store in memory cache for future identical questions ───────────────────
    set_cached_qa(req.patient_id, req.question, result)

    # ── Audit log runs after response is sent — not in request path ───────────
    background_tasks.add_task(
        log_event,
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
