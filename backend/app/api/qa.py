from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.schemas import QARequest
from app.services.audit_logger import log_event
from app.services.phi_masking import mask_phi
from app.services.qa_service import answer_question, classify_question
# security MUST be imported before retriever: GPT-2 (CPU) loads at security import
# time and must be in memory before MedCPT models load on MPS (import order fix for
# Apple Silicon Metal segfault — exit 139 when load order is reversed)
from app.services.security import check_indirect_injection, check_prompt_injection, detect_suspicious_intent, validate_output
from app.services.retriever import retrieve
from app.services.runtime_store import get_cached_qa, get_patient_index, set_cached_qa

router = APIRouter()


@router.post("/qa")
def ask_question(req: QARequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    blocked, block_details = check_prompt_injection(req.question)

    intent_flag, intent_reason = detect_suspicious_intent(req.question)

    if intent_flag:
        block_details["intent_flag"] = intent_reason
        blocked = True

    if blocked:
        background_tasks.add_task(
            log_event,
            db,
            event_type="blocked_query",
            patient_id=req.patient_id,
            payload={"question_length": len(req.question), **block_details},
        )
        raise HTTPException(status_code=400, detail="Query not allowed.")

    if get_patient_index(req.patient_id) is None:
        raise HTTPException(status_code=404, detail="Patient not found. Run preprocessing first.")

    # ── Return cached answer for identical questions ───────────────────────────
    cached_result = get_cached_qa(req.patient_id, req.question)
    if cached_result is not None:
        return cached_result

    # Classify once — reused by both retrieve() and answer_question()
    qtype = classify_question(req.question)

    index, chunk_meta = get_patient_index(req.patient_id)
    retrieved = retrieve(req.question, index, chunk_meta, question_type=qtype)

    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant notes found for this question.")

    # ── Indirect injection scan — remove any chunks containing injection patterns
    retrieved = check_indirect_injection(retrieved)
    if not retrieved:
        raise HTTPException(status_code=400, detail="Query not allowed.")

    result = answer_question(req.question, retrieved, question_type=qtype)

    is_safe, reason = validate_output(result["answer"])
    if not is_safe:
        background_tasks.add_task(
            log_event,
            db,
            event_type="blocked_output",
            patient_id=req.patient_id,
            payload={"reason": reason},
        )
        raise HTTPException(
            status_code=400,
            detail="Response blocked due to safety policy."
        )

    result["answer"] = mask_phi(result["answer"])

    # ── Store in memory cache for future identical questions ───────────────────
    set_cached_qa(req.patient_id, req.question, result)

    # ── Audit log — question is PHI-masked before storage ─────────────────────
    masked_question = mask_phi(req.question)
    background_tasks.add_task(
        log_event,
        db,
        event_type="qa_generated",
        patient_id=req.patient_id,
        payload={
            "question": masked_question,
            "question_type": result["question_type"],
            "citation_count": len(result["citations"]),
        },
    )

    return result