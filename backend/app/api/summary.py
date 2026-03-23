from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.postgres import CachedSummary, get_db
from app.schemas import SummaryRequest, SummaryResponse, SummarySource
from app.services.audit_logger import log_event
from app.services.phi_masking import mask_phi
from app.services.runtime_store import (
    get_cached_sentences,
    get_patient_notes,
    set_cached_sentences,
)
from app.services.sentence_extractor import extract_relevant_sentences
from app.services.summarizer import generate_structured_summary
from app.services.verification import verify_summary

router = APIRouter()


@router.post("/summary", response_model=SummaryResponse)
def summarize_patient(req: SummaryRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # ── Serve from cache if already generated ─────────────────────────────────
    cached = db.query(CachedSummary).filter(
        CachedSummary.subject_id == req.patient_id
    ).order_by(CachedSummary.created_at.desc()).first()

    if cached:
        return SummaryResponse(
            patient_id=req.patient_id,
            summary={
                "Chief Complaint": cached.chief_complaint or "Not documented",
                "Active Diagnoses": cached.active_diagnoses or "Not documented",
                "Current Medications": cached.current_medications or "Not documented",
                "Recent History and Care Plan": cached.recent_history or "Not documented",
            },
            citations=[SummarySource(**c) for c in (cached.citations or [])],
            warnings=cached.warnings or [],
        )

    # ── Load from in-memory cache (populated at startup) ──────────────────────
    notes = get_patient_notes(req.patient_id)
    if not notes:
        raise HTTPException(status_code=404, detail="Patient not found. Run preprocessing first.")

    # ── Extract relevant sentences — use memory cache to skip BioClinicalBERT ──
    extracted = get_cached_sentences(req.patient_id)
    if extracted is None:
        extracted = extract_relevant_sentences(notes)
        if not extracted:
            raise HTTPException(status_code=422, detail="No sentences could be extracted from patient notes.")
        set_cached_sentences(req.patient_id, extracted)

    # ── Generate summary (4 sections in parallel, each with filtered context) ──
    summary, raw_citations = generate_structured_summary(extracted)
    summary = {k: mask_phi(v) for k, v in summary.items()}

    sentences_only = [item["sentence"] for item in extracted]
    warnings = verify_summary(summary, sentences_only)

    # ── Persist to Postgres (cache for future requests) ───────────────────────
    db.add(CachedSummary(
        subject_id=req.patient_id,
        chief_complaint=summary.get("Chief Complaint"),
        active_diagnoses=summary.get("Active Diagnoses"),
        current_medications=summary.get("Current Medications"),
        recent_history=summary.get("Recent History and Care Plan"),
        citations=raw_citations,
        warnings=warnings,
    ))
    db.commit()

    # ── Audit log runs after response is sent — not in request path ───────────
    background_tasks.add_task(
        log_event,
        db,
        event_type="summary_generated",
        patient_id=req.patient_id,
        payload={"warning_count": len(warnings), "citation_count": len(raw_citations)},
    )

    return SummaryResponse(
        patient_id=req.patient_id,
        summary=summary,
        citations=[SummarySource(**c) for c in raw_citations],
        warnings=warnings,
    )
