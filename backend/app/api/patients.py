from fastapi import APIRouter

from app.services.mimic_loader import load_mvp_patient_notes, get_patient_ids

router = APIRouter()


@router.get("/patients")
def patients():
    df = load_mvp_patient_notes(max_patients=1, max_notes_per_patient=10)
    return {"patients": get_patient_ids(df)}