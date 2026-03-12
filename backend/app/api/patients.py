from fastapi import APIRouter

from app.services.runtime_store import get_all_patient_ids

router = APIRouter()


@router.get("/patients")
def patients():
    return {"patients": get_all_patient_ids()}
