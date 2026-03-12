from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SummaryRequest(BaseModel):
    patient_id: int = Field(..., description="Patient SUBJECT_ID")


class SummaryResponse(BaseModel):
    patient_id: int
    summary: Dict[str, str]
    warnings: List[str] = []


class QARequest(BaseModel):
    patient_id: int
    question: str


class Citation(BaseModel):
    id: int
    date: str
    note_type: str
    section_name: str
    note_id: str


class QAResponse(BaseModel):
    question_type: str
    answer: str
    citations: List[Citation]
    warnings: List[str] = []


class PatientsResponse(BaseModel):
    patients: List[int]


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class AuditLogCreate(BaseModel):
    patient_id: Optional[int] = None
    event_type: str
    payload: Dict = {}
