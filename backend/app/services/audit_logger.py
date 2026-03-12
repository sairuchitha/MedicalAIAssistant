from sqlalchemy.orm import Session
from app.db.postgres import AuditLog


def log_event(db: Session, event_type: str, payload: dict, patient_id: int | None = None) -> None:
    row = AuditLog(patient_id=patient_id, event_type=event_type, payload=payload)
    db.add(row)
    db.commit()
