from sqlalchemy import (
    LargeBinary, create_engine, Column, Integer, String, DateTime, JSON, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

from app.config import settings


engine = create_engine(settings.POSTGRES_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=True)
    event_type = Column(String(100), nullable=False)
    payload = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Patient(Base):
    __tablename__ = "patients"

    subject_id = Column(Integer, primary_key=True)
    note_count = Column(Integer, nullable=False)
    categories = Column(JSON, nullable=False, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, nullable=False, index=True)
    row_id = Column(String(50), nullable=True)
    chart_date = Column(String(50), nullable=True)
    category = Column(String(100), nullable=True)
    description = Column(String(255), nullable=True)
    masked_text = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FaissIndex(Base):
    __tablename__ = "faiss_indexes"

    subject_id = Column(Integer, primary_key=True)
    index_data = Column(LargeBinary, nullable=False)
    chunk_meta = Column(JSON, nullable=False, default=[])
    updated_at = Column(DateTime(timezone=True), server_default=func.now())


class CachedSummary(Base):
    __tablename__ = "cached_summaries"

    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, nullable=False, index=True)
    chief_complaint = Column(Text, nullable=True)
    active_diagnoses = Column(Text, nullable=True)
    current_medications = Column(Text, nullable=True)
    recent_history = Column(Text, nullable=True)
    citations = Column(JSON, nullable=False, default=[])
    warnings = Column(JSON, nullable=False, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
