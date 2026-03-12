from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
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


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
