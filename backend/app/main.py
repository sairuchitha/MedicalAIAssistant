from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.qa import router as qa_router
from app.api.patients import router as patients_router
#from app.api.qa import router as qa_router
from app.api.summary import router as summary_router
from app.config import settings
from app.db.postgres import init_db


def ensure_directories() -> None:
    for p in [
        settings.PROCESSED_DIR,
        settings.PATIENT_NOTES_DIR,
        settings.SUMMARIES_DIR,
        settings.INDEX_DIR,
        settings.CHROMA_DIR,
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    init_db()
    from app.services.runtime_store import initialize_runtime
    initialize_runtime()
    print("Startup complete")
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.get("/")
def root():
    return {
        "message": "Vision AI backend is running",
        "version": settings.APP_VERSION,
        "env": settings.APP_ENV,
    }


app.include_router(patients_router, prefix="/api", tags=["patients"])
app.include_router(summary_router, prefix="/api", tags=["summary"])
app.include_router(qa_router, prefix="/api", tags=["qa"])