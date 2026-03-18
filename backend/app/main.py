from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api.patients import router as patients_router
from app.api.qa import router as qa_router
from app.api.summary import router as summary_router
from app.config import settings
from app.db.postgres import init_db

# ── Rate limiter setup ─────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Input sanitization constants ───────────────────────────────────────────────
MAX_QUESTION_LENGTH = 500
MAX_BODY_SIZE_BYTES = 10_000


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

# ── Attach rate limiter to app ─────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Security headers middleware ────────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Cache-Control"] = "no-store"
    return response


# ── Input sanitization middleware ──────────────────────────────────────────────
@app.middleware("http")
async def sanitize_inputs(request: Request, call_next):
    if request.method == "POST":
        # Block oversized request bodies
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_SIZE_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"}
            )

        try:
            body = await request.json()

            # Check question field if present
            question = body.get("question", "")

            # Block questions that are too long
            if len(question) > MAX_QUESTION_LENGTH:
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Question too long. Maximum {MAX_QUESTION_LENGTH} characters allowed."}
                )

            # Block null bytes and dangerous control characters
            if any(ord(c) < 32 and c not in '\n\t\r' for c in question):
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid characters detected in question."}
                )

            # Block empty questions
            if question.strip() == "" and "question" in body:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Question cannot be empty."}
                )

        except Exception:
            # If body isn't JSON or has no question field, let it pass through
            pass

    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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






'''
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.patients import router as patients_router
from app.api.qa import router as qa_router
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
'''
