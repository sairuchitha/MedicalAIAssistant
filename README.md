
# MedMind AI – Privacy-Preserving Clinical AI Assistant

MedMind AI is a privacy-preserving clinical AI system that generates structured patient summaries and supports grounded question answering from electronic health records (EHR).

The system uses **MIMIC-III clinical notes**, performs **PHI masking via Microsoft Presidio**, extracts clinically relevant sentences using **BioClinicalBERT**, indexes patient notes with a **MedCPT 3-encoder RAG pipeline backed by FAISS**, and generates structured summaries and answers using **llama3.1:8b via Ollama** — all fully on-premise with no external API calls.

Scales to **100+ patients** with config-driven auto-selection, parallel PHI masking, adaptive retrieval, and parallel cache warming.

---

# System Architecture

```
MIMIC-III Clinical Notes (NOTEEVENTS.csv)
↓
Auto-select top-N patients (configurable via MAX_PATIENTS in .env)
↓
PHI Masking — Microsoft Presidio (pre-model, parallel per note)
↓
Preprocessing — Deduplication, chunking (240 tokens / 100 stride), FAISS indexing → stored in PostgreSQL
↓
FastAPI Server startup → loads all patients in parallel (8 threads) from PostgreSQL into RAM
↓
┌──────────────────────────────────────────────────────────────┐
│                       Two Pipelines                          │
│                                                              │
│  Summarization                    RAG / QA                   │
│  ─────────────                    ────────                   │
│  BioClinicalBERT sentence         Classify question          │
│  extraction (mean-pool,           (lookup vs reasoning)      │
│  60% soft threshold)              ↓                          │
│  Cached per patient in memory     lookup: FAISS top-4        │
│  ↓                                  → no reranking           │
│  Section-filtered context         reasoning: FAISS top-10    │
│  (40 sentences per section)         → MedCPT Cross-Encoder   │
│  ↓                                ↓                          │
│  llama3.1:8b (4 sections          llama3.1:8b QA             │
│  parallel via Ollama,             ↓                          │
│  cached in Postgres)              QA result cached in memory │
│  ↓                                                           │
│  Faithfulness verification                                   │
│  (entity-level regex check)                                  │
└──────────────────────────────────────────────────────────────┘
↓
PHI Output Filtering — Microsoft Presidio (post-model)
↓
Audit log (background task, non-blocking)
↓
MedMind AI UI (React) + PostgreSQL (notes, indexes, summaries, audit log)
```

---

# Features

### 1. Config-Driven Patient Selection
Patient count is controlled entirely by `MAX_PATIENTS` in `.env` (default: 100). No patient IDs are hardcoded.

Auto-selection criteria from MIMIC-III:
- Must have a Discharge summary (enables ROUGE-L evaluation)
- Must have notes in ≥2 distinct categories (clinical diversity)
- Must have ≥20 qualifying notes (temporal depth for RAG)
- Ranked by note count descending; top-N selected

Override at runtime: `MAX_PATIENTS=20 python -m scripts.preprocess`
Process specific patients: `python -m scripts.preprocess --patient-ids 95324 64925`

### 2. Two-Layer PHI Protection
- **Pre-model**: Presidio masks names, MRNs, dates, SSNs, phone numbers in parallel before any model sees text
- **Post-model**: Presidio scans every model output before display

### 3. Clinical Sentence Extraction (BioClinicalBERT)
- Mean-pool embeddings across all token positions (not CLS)
- Two-pass selection: always-keep pass (allergies, meds, diagnoses) + soft-threshold pass (≥60% of max cosine similarity)
- Results cached per patient in memory — BioClinicalBERT only runs once per patient per server session
- Input: list of note dicts with source provenance (note ID, date, category)
- Output: sentences with metadata for citation generation

### 4. Structured Patient Summary (llama3.1:8b via Ollama)
4-section structured summary:
- Chief Complaint
- Active Diagnoses
- Current Medications
- Recent History and Care Plan

Each section is generated in parallel via `ThreadPoolExecutor` with **section-filtered context** — each worker only receives sentences relevant to its section (keyword-matched, max 40 sentences), keeping prompts lean and focused. Results cached in PostgreSQL — subsequent requests served in <100ms.

Includes entity-level faithfulness verification (medications, dosages, lab values, dates).

### 5. Adaptive RAG-Based Question Answering (MedCPT + llama3.1:8b)
Question type is classified once and flows through the entire pipeline:

- **Chunking**: 240-token windows with 100-token stride (fills MedCPT's 256-token limit, half the overlap of the previous design = fewer chunks per patient)
- **Indexing**: MedCPT Article Encoder → per-patient FAISS IndexFlatIP (L2-normalized), stored in PostgreSQL
- **Adaptive retrieval**:
  - `lookup` (factual): FAISS top-4, no cross-encoder reranking → saves ~1–2s per call
  - `reasoning` (trend/multi-hop): FAISS top-10 → MedCPT Cross-Encoder reranking
- **Generation**: llama3.1:8b via Ollama with routing-specific prompts
- **QA result cache**: identical questions return instantly from memory (SHA-256 keyed)

### 6. Parallel Preprocessing & Cache Warming
- PHI masking parallelized within each patient (4 workers per patient)
- `warm_cache.py` processes multiple patients concurrently (2 parallel, each with its own DB session)
- Server startup loads all FAISS indexes in parallel (8 threads) — startup time scales sub-linearly with patient count

### 7. Security Layer
- Prompt injection detection (pattern matching on physician inputs)
- PHI masking at input and output
- Audit logging runs as a background task (non-blocking, after response is sent)
- PostgreSQL audit log for every query and summary event (HIPAA-ready)

---

# Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite |
| Backend | FastAPI, Python 3.13, Pydantic v2 |
| LLM Serving | Ollama (`llama3.1:8b`) |
| Sentence Extraction | BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) |
| RAG Retrieval | MedCPT Article/Query/Cross-Encoder (`ncbi/MedCPT-*`) |
| Vector Search | FAISS `IndexFlatIP`, L2-normalized, per-patient |
| PHI Masking | Microsoft Presidio |
| Database | PostgreSQL 16 + SQLAlchemy ORM |
| Acceleration | Apple Silicon MPS (PyTorch Metal) |
| Dataset | MIMIC-III NOTEEVENTS (up to 100 patients, 100 notes each) |

---

# Project Structure

```
vision_ai_full_project/
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, lifespan startup, initialize_runtime()
│   │   ├── config.py                # Settings (loaded from .env)
│   │   ├── schemas.py               # Pydantic request/response models
│   │   ├── api/
│   │   │   ├── patients.py          # GET /api/patients
│   │   │   ├── summary.py           # POST /api/summary (cache-first, background audit)
│   │   │   └── qa.py                # POST /api/qa (adaptive retrieval, QA cache)
│   │   ├── db/
│   │   │   └── postgres.py          # SQLAlchemy ORM: Patient, Note, FaissIndex, CachedSummary, AuditLog
│   │   └── services/
│   │       ├── runtime_store.py     # In-memory cache (parallel load from Postgres at startup)
│   │       ├── phi_masking.py       # Presidio PHI masking (pre + post model)
│   │       ├── sentence_extractor.py # BioClinicalBERT sentence extraction with provenance
│   │       ├── medcpt_indexer.py    # MedCPT Article Encoder + FAISS index builder (batch=32)
│   │       ├── retriever.py         # Adaptive retrieval: lookup skips cross-encoder
│   │       ├── summarizer.py        # 4-section parallel summarization, section-filtered context
│   │       ├── qa_service.py        # Question routing + LLM answer generation
│   │       ├── llm_client.py        # Ollama LLM client (generate_with_llm)
│   │       ├── chunker.py           # Token-window chunking (240 tokens, 100 stride)
│   │       ├── dedup.py             # Note deduplication (MD5)
│   │       ├── verification.py      # Entity-level faithfulness check
│   │       └── audit_logger.py      # PostgreSQL audit logging
│   ├── scripts/
│   │   ├── preprocess.py            # Config-driven: auto-select patients, parallel PHI mask, FAISS, warm cache
│   │   ├── warm_cache.py            # Parallel summary cache generation (2 patients concurrently)
│   │   └── evaluate.py              # ROUGE-L, BERTScore, Recall@3 evaluation pipeline
│   ├── .env                         # Environment config
│   └── requirements.txt
│
├── frontend/
│   ├── index.html                   # MedMind AI title, Google Fonts
│   ├── src/
│   │   ├── App.jsx                  # Production landing page: hero, features, how-it-works, live assistant
│   │   ├── api.js                   # API client (AbortController for summary cancellation)
│   │   ├── styles/app.css           # Full design system (navbar, hero, cards, panels, skeleton loaders)
│   │   └── components/
│   │       ├── SummaryPanel.jsx
│   │       ├── QAChat.jsx
│   │       └── CitationList.jsx
│
└── README.md
```

---

# Setup Instructions

## Prerequisites
- Python 3.12+ (with virtualenv)
- Node.js 18+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — for PostgreSQL only
- [Ollama](https://ollama.com) installed and running on your Mac
- MIMIC-III `NOTEEVENTS.csv` placed at `backend/data/raw/NOTEEVENTS.csv`

> **Note:** PostgreSQL runs in Docker. You do **not** need to install or create a database manually — Docker creates it automatically.

---

## Step 1 — Pull the LLM model
```bash
ollama pull llama3.1:8b
```

## Step 2 — Start PostgreSQL via Docker
```bash
docker-compose up -d postgres
```
This starts PostgreSQL on port 5432 and automatically creates the `vision_ai` database. No `createdb` needed.

Verify it's ready:
```bash
docker-compose ps
# postgres should show: healthy
```

## Step 3 — Backend setup
```bash
cd backend

# Create and activate virtualenv (use python3.12, NOT conda)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Important:** If your prompt shows `(.venv) (base)`, conda is overriding the venv.
> Run `conda deactivate` first, then `source .venv/bin/activate`.

## Step 4 — Configure (optional)
Edit `backend/.env` to set how many patients and notes to process:
```
MAX_PATIENTS=100
MAX_NOTES_PER_PATIENT=100
TOKENIZERS_PARALLELISM=false
```

## Step 5 — Preprocess MIMIC data (run once)
Reads NOTEEVENTS.csv, auto-selects patients, masks PHI in parallel, builds FAISS indexes, and pre-generates all summaries into PostgreSQL:
```bash
cd backend
source .venv/bin/activate

python -m scripts.preprocess
```

Other options:
```bash
# Process specific patient IDs only
python -m scripts.preprocess --patient-ids 95324 64925 62561

# Skip summary pre-generation (index only)
python -m scripts.preprocess --skip-cache

# Regenerate summaries for already-indexed patients
python -m scripts.warm_cache

# Force-regenerate even if summaries already exist
python -m scripts.warm_cache --force
```

> This is a one-time step. Re-runs are safe (idempotent — existing data is cleared and rebuilt).

## Step 6 — Start the backend
```bash
cd backend
source .venv/bin/activate

uvicorn app.main:app --reload --port 8000
```

Backend runs at `http://127.0.0.1:8000`

## Step 7 — Start the frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

---

## Full startup order (every session after first setup)

```bash
# 1. Start Ollama (if not already running)
ollama serve

# 2. Start PostgreSQL
docker-compose up -d postgres

# 3. Start backend (in backend/ with .venv active)
conda deactivate          # if needed
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# 4. Start frontend (separate terminal)
cd frontend && npm run dev
```

---

# Docker Deployment (Full Stack)

Use this if you want to run the **entire project in Docker** — e.g., to share with a teammate or deploy on another machine.

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [Ollama](https://ollama.com) installed and running **on the host machine** (not in Docker)
- MIMIC-III `NOTEEVENTS.csv` available

> **Why Ollama runs on the host:** Ollama requires direct GPU/Metal access which Docker cannot provide on Apple Silicon. The backend container reaches it via `host.docker.internal:11434` (Mac/Windows) automatically.

## Step 1 — Place MIMIC data
```bash
# Create the data directory and place your CSV
mkdir -p backend/data/raw
cp /path/to/your/NOTEEVENTS.csv backend/data/raw/NOTEEVENTS.csv
```

## Step 2 — Pull the LLM model (on the host)
```bash
ollama pull llama3.1:8b
```

## Step 3 — Configure `.env`
```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` — key settings to check:
```
MAX_PATIENTS=100
MAX_NOTES_PER_PATIENT=100
TOKENIZERS_PARALLELISM=false
OLLAMA_MODEL=llama3.1:8b
```

Leave `POSTGRES_HOST=localhost` as-is — docker-compose overrides it to `postgres` automatically for the backend container.

## Step 4 — Build and start all services
```bash
docker-compose up --build -d
```

This starts three containers:
| Container | Port | Role |
|---|---|---|
| `medmind-postgres` | 5432 | PostgreSQL — DB auto-created |
| `medmind-backend` | 8000 | FastAPI backend |
| `medmind-frontend` | 5173 | React frontend |

Check all are running:
```bash
docker-compose ps
```

## Step 5 — Run preprocessing (first time only)
The backend container is running but the database is empty. Run preprocessing inside the container:

```bash
docker-compose exec backend python -m scripts.preprocess
```

This will:
- Auto-select patients from NOTEEVENTS.csv
- Mask PHI, build FAISS indexes, store in PostgreSQL
- Pre-generate and cache summaries

For specific patients only:
```bash
docker-compose exec backend python -m scripts.preprocess --patient-ids 95324 64925 62561
```

To regenerate summaries only (indexes already built):
```bash
docker-compose exec backend python -m scripts.warm_cache
```

## Step 6 — Open the app
```
http://localhost:5173
```

## Stopping and restarting

```bash
# Stop all containers (data is preserved in the Docker volume)
docker-compose down

# Restart (no rebuild needed — data still in PostgreSQL)
docker-compose up -d

# Stop and delete all data (full reset)
docker-compose down -v
```

## Checking logs
```bash
docker-compose logs -f backend    # backend logs
docker-compose logs -f postgres   # DB logs
docker-compose logs -f frontend   # frontend logs
```

## Linux notes
On Linux, `host.docker.internal` does not resolve by default. Add this to your `docker-compose.yml` under the `backend` service:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```
Or use your host machine's IP address directly in `.env`:
```
OLLAMA_HOST=http://192.168.1.x:11434
```

---

## Environment Variables (`backend/.env`)
```
MAX_PATIENTS=100
MAX_NOTES_PER_PATIENT=100
MIN_NOTE_WORDS=100
TOKENIZERS_PARALLELISM=false

OLLAMA_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434

BIOCLINICALBERT_MODEL=emilyalsentzer/Bio_ClinicalBERT
MEDCPT_ARTICLE_ENCODER=ncbi/MedCPT-Article-Encoder
MEDCPT_QUERY_ENCODER=ncbi/MedCPT-Query-Encoder
MEDCPT_CROSS_ENCODER=ncbi/MedCPT-Cross-Encoder

CHUNK_TOKENS=240
CHUNK_STRIDE=100
LOOKUP_TOP_K_RETRIEVE=4
REASONING_TOP_K_RETRIEVE=10
LOOKUP_TOP_K=3
REASONING_TOP_K=5

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vision_ai
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

MIMIC_NOTEEVENTS_CSV=data/raw/NOTEEVENTS.csv
```

---

# Example Workflow

1. Select a patient from the dropdown (loads in <100ms from memory)
2. Click **Generate Summary** — served from PostgreSQL cache in <100ms, or generated fresh via BioClinicalBERT + llama3.1:8b if not cached
3. Ask clinical questions — question type is classified once, adaptive retrieval runs (lookup skips reranking), llama3.1:8b generates a grounded answer with citations

---

# Implementation Status

## Completed
- MIMIC-III preprocessing pipeline with parallel PHI masking (Microsoft Presidio)
- Config-driven auto-patient selection (no hardcoded IDs, scales to 100+ patients)
- BioClinicalBERT sentence extraction with source provenance; cached per patient in memory
- Section-filtered context per LLM call (keyword-matched, 40 sentences per section)
- MedCPT 3-encoder RAG pipeline: Article Encoder → FAISS → Query Encoder → adaptive Cross-Encoder
- Adaptive retrieval: lookup questions skip cross-encoder reranking (~1–2s saved per call)
- QA result memory cache (SHA-256 keyed, identical questions return instantly)
- Lookup vs Reasoning question routing with per-type chunk selection
- llama3.1:8b 4-section structured summarization via Ollama (4 sections in parallel)
- Summary and QA citations (note name, date, type)
- Entity-level faithfulness verification (regex-based, flags unverified medications/labs/dates)
- PostgreSQL full data store: notes, FAISS indexes, cached summaries, audit log
- Parallel runtime initialization: 8 threads load FAISS indexes concurrently at startup
- Parallel warm cache: 2 patients processed concurrently, each with own DB session
- Background audit logging (non-blocking, fires after response is sent)
- Apple Silicon MPS acceleration for all PyTorch models
- Production-quality React UI: hero landing page, features section, how-it-works, live assistant
- One-time preprocessing script with CLI flags (`--patient-ids`, `--skip-cache`, `--csv`)
- Cache warm-up script with CLI flags (`--force`, `--patient-ids`)
- Evaluation pipeline script (`scripts/evaluate.py`) — ROUGE-L, BERTScore, Recall@3

## Not Yet Implemented
- Citation display in UI (inline citations + clickable footnotes)
- GPT-2 perplexity-based prompt injection detection (currently pattern matching only)
- BERTScore faithfulness verification (currently regex-based entity check)
- Docker Compose deployment

---

# Limitations

- LLM response quality depends on extracted sentence quality and Ollama model performance
- Prompt injection detection uses simple pattern matching, not perplexity-based detection
- Preprocessing 100 patients is a ~1.5–2 hour one-time operation

---

# License

This project uses MIMIC-III clinical data for research purposes only.
Use of the dataset requires appropriate data access approval from PhysioNet.

---

# Authors

Sai Ruchitha
Georgia State University
