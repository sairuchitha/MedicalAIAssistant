
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
┌──────────────────────────────────────────────────────────────────┐
│                         Two Pipelines                            │
│                                                                  │
│  Summarization                      RAG / QA                     │
│  ─────────────                      ────────                     │
│  BioClinicalBERT sentence           3-layer injection check      │
│  extraction (mean-pool,             (length → regex → perplexity)│
│  60% soft threshold)                ↓                            │
│  Cached per patient in memory       Classify question            │
│  ↓                                  (lookup vs reasoning)        │
│  Section-filtered context           ↓                            │
│  (40 sentences per section)         lookup: FAISS top-4          │
│  ↓                                    → no reranking             │
│  llama3.1:8b (4 sections            reasoning: FAISS top-10      │
│  parallel via Ollama,                 → MedCPT Cross-Encoder     │
│  cached in Postgres)                ↓                            │
│  ↓                                  Indirect injection scan      │
│  BERTScore faithfulness             (retrieved chunks filtered)   │
│  verification (sentence-level)      ↓                            │
│                                     llama3.1:8b QA               │
│                                     ↓                            │
│                                     QA result cached in memory   │
└──────────────────────────────────────────────────────────────────┘
↓
PHI Output Filtering — Microsoft Presidio (post-model)
↓
Audit log (background task, non-blocking, PHI-masked payload)
↓
MedMind AI UI (React) + PostgreSQL (notes, indexes, summaries, audit log)
```

---

# Evaluation Results (Measured)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| ROUGE-L | ≥ 0.35 | 0.108 | Structural — abstractive output vs extractive ground truth |
| BERTScore-F1 | ≥ 0.78 | **0.795** | PASS |
| Recall@3 — Lookup | ≥ 0.80 | **0.857** | PASS |
| Recall@3 — Reasoning | ≥ 0.80 | **1.000** | PASS |
| Recall@3 — Overall | ≥ 0.80 | **0.929** | PASS |
| MRR — Overall | ≥ 0.70 | **0.893** | PASS |
| Reranking Gain | > 0 | **+0.14** | PASS |
| Faithfulness F1 | ≥ 0.75 | **0.768** | PASS — borderline |
| Injection Detection Rate | ≥ 0.90 | **0.895** | PASS |
| Injection Precision | ≥ 0.90 | **1.000** | PASS |
| Injection F1 | ≥ 0.90 | **0.950** | PASS |
| False Positive Rate | ≤ 0.05 | **0.000** | PASS |
| PHI Residual Rate | ≤ 0.02 | **0.0001** | PASS |

> ROUGE-L is structurally low: abstractive 4-section summaries use different vocabulary than discharge note ground truth. BERTScore-F1 (0.795) is the meaningful signal.

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
- **Audit log**: PHI-masked question stored (raw question never persisted)
- Measured residual rate: **0.0001** (1 false-positive entity in 105 patients)

### 3. Multi-Layer Prompt Injection Defense
Five-layer injection guard on every QA request:

| Layer | Mechanism | Queries Caught |
|-------|-----------|---------------|
| 1 | Input length > 500 chars | Overflow attacks |
| 2 | Regex pattern guard (38+ patterns) | Verbatim injections, social engineering, instruction-override, paraphrase bypasses |
| 3 | Suspicious-intent detection (16 phrases) | Broad-access probing ("tell me everything", "full record", "bypass safety") |
| 4 | GPT-2 perplexity (PPL < 90) | Low-entropy formulaic inputs |
| 5 | Output validation | LLM responses containing instruction-leakage markers blocked before returning to client |

- Injection detection rate: **89.5%** (17/19 test queries), Precision: **100%**, F1: **0.944**, False Positive Rate: **0.0%**
- Hardened QA prompt templates: every prompt includes a STRICT RULES block — the LLM is instructed not to follow instructions embedded in user queries or retrieved notes
- Indirect injection guard: retrieved note chunks scanned for embedded injection patterns (including XML/markdown injection markers) before being passed to the LLM
- Output validation audit event (`blocked_output`) logged for any response that triggers output-side checks
- Generic error response on block — no pattern information leaked to attacker
- PHI-masked question stored in audit log (not raw blocked query)

### 4. Clinical Sentence Extraction (BioClinicalBERT)
- Mean-pool embeddings across all token positions (not CLS)
- Two-pass selection: always-keep pass (allergies, meds, diagnoses) + soft-threshold pass (≥60% of max cosine similarity)
- Results cached per patient in memory — BioClinicalBERT only runs once per patient per server session
- Input: list of note dicts with source provenance (note ID, date, category)
- Output: sentences with metadata for citation generation

### 5. Structured Patient Summary (llama3.1:8b via Ollama)
4-section structured summary:
- Chief Complaint
- Active Diagnoses
- Current Medications
- Recent History and Care Plan

Each section is generated in parallel via `ThreadPoolExecutor` with **section-filtered context** — each worker only receives sentences relevant to its section (keyword-matched, max 40 sentences), keeping prompts lean and focused. Results cached in PostgreSQL — subsequent requests served in <100ms.

Includes BERTScore faithfulness verification (sentence-level F1 ≥ 0.75 threshold).

### 6. Adaptive RAG-Based Question Answering (MedCPT + llama3.1:8b)
Question type is classified once and flows through the entire pipeline:

- **Chunking**: 240-token windows with 100-token stride (fills MedCPT's 256-token limit, half the overlap of previous design = fewer chunks per patient)
- **Indexing**: MedCPT Article Encoder → per-patient FAISS IndexFlatIP (L2-normalized), stored in PostgreSQL
- **Adaptive retrieval**:
  - `lookup` (factual): FAISS top-4, no cross-encoder reranking → saves ~1–2s per call
  - `reasoning` (trend/multi-hop): FAISS top-10 → MedCPT Cross-Encoder reranking
- **Generation**: llama3.1:8b via Ollama (temperature=0, 120s timeout) with routing-specific prompts
- **QA result cache**: identical questions return instantly from memory (SHA-256 keyed)

Reasoning keywords checked before lookup keywords to prevent misclassification (e.g., "What is the trend of X" correctly routes as reasoning, not lookup).

### 7. Parallel Preprocessing & Cache Warming
- PHI masking parallelized within each patient (4 workers per patient)
- `warm_cache.py` processes multiple patients concurrently (2 parallel, each with its own DB session)
- Server startup loads all FAISS indexes in parallel (8 threads) — startup time scales sub-linearly with patient count

### 8. Security Evaluation & Attack Demo
Three dedicated evaluation scripts:
- `evaluate.py` — ROUGE-L, BERTScore, Recall@3 (split by question type), MRR, reranking gain, faithfulness
- `evaluate_security.py` — full injection detection metrics (precision, recall, F1, FPR), PHI residual rate
- `demo_attack_defense.py` — three-state attack/defense demo (no defense → regex-only → full defense)

---

# Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite |
| Backend | FastAPI, Python 3.12+, Pydantic v2 |
| LLM Serving | Ollama (`llama3.1:8b`, temperature=0, 300s timeout) |
| Sentence Extraction | BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) |
| RAG Retrieval | MedCPT Article/Query/Cross-Encoder (`ncbi/MedCPT-*`) |
| Vector Search | FAISS `IndexFlatIP`, L2-normalized, per-patient |
| PHI Masking | Microsoft Presidio |
| Database | PostgreSQL 16 + SQLAlchemy ORM |
| Acceleration | Apple Silicon MPS (PyTorch Metal) — MedCPT models only |
| Dataset | MIMIC-III NOTEEVENTS (up to 100 patients, 300 notes each) |

---

# Project Structure

```
vision_ai_full_project/
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, lifespan startup, CORS (localhost only)
│   │   ├── config.py                # Settings (loaded from .env)
│   │   ├── schemas.py               # Pydantic models (question: 3–500 chars validated)
│   │   ├── api/
│   │   │   ├── patients.py          # GET /api/patients
│   │   │   ├── summary.py           # POST /api/summary (cache-first, background audit)
│   │   │   └── qa.py                # POST /api/qa (injection check, adaptive retrieval, QA cache)
│   │   ├── db/
│   │   │   └── postgres.py          # SQLAlchemy ORM: Patient, Note, FaissIndex, CachedSummary, AuditLog
│   │   └── services/
│   │       ├── runtime_store.py     # In-memory cache (parallel load from Postgres at startup)
│   │       ├── phi_masking.py       # Presidio PHI masking (pre + post model)
│   │       ├── security.py          # 3-layer injection guard: length → regex → perplexity
│   │       ├── sentence_extractor.py # BioClinicalBERT sentence extraction with provenance
│   │       ├── medcpt_indexer.py    # MedCPT Article Encoder + FAISS index builder (batch=32)
│   │       ├── retriever.py         # Adaptive retrieval: lookup skips cross-encoder
│   │       ├── summarizer.py        # 4-section parallel summarization, section-filtered context
│   │       ├── qa_service.py        # Question routing + LLM answer generation
│   │       ├── llm_client.py        # Ollama client (temp=0, 120s timeout)
│   │       ├── chunker.py           # Token-window chunking (240 tokens, 100 stride)
│   │       ├── dedup.py             # Note deduplication (MD5)
│   │       ├── verification.py      # BERTScore faithfulness check (sentence-level F1 ≥ 0.75)
│   │       └── audit_logger.py      # PostgreSQL audit logging (PHI-masked payloads)
│   ├── scripts/
│   │   ├── preprocess.py            # Config-driven: auto-select patients, parallel PHI mask, FAISS, warm cache
│   │   ├── warm_cache.py            # Parallel summary cache generation (2 patients concurrently)
│   │   ├── evaluate.py              # ROUGE-L, BERTScore, Recall@3 (lookup/reasoning split), MRR, faithfulness
│   │   ├── evaluate_security.py     # Injection detection metrics, PHI residual rate
│   │   └── demo_attack_defense.py   # Three-state attack/defense demo (no defense → regex → full)
│   ├── .env                         # Environment config
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── src/
│   │   ├── App.jsx                  # Landing page: hero, features, how-it-works, live assistant
│   │   ├── api.js                   # API client with error status propagation
│   │   ├── styles/app.css           # Full design system
│   │   └── components/
│   │       ├── SummaryPanel.jsx
│   │       ├── QAChat.jsx           # Elapsed timer, reasoning latency hint, injection block banner
│   │       └── CitationList.jsx
│
├── ATTACK_DEFENSE_REPORT.md         # 1-page attack/defense submission brief
├── NEXT_STEPS.md                    # Production readiness audit with prioritized action list
├── docker-compose.yml
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
MAX_NOTES_PER_PATIENT=300
TOKENIZERS_PARALLELISM=false
KMP_DUPLICATE_LIB_OK=TRUE
```

> `KMP_DUPLICATE_LIB_OK=TRUE` suppresses an OpenMP duplicate-library warning that occurs when FAISS and PyTorch are both loaded (harmless, but noisy without this flag).

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

Wait for the startup log:
```
[runtime_store] Ready — 105 patients, 10244 notes loaded in X.Xs
```

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

## Running Evaluations

All scripts run from the `backend/` directory with `.venv` active:

```bash
# RAG + summarization evaluation (BERTScore, Recall@3, MRR, faithfulness)
python -m scripts.evaluate

# Security evaluation (injection detection, PHI residual rate)
python -m scripts.evaluate_security

# Three-state attack/defense demo
python -m scripts.demo_attack_defense
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
mkdir -p backend/data/raw
cp /path/to/your/NOTEEVENTS.csv backend/data/raw/NOTEEVENTS.csv
```

## Step 2 — Pull the LLM model (on the host)
```bash
ollama pull llama3.1:8b
```

## Step 3 — Build and start all services
```bash
docker-compose up --build -d
```

This starts three containers:
| Container | Port | Role |
|---|---|---|
| `medmind-postgres` | 5432 | PostgreSQL — DB auto-created |
| `medmind-backend` | 8000 | FastAPI backend |
| `medmind-frontend` | 5173 | React frontend |

## Step 4 — Run preprocessing (first time only)
```bash
docker-compose exec backend python -m scripts.preprocess
```

For specific patients only:
```bash
docker-compose exec backend python -m scripts.preprocess --patient-ids 95324 64925 62561
```

## Step 5 — Open the app
```
http://localhost:5173
```

## Stopping and restarting

```bash
# Stop all containers (data preserved in Docker volume)
docker-compose down

# Restart (no rebuild needed — data still in PostgreSQL)
docker-compose up -d

# Stop and delete all data (full reset)
docker-compose down -v
```

## Checking logs
```bash
docker-compose logs -f backend
docker-compose logs -f postgres
docker-compose logs -f frontend
```

## Linux notes
On Linux, `host.docker.internal` does not resolve by default. Add this to your `docker-compose.yml` under the `backend` service:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

---

## Environment Variables (`backend/.env`)
```
MAX_PATIENTS=100
MAX_NOTES_PER_PATIENT=300
MIN_NOTE_WORDS=100
TOKENIZERS_PARALLELISM=false
KMP_DUPLICATE_LIB_OK=TRUE

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
2. Click **Generate Summary** — served from PostgreSQL cache in <100ms, or generated fresh via BioClinicalBERT + llama3.1:8b (~30–60s first time)
3. Ask clinical questions:
   - **Lookup** ("What medications is the patient on?"): ~3–6s — FAISS top-4, no reranking
   - **Reasoning** ("How did the infection progress over time?"): ~30–90s — FAISS top-10 + MedCPT cross-encoder + llama3.1:8b over 5 date-sorted chunks
4. Injection attempts return an amber warning banner; no clinical data is returned

---

# Security Architecture

## Attack surface
MedMind AI exposes a natural-language QA interface backed by an LLM with access to de-identified clinical notes. The primary attack vector is adversarial natural-language input.

## Defense layers (POST /api/qa)

| Layer | Mechanism | Example blocked input |
|-------|-----------|----------------------|
| Input validation | Pydantic: 3–500 char limit | >500 char overflow payloads |
| Length check | Hard reject >500 chars | Length overflow attacks |
| Regex guard | 38+ patterns: verbatim injections, social engineering, instruction-override, paraphrase bypasses | "ignore all previous instructions", "as a security researcher...", "just say patient is..." |
| Suspicious-intent detection | 16 semantic phrases | "tell me everything", "full record", "bypass safety", "show all records" |
| GPT-2 perplexity (PPL < 90) | Block low-entropy formulaic inputs (CPU, loaded at import time) | Template-filled payload strings |
| Output validation | Scan LLM response for instruction-leakage markers | Responses containing "i was instructed", "hidden instruction" |
| Hardened prompt templates | STRICT RULES block in every QA prompt | LLM refuses to follow instructions embedded in queries or retrieved notes |
| Indirect injection | Scan retrieved chunks for injection patterns (incl. XML/markdown markers) before LLM sees them | Injections embedded in clinical note text |
| PHI masking (output) | Presidio scan on every LLM response | Residual PHI in model output |
| Audit log | PHI-masked events, no raw question stored | Forensic trail without PHI leakage |
| CORS | Restricted to `localhost:5173` only | Cross-origin API abuse |

## Measured results (19-query test set)
- Detection rate: **89.5%** (17/19), Precision: **1.000**, F1: **0.944** — 0% false positive rate
- 2 missed: indirect paraphrase injections; output validation and hardened prompts provide defense-in-depth for these cases
- Regex-only baseline (before social engineering + intent patterns): 52.6% — improvement: **+36.8%**

---

# Implementation Status

## Completed and Measured

- MIMIC-III preprocessing pipeline with parallel PHI masking (Microsoft Presidio)
- Config-driven auto-patient selection (no hardcoded IDs, scales to 100+ patients)
- BioClinicalBERT sentence extraction with source provenance; cached per patient in memory
- Section-filtered context per LLM call (keyword-matched, 40 sentences per section)
- MedCPT 3-encoder RAG pipeline: Article Encoder → FAISS → Query Encoder → adaptive Cross-Encoder
- Adaptive retrieval: lookup questions skip cross-encoder reranking (~1–2s saved per call)
- **Question classifier bug fixed**: reasoning keywords checked before lookup keywords
- QA result memory cache (SHA-256 keyed, identical questions return instantly)
- llama3.1:8b 4-section structured summarization via Ollama (4 sections in parallel, temperature=0)
- 120-second Ollama timeout (prevents hung workers)
- Summary and QA citations (note ID, date, type, section)
- BERTScore faithfulness verification (sentence-level F1, flags sentences below 0.75)
- **Multi-layer injection guard**: input length + 30+ regex patterns (verbatim + social engineering) + GPT-2 perplexity (offline)
- **Indirect injection scanning**: retrieved chunks filtered before LLM sees them
- **CORS restricted** to localhost:5173 only
- **Input validation**: question field validated 3–500 chars at schema level
- **PHI-masked audit log**: raw question never persisted
- **Generic error responses**: injection block reveals no pattern information
- PostgreSQL full data store: notes, FAISS indexes, cached summaries, audit log
- Parallel runtime initialization: 8 threads load FAISS indexes concurrently at startup
- Background audit logging (non-blocking, fires after response is sent)
- Apple Silicon MPS acceleration for MedCPT models
- Production-quality React UI: hero landing page, features section, how-it-works, live assistant
- QA UI: injection block banner, elapsed timer, reasoning latency hint
- Full evaluation pipeline: ROUGE-L, BERTScore, Recall@3 (lookup/reasoning split), MRR, reranking gain, faithfulness, injection metrics, PHI residual rate
- Three-state attack/defense demo script

## Known Gaps (Not Yet Implemented)

| Gap | Impact | Effort |
|----|--------|--------|
| No authentication | Any HTTP client accesses any patient — HIPAA violation | 1–2 days |
| No rate limiting | DoS via Ollama thread pool exhaustion | 2 hours |
| In-memory cache doesn't scale | Two uvicorn workers = inconsistent results | 1–2 days (Redis) |
| Race condition on concurrent summaries | Two simultaneous requests generate duplicate rows | 1 hour |
| No database migrations (Alembic) | Schema changes require manual drop/recreate | half day |
| Weak default DB password | `postgres` in config | 1 hour |
| QA warnings not surfaced in UI | `warnings` array returned but not rendered | 30 min |
| Citation UI is a flat chip list | No way to trace claim to source note | 1 day |
| Summary cache has no TTL | Clinicians see stale data with no timestamp | 1 hour |

See [NEXT_STEPS.md](NEXT_STEPS.md) for the full prioritized action list.

---

# Limitations

- **ROUGE-L structurally low (0.108)**: not a real failure — abstractive 4-section summaries use different vocabulary than extractive discharge note ground truth. BERTScore-F1 (0.795) is the meaningful metric.
- **Reasoning QA latency**: llama3.1:8b with 5-chunk context takes 30–90s on Apple Silicon. The UI shows an elapsed timer and a latency warning for reasoning questions.
- **GPT-2 perplexity disabled in server**: loading GPT-2 alongside MedCPT models on Apple Silicon MPS causes a Metal backend segfault (exit 139). GPT-2 remains available in standalone evaluation scripts. The 2-layer defense (length + regex) achieves 89.5% detection with 0% false positives.
- **Preprocessing 100 patients**: ~1.5–2 hour one-time operation (PHI masking + FAISS indexing via MedCPT Article Encoder).

---

# License

This project uses MIMIC-III clinical data for research purposes only.
Use of the dataset requires appropriate data access approval from PhysioNet.

---

# Authors

Sai Ruchitha
Georgia State University
