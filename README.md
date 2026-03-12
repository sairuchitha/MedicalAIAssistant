
# Vision AI – Privacy-Preserving Clinical AI Assistant

Vision AI is a privacy-preserving clinical AI system that generates structured patient summaries and supports grounded question answering from electronic health records (EHR).

The system uses **MIMIC-III clinical notes**, performs **PHI masking via Microsoft Presidio**, extracts clinically relevant sentences using **BioClinicalBERT**, indexes patient notes with a **MedCPT 3-encoder RAG pipeline backed by FAISS**, and generates structured summaries and answers using **llama3.1:8b via Ollama** — all fully on-premise with no external API calls.

---

# System Architecture

```
MIMIC-III Clinical Notes (NOTEEVENTS.csv)
↓
PHI Masking — Microsoft Presidio (pre-model)
↓
Preprocessing — Deduplication, chunking, FAISS indexing → stored in PostgreSQL
↓
FastAPI Server startup → loads all patient data into memory from PostgreSQL
↓
┌─────────────────────────────────────────────────────┐
│                   Two Pipelines                     │
│                                                     │
│  Summarization                  RAG / QA            │
│  ────────────                   ────────            │
│  BioClinicalBERT sentence       MedCPT Article      │
│  extraction (mean-pool,         Encoder → FAISS     │
│  60% soft threshold)            per-patient index   │
│  ↓                              ↓                   │
│  BioMistral-7B (4-section       MedCPT Query        │
│  prompting via Ollama)          Encoder + FAISS     │
│  ↓                              ↓                   │
│  Faithfulness verification      MedCPT Cross-Encoder│
│  (entity-level regex check)     reranking           │
│                                 ↓                   │
│                                 BioMistral-7B QA    │
└─────────────────────────────────────────────────────┘
↓
PHI Output Filtering — Microsoft Presidio (post-model)
↓
Physician UI (React) + PostgreSQL (notes, indexes, summaries, audit log)
```

---

# Features

### 1. Patient Selection
5 clinically diverse patients pre-selected from MIMIC-III and stored in PostgreSQL. Patient list loads in <100ms at startup from in-memory cache.

**Why these 5 patients:**
- All have Physician progress notes (richest for QA) AND Discharge summaries (ground truth for evaluation)
- 50-60 clinical notes each — enough temporal depth for trend/reasoning questions
- Diverse profiles: ICU/sepsis (95324), neuro-oncology (64925), oncology (62561), trauma/Afib (32639), cardiac (64230)

### 2. Two-Layer PHI Protection
- **Pre-model**: Presidio masks names, MRNs, dates, SSNs, phone numbers before any model sees text
- **Post-model**: Presidio scans every model output before display

### 3. Clinical Sentence Extraction (BioClinicalBERT)
- Mean-pool embeddings across all token positions (not CLS)
- Two-pass selection: always-keep pass (allergies, meds, diagnoses) + soft-threshold pass (≥60% of max cosine similarity)
- Input: list of note dicts with source provenance (note ID, date, category)
- Output: sentences with metadata for citation generation

### 4. Structured Patient Summary (llama3.1:8b via Ollama)
4-section structured summary:
- Chief Complaint
- Active Diagnoses
- Current Medications
- Recent History and Care Plan

All 4 sections generated in parallel via `ThreadPoolExecutor`. Results cached in PostgreSQL — subsequent requests served in <100ms.

Includes entity-level faithfulness verification (medications, dosages, lab values, dates).

### 5. RAG-Based Question Answering (MedCPT + llama3.1:8b)
Full dense-retrieval pipeline:
- **Chunking**: 200-token overlapping windows with 50-token stride, with chunk metadata (note type, date, section)
- **Indexing**: MedCPT Article Encoder → per-patient FAISS IndexFlatIP (L2-normalized), stored in PostgreSQL
- **Retrieval**: MedCPT Query Encoder → FAISS top-10 → MedCPT Cross-Encoder reranking
- **Routing**: keyword heuristic classifies question as Lookup (top-3 chunks) or Reasoning (top-5 date-sorted chunks)
- **Generation**: BioMistral-7B with constrained prompts (lookup: "not documented if absent", reasoning: chronological synthesis)

### 6. Security Layer
- Prompt injection detection (pattern matching on physician inputs)
- PHI masking at input and output
- PostgreSQL audit log for every query and summary event (HIPAA)

---

# Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, Vite |
| Backend | FastAPI, Python 3.13, Pydantic |
| LLM Serving | Ollama (`llama3.1:8b`) |
| Sentence Extraction | BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) |
| RAG Retrieval | MedCPT Article/Query/Cross-Encoder (`ncbi/MedCPT-*`) |
| Vector Search | FAISS `IndexFlatIP`, L2-normalized, per-patient |
| PHI Masking | Microsoft Presidio |
| Database | PostgreSQL + SQLAlchemy ORM |
| Acceleration | Apple Silicon MPS (PyTorch Metal) |
| Dataset | MIMIC-III NOTEEVENTS (5 patients, ~244 notes) |

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
│   │   │   ├── summary.py           # POST /api/summary (cache-first)
│   │   │   └── qa.py                # POST /api/qa
│   │   ├── db/
│   │   │   └── postgres.py          # SQLAlchemy ORM: Patient, Note, FaissIndex, CachedSummary, AuditLog
│   │   └── services/
│   │       ├── runtime_store.py     # In-memory cache (loaded from Postgres at startup)
│   │       ├── phi_masking.py       # Presidio PHI masking (pre + post model)
│   │       ├── sentence_extractor.py # BioClinicalBERT sentence extraction with provenance
│   │       ├── medcpt_indexer.py    # MedCPT Article Encoder + FAISS index builder
│   │       ├── retriever.py         # MedCPT Query Encoder + Cross-Encoder reranker
│   │       ├── summarizer.py        # 4-section parallel summarization + citations
│   │       ├── qa_service.py        # Question routing + LLM answer generation
│   │       ├── llm_client.py        # Ollama LLM client (generate_with_llm)
│   │       ├── chunker.py           # Token-window chunking with metadata
│   │       ├── dedup.py             # Note deduplication
│   │       ├── verification.py      # Entity-level faithfulness check
│   │       └── audit_logger.py      # PostgreSQL audit logging
│   ├── scripts/
│   │   ├── preprocess.py            # One-time: mask PHI, build FAISS, store in Postgres, warm cache
│   │   ├── warm_cache.py            # Re-generate cached summaries without full reprocessing
│   │   └── evaluate.py              # ROUGE-L, BERTScore, Recall@3 evaluation pipeline
│   ├── .env                         # Environment config
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Main UI with AbortController for request cancellation
│   │   ├── api.js                   # API client
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
- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running
- PostgreSQL running locally (default: `vision_ai` database, user `postgres`)
- MIMIC-III `NOTEEVENTS.csv` at `backend/data/raw/NOTEEVENTS.csv`

## 1. Pull the LLM model
```bash
ollama pull llama3.1:8b
```

## 2. Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Create the PostgreSQL database
```bash
createdb vision_ai
```

## 4. Preprocess MIMIC data (run once)
Selects 5 patients, masks PHI, builds FAISS indexes, pre-generates all summaries:
```bash
python -m scripts.preprocess
```

## 5. Start the backend
```bash
uvicorn app.main:app --reload --port 8000
```

## 6. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`, backend at `http://127.0.0.1:8000`

---

## Environment Variables (`backend/.env`)
```
OLLAMA_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434

BIOCLINICALBERT_MODEL=emilyalsentzer/Bio_ClinicalBERT
MEDCPT_ARTICLE_ENCODER=ncbi/MedCPT-Article-Encoder
MEDCPT_QUERY_ENCODER=ncbi/MedCPT-Query-Encoder
MEDCPT_CROSS_ENCODER=ncbi/MedCPT-Cross-Encoder

MIN_NOTE_WORDS=100
CHUNK_TOKENS=200
CHUNK_STRIDE=50
RETRIEVAL_TOP_K=10
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

1. Select a patient from the dropdown (loads in <100ms)
2. Click **Generate Summary** — served from PostgreSQL cache in <100ms, or generated fresh via BioClinicalBERT + llama3.1:8b if not cached
3. Ask clinical questions in the QA box — MedCPT retrieves and reranks relevant chunks, llama3.1:8b generates a grounded answer with citations

---

# Implementation Status

## Completed
- MIMIC-III preprocessing pipeline with PHI masking (Microsoft Presidio)
- BioClinicalBERT sentence extraction with source provenance (note ID, date, category)
- MedCPT 3-encoder RAG pipeline: Article Encoder → FAISS → Query Encoder → Cross-Encoder reranking
- Lookup vs Reasoning question routing with per-type chunk selection
- llama3.1:8b 4-section structured summarization via Ollama (4 sections in parallel)
- Summary and QA citations (note name, date, type)
- Entity-level faithfulness verification (regex-based, flags unverified medications/labs/dates)
- PostgreSQL full data store: notes, FAISS indexes, cached summaries, audit log
- In-memory runtime cache loaded from PostgreSQL at startup (<100ms patient load)
- Pre-generated summary cache (all 5 patients cached at preprocessing time)
- Apple Silicon MPS acceleration for all PyTorch models
- React UI with AbortController request cancellation
- One-time preprocessing script (`scripts/preprocess.py`)
- Cache warm-up script (`scripts/warm_cache.py`)
- Evaluation pipeline script (`scripts/evaluate.py`) — ROUGE-L, BERTScore, Recall@3

## Not Yet Implemented
- Citation display in UI (inline citations + clickable footnotes)
- GPT-2 perplexity-based prompt injection detection (currently pattern matching only)
- BERTScore faithfulness verification (currently regex-based entity check)
- Docker Compose deployment

---

# Limitations

- Processes 5 patients from MIMIC-III (selected for clinical diversity and evaluation ground truth)
- Prompt injection detection uses simple pattern matching, not perplexity-based detection
- LLM response quality depends on extracted sentence quality and Ollama model performance

---

# License

This project uses MIMIC-III clinical data for research purposes only.
Use of the dataset requires appropriate data access approval from PhysioNet.

---

# Authors

Sai Ruchitha
Georgia State University
