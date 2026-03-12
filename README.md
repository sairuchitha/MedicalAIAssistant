
# Vision AI – Privacy-Preserving Clinical AI Assistant

Vision AI is a privacy-preserving clinical AI system that generates structured patient summaries and supports grounded question answering from electronic health records (EHR).

The system uses **MIMIC-III clinical notes**, performs **PHI masking via Microsoft Presidio**, extracts clinically relevant sentences using **BioClinicalBERT**, indexes patient notes with a **MedCPT 3-encoder RAG pipeline backed by FAISS**, and generates structured summaries and answers using a **local LLM via Ollama** — all fully on-premise with no external API calls.

---

# System Architecture

```
MIMIC-III Clinical Notes (NOTEEVENTS.csv)
↓
PHI Masking — Microsoft Presidio (pre-model)
↓
Preprocessing — Section parsing, deduplication, chunking
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
Physician UI (React) + PostgreSQL Audit Log
```

---

# Features

### 1. Patient Selection
Loads patient IDs from MIMIC-III clinical notes. Currently configured for 1 patient with up to 10 notes (`MAX_PATIENTS=1`, `MAX_NOTES_PER_PATIENT=10` in `.env`).

### 2. Two-Layer PHI Protection
- **Pre-model**: Presidio masks names, MRNs, dates, SSNs, phone numbers before any model sees text
- **Post-model**: Presidio scans every model output before display

### 3. Clinical Sentence Extraction (BioClinicalBERT)
- Mean-pool embeddings across all token positions (not CLS)
- Two-pass selection: always-keep pass (allergies, meds, diagnoses) + soft-threshold pass (≥60% of max cosine similarity)

### 4. Structured Patient Summary (BioMistral-7B)
4-section structured summary via Ollama:
- Chief Complaint
- Active Diagnoses
- Current Medications
- Recent History and Care Plan

Includes entity-level faithfulness verification (medications, dosages, lab values, dates).

### 5. RAG-Based Question Answering (MedCPT + BioMistral-7B)
Full dense-retrieval pipeline:
- **Chunking**: 200-token overlapping windows with 50-token stride, with chunk metadata (note type, date, section)
- **Indexing**: MedCPT Article Encoder → per-patient FAISS index
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
| Frontend | React, Vite, Axios |
| Backend | FastAPI, Python, Pydantic |
| LLM Serving | Ollama (`llama3.1:8b`) |
| Sentence Extraction | BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) |
| RAG Retrieval | MedCPT Article/Query/Cross-Encoder (`ncbi/MedCPT-*`) |
| Vector Search | FAISS (in-memory, per-patient) |
| PHI Masking | Microsoft Presidio |
| Database | PostgreSQL (audit logging only) |
| Dataset | MIMIC-III NOTEEVENTS |

---

# Project Structure

```
vision_ai_full_project/
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, lifespan startup
│   │   ├── config.py                # Settings (loaded from .env)
│   │   ├── api/
│   │   │   ├── patients.py          # GET /patients
│   │   │   ├── summary.py           # POST /summary
│   │   │   └── qa.py                # POST /qa
│   │   ├── services/
│   │   │   ├── mimic_loader.py      # MIMIC-III CSV loading
│   │   │   ├── phi_masking.py       # Presidio PHI masking
│   │   │   ├── section_parser.py    # Clinical section extraction
│   │   │   ├── dedup.py             # Note deduplication
│   │   │   ├── chunker.py           # Token-window chunking
│   │   │   ├── sentence_extractor.py # BioClinicalBERT extraction
│   │   │   ├── medcpt_indexer.py    # MedCPT Article Encoder + FAISS
│   │   │   ├── retriever.py         # MedCPT Query + Cross-Encoder
│   │   │   ├── summarizer.py        # BioMistral-7B summarization
│   │   │   ├── qa_service.py        # BioMistral-7B QA generation
│   │   │   ├── biomistral_client.py # Ollama LLM client
│   │   │   ├── runtime_store.py     # In-memory patient index cache
│   │   │   ├── security.py          # Prompt injection detection
│   │   │   ├── verification.py      # Faithfulness verification
│   │   │   └── audit_logger.py      # PostgreSQL audit logging
│   │   └── db/
│   │       └── postgres.py          # SQLAlchemy engine + AuditLog model
│   ├── .env                         # Environment config (see below)
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Main UI
│   │   ├── api.js                   # API client (BASE_URL: 127.0.0.1:8000)
│   │   └── components/
│   │       ├── SummaryPanel.jsx
│   │       └── QAChat.jsx
│
└── README.md
```

---

# Setup Instructions

## Prerequisites
- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running
- PostgreSQL running locally
- MIMIC-III `NOTEEVENTS.csv` placed at `backend/data/raw/NOTEEVENTS.csv`

## Pull the LLM model
```bash
ollama pull llama3.1:8b
```

## Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend runs at `http://127.0.0.1:8000`

## Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

## Environment Variables (backend/.env)
```
BIOMISTRAL_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
MAX_PATIENTS=1
MAX_NOTES_PER_PATIENT=10
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vision_ai
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
MIMIC_NOTEEVENTS_CSV=data/raw/NOTEEVENTS.csv
```

---

# Example Workflow

1. Select a patient from the dropdown
2. Click **Generate Summary** — BioClinicalBERT extracts sentences, BioMistral-7B generates 4-section summary
3. Ask clinical questions in the QA box — MedCPT retrieves relevant chunks, BioMistral-7B generates a grounded answer with citations

---

# Current Implementation Status

## Completed
- MIMIC-III data loading and preprocessing pipeline
- Microsoft Presidio PHI masking (pre-model and post-model)
- BioClinicalBERT sentence extraction (mean-pool, two-pass selection)
- BioMistral-7B 4-section structured summarization via Ollama
- Full MedCPT 3-encoder RAG pipeline (Article Encoder → FAISS → Query Encoder → Cross-Encoder reranking)
- Lookup vs Reasoning question routing
- Entity-level faithfulness verification (regex-based)
- Prompt injection pattern detection
- PostgreSQL audit logging
- React UI with summary panel and QA chat interface
- Note chunking with metadata (note type, date, section)
- Section parsing, deduplication

## Known Issues / Not Yet Wired Up
- `runtime_store.py` (patient index cache) is implemented but not called at startup — causes ~1 min load times for patients, summary, and QA
- 4 summarization LLM calls run sequentially — can be parallelized

## Not Yet Implemented
- GPT-2 perplexity-based prompt injection detection (currently only pattern matching)
- BERTScore faithfulness verification layer
- Citation display in UI (inline citations + clickable footnotes)
- Warning badges for unverified claims
- spaCy section-aware chunking (currently token-window only)
- ChromaDB persistent vector storage
- Evaluation pipeline (ROUGE-L, BERTScore, Recall@3, Hallucination rate)
- Docker Compose deployment

---

# Limitations

- Processes a small subset of MIMIC patients due to hardware constraints (currently MAX_PATIENTS=1)
- All indexes are in-memory and rebuilt on every request (caching not yet active)
- Prompt injection detection uses simple pattern matching, not perplexity-based detection
- LLM response quality depends on extracted sentence quality

---

# Future Work

- Wire `initialize_runtime()` at startup to eliminate per-request CSV reads and index rebuilds
- Parallelize 4-section summarization LLM calls
- Implement GPT-2 perplexity-based injection detection
- Add BERTScore faithfulness verification
- Add citation UI with clickable footnotes
- Add evaluation metrics (ROUGE-L, Recall@3, Hallucination rate)
- Docker Compose deployment

---

# License

This project uses MIMIC-III clinical data for research purposes only.
Use of the dataset requires appropriate data access approval from PhysioNet.

---

# Authors

Vision AI Clinical System
Georgia State University
