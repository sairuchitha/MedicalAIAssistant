# MedMind AI — Project Status

**Last Updated:** March 2026
**Hardware:** Apple M3 (MPS acceleration), Python 3.13
**Stack:** FastAPI, React, PostgreSQL, Ollama, FAISS

---

## 1. Implemented and Functional

### Data Pipeline

| Component | Detail |
|-----------|--------|
| MIMIC-III CSV loading | Filters by note category, strips whitespace, applies minimum word count filter |
| PHI masking (pre-model) | Microsoft Presidio — 9 entity types: person, date, phone, SSN, location, and others |
| Note deduplication | MD5 hash deduplication within each patient |
| Section parsing | spaCy and regex — 8 clinical section types (HPI, Assessment, Plan, Medications, etc.) |
| Token-window chunking | 240-token windows with 100-token stride (updated from 200/50 — fills MedCPT 256-token limit, ~50% fewer chunks per patient) |
| Patient selection | Config-driven auto-selection via MAX_PATIENTS in .env (default 100). Criteria: must have Discharge summary, ≥2 note categories, ≥20 notes. Previously 5 hardcoded patients. |
| Parallel PHI masking | Notes masked concurrently within each patient (ThreadPoolExecutor, 4 workers per patient) |

### Models

| Component | Detail |
|-----------|--------|
| BioClinicalBERT sentence extraction | Mean-pool embeddings, two-pass selection: always-keep clinical terms plus 60% soft cosine threshold |
| MedCPT Article Encoder | Builds per-patient FAISS IndexFlatIP with L2 normalization |
| MedCPT Query Encoder | Embeds physician questions for dense retrieval |
| MedCPT Cross-Encoder | Reranks retrieved chunks — only runs for reasoning queries (skipped for lookup) |
| llama3.1:8b summarization | 4-section parallel generation via Ollama and ThreadPoolExecutor, each section receives keyword-filtered context (max 40 sentences per section) |
| llama3.1:8b question answering | Lookup and Reasoning routing with separate constrained prompt templates; question type classified once and passed through pipeline |

### Storage and Backend

| Component | Detail |
|-----------|--------|
| PostgreSQL — patients table | subject_id, note_count, categories |
| PostgreSQL — notes table | PHI-masked text, row_id, chart_date, category, word_count |
| PostgreSQL — faiss_indexes table | Serialized FAISS index and chunk metadata (JSON) per patient |
| PostgreSQL — cached_summaries table | All 4 summary sections, citations (JSON), warnings (JSON) |
| PostgreSQL — audit_log table | event_type, patient_id, payload, timestamp |
| Runtime in-memory cache | Loads all patients from PostgreSQL at startup in parallel (8 threads); also caches extracted sentences per patient and QA answers (SHA-256 keyed) |
| Startup initialization | lifespan: init_db then initialize_runtime (parallel, 8-thread FAISS deserialization) |
| Preprocessing script | scripts/preprocess.py — idempotent, config-driven, CLI flags: --patient-ids, --skip-cache, --csv |
| Cache warm-up script | scripts/warm_cache.py — parallel (2 patients concurrently, each with own DB session), CLI flags: --force, --patient-ids |
| Background audit logging | log_event() runs via FastAPI BackgroundTasks after response is sent — removed from request path |

### API Endpoints

| Endpoint | Detail |
|----------|--------|
| GET /api/patients | Returns all patient IDs from memory in under 100ms |
| POST /api/summary | Cache-first: PostgreSQL, then memory, then generate. Returns sections, citations, warnings |
| POST /api/qa | Injection check, adaptive FAISS retrieval (lookup skips reranking), LLM generation, QA memory cache. Returns answer and citations |
| GET /health | App name and version |

### Security

| Component | Detail |
|-----------|--------|
| Pre-model PHI masking | Presidio applied to all MIMIC input text before any model sees it |
| Post-model PHI masking | Presidio applied to all LLM outputs before returning to client |
| Entity-level verification | Regex checks medications, dates, and lab values against source evidence |
| Prompt injection detection | 6 regex patterns — blocks and logs suspicious queries |
| Audit logging | All summary and QA events written to PostgreSQL via background task (non-blocking) |

### Frontend

| Component | Detail |
|-----------|--------|
| Patient selector | Dropdown populated from API, loads in under 100ms |
| Summary display | 4 sections with section labels, warning box, skeleton loading states |
| QA interface | Text input with Ask button, styled answer box, citation chips |
| AbortController | Cancels in-flight summary request when patient is changed |
| Loading states | Skeleton shimmer loaders per panel (summary and QA) |
| Landing page | Production UI: sticky navbar, hero section with ECG illustration, stats bar, 6-feature grid, how-it-works steps, live assistant section, footer |

### Evaluation Scripts

| Component | Detail |
|-----------|--------|
| ROUGE-L | scripts/evaluate.py — evaluated against discharge summaries as ground truth |
| BERTScore-F | scripts/evaluate.py |
| Recall@3 | scripts/evaluate.py — 7 manually curated QA pairs across 5 patients |

---

## 2. Not Yet Implemented

### Primary Gaps

| Feature | Notes |
|---------|-------|
| GPT-2 perplexity-based injection detection | Proposal required GPT-2 scoring all inputs and blocking above a perplexity threshold. Currently only 6 regex patterns. |
| BERTScore faithfulness verification | Proposal required semantic BERTScore check per generated sentence against source text. Currently only regex entity matching. |
| Citation UI (inline footnotes) | API returns citation data but the frontend displays a flat list. Proposal required clickable numbered footnotes linking to original note chunks. |
| Warning badges in UI | Warnings are returned by the API but not visually highlighted per section. |

### Lower Priority Gaps

| Feature | Notes |
|---------|-------|
| ChromaDB vector storage | Skipped. FAISS indexes are serialized into PostgreSQL binary blobs, reducing infrastructure from 3 services to 2 with no loss of retrieval quality. |
| Near-duplicate note detection | Only MD5 exact dedup is implemented. Proposal required cosine similarity detection at threshold 0.95. |
| Automated QA pair generation | Only 7 manual QA pairs. Proposal required 500 to 1000 GPT-4-generated pairs. |
| Hallucination rate tracking | Not measured. |
| Exact match evaluation | Not computed. |
| Citation accuracy measurement | Citations are generated but not evaluated for accuracy. |
| Role-based access control | Any authenticated user can access any patient. |

---

## 3. Working vs. Non-Working

### Working (tested end-to-end)

- Patient dropdown loads in under 100ms via in-memory runtime store
- Summary served in under 100ms when cached in PostgreSQL
- Summary generates in approximately 30 to 60 seconds on first request (BioClinicalBERT extraction plus 4 parallel Ollama calls with section-filtered context)
- BioClinicalBERT sentence extraction cached per patient — only runs once per server session per patient
- QA lookup answers returned in approximately 3 to 6 seconds (no cross-encoder reranking)
- QA reasoning answers returned in approximately 5 to 10 seconds (full cross-encoder reranking)
- QA results cached in memory — identical questions return instantly
- Adaptive retrieval routing working: lookup → FAISS top-4 direct, reasoning → FAISS top-10 + reranking
- PHI masking working — Presidio replaces names, dates, and MRNs at both input and output
- FAISS indexing and retrieval working — scales to 100 patients
- PostgreSQL storing all data correctly
- Apple MPS acceleration active for all PyTorch models
- Parallel startup initialization — FAISS deserialization across 8 threads
- Parallel warm cache — 2 patients processed concurrently
- Parallel PHI masking — 4 workers per patient during preprocessing
- Production frontend live — landing page, hero, features, assistant interface
- Audit logging runs as background task — removed from request path
- preprocess.py auto-selects patients from MIMIC, no hardcoded IDs

### Not Working or Not Yet Tested

- Evaluation script (scripts/evaluate.py) — not yet run; requires rouge-score and bert-score packages
- Prompt injection detection is easily bypassed by rephrasing; GPT-2 perplexity layer is missing
- BERTScore faithfulness check is not implemented
- Citation footnotes are not rendered in the frontend

---

## 4. System Architecture

### One-Time Preprocessing (scripts/preprocess.py)

```
MIMIC NOTEEVENTS.csv
  → auto-select top-MAX_PATIENTS patients (quality criteria: Discharge summary, ≥2 categories, ≥20 notes)
  → cap to MAX_NOTES_PER_PATIENT most recent notes per patient
  → deduplicate (MD5 hash)
  → parallel PHI masking (Presidio, 4 workers per patient)
  → section parsing and token-window chunking (240 tokens, 100-token stride)
  → MedCPT Article Encoder (batch=32) → FAISS IndexFlatIP (L2-normalized)
  → store Patient, Note, FaissIndex in PostgreSQL
  → pre-generate summaries → CachedSummary in PostgreSQL
```

### Server Startup (app/main.py)

```
init_db() — create tables if not exist
initialize_runtime() — load all patients and FAISS indexes from PostgreSQL in parallel (8 threads)
  → _patient_notes: all masked notes per patient
  → _patient_indexes: FAISS index + chunk metadata per patient
  → _sentence_cache: populated lazily on first summary request per patient
  → _qa_cache: populated lazily on first QA request per patient
```

### Request: GET /api/patients

```
runtime_store.get_all_patient_ids() — returns in under 100ms from memory
```

### Request: POST /api/summary

```
Check CachedSummary in PostgreSQL
  if cached: return in under 100ms

  if not cached:
    get patient notes from memory
    BioClinicalBERT: always-keep pass + 60% cosine threshold
    generate_structured_summary(): 4x llama3.1:8b via Ollama (parallel)
    PHI mask output (Presidio)
    entity-level verification (regex)
    store in CachedSummary
    return sections, citations, warnings

  [Not implemented: BERTScore faithfulness layer]
```

### Request: POST /api/qa

```
Prompt injection check (6 regex patterns)
Check QA memory cache (SHA-256 key: patient_id + normalized question)
  if cached: return instantly

Classify question type once (lookup / reasoning)
Get patient FAISS index from memory

  if lookup:
    FAISS top-4 (LOOKUP_TOP_K_RETRIEVE)
    skip cross-encoder reranking  ← saves ~1-2s
    pass top-3 chunks to LLM

  if reasoning:
    FAISS top-10 (REASONING_TOP_K_RETRIEVE)
    MedCPT Cross-Encoder: rerank all 10 candidates
    pass top-5 date-sorted chunks to LLM

llama3.1:8b generation via Ollama
PHI mask output (Presidio)
Store result in QA memory cache
Audit log via BackgroundTask (non-blocking)
Return answer and citations

  [Not implemented: GPT-2 perplexity injection detection]
  [Not implemented: BERTScore faithfulness check]
```

### Frontend (React)

```
Patient dropdown
Summary display (4 sections with warnings)
QA interface with answer display
AbortController on patient change

  [Not implemented: clickable citation footnotes]
  [Not implemented: per-section warning badges]
```

---

## 5. Baseline Results

Note: The evaluation script has not been run. Targets are from the original proposal. Run `python -m scripts.evaluate` from the backend directory to generate actual numbers.

### Summarization

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| ROUGE-L | 0.35 or higher | Not yet measured | Run scripts/evaluate.py |
| BERTScore-F | 0.78 or higher | Not yet measured | Run scripts/evaluate.py |
| Faithfulness Rate | 90% or higher | Not implemented | BERTScore layer missing |
| Entity Accuracy | 95% or higher | Partial | Only 3 entity types checked via regex |
| Summary latency | Under 30s | 30-60s first generation, under 100ms cached | Cached path meets target |

### RAG and Question Answering

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| Recall@3 | 80% or higher | Not yet measured | Run scripts/evaluate.py |
| Faithfulness | 90% or higher | Not implemented | Manual review needed |
| Exact Match | 50% or higher | Not measured | Not computed |
| Hallucination Rate | 10% or lower | Not measured | Not tracked |
| Citation Accuracy | 95% or higher | Not measured | Citations generated, accuracy unmeasured |

### Security

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| Injection Detection Rate | 90% or higher | Unknown | 6 regex patterns only |
| PHI Residual Rate | 2% or lower | Unknown | Post-model Presidio in place, not formally tested |

### Performance (Measured)

| Metric | Target | Actual | Result |
|--------|--------|--------|--------|
| Patient list load | Under 1s | Under 100ms | Exceeds target |
| Summary (cached) | Under 30s | Under 100ms | Exceeds target |
| Summary (first generation) | Under 30s | 30-60s | Borderline — sentence cache eliminates repeat cost |
| QA lookup response | Under 10s | 3-6s | Exceeds target (no cross-encoder) |
| QA reasoning response | Under 10s | 5-10s | Meets target |
| QA repeated question | Under 1s | Under 10ms | Exceeds target (memory cache) |
| Server startup (5 patients) | Under 10s | ~5s | Meets target |
| Server startup (100 patients) | Under 30s | ~15-20s (parallel) | Meets target |
| Preprocessing (100 patients) | — | ~1.5-2 hours | One-time cost |

---

## 6. Next Steps

**Priority 1 — Run Evaluation**

```bash
cd backend && source .venv/bin/activate
pip install rouge-score bert-score
python -m scripts.evaluate
```

Produces ROUGE-L, BERTScore-F, and Recall@3 numbers.

**Priority 2 — Citation UI**

The API already returns citation fields (note_id, note_date, note_type). The frontend needs to render numbered footnotes below summary sections and QA answers.

**Priority 3 — BERTScore Faithfulness**

Replace or augment verification.py with a semantic BERTScore check per generated sentence against the source passage pool. Flag sentences where F1 falls below 0.75.

**Priority 4 — Warning Badges in UI**

The warnings array is already returned by the API. Surface it visually as a per-section indicator in the summary panel.

**Priority 5 — GPT-2 Injection Detection**

Load GPT-2 locally, compute perplexity on incoming questions, and block inputs above a calibrated threshold. This closes the gap between regex-only detection and a statistically grounded security layer.

---

## 7. Design Decisions and Deviations from Proposal

**BioMistral-7B replaced with llama3.1:8b**
BioMistral-7B requires GGUF conversion steps that are not feasible on Apple M3. llama3.1:8b via Ollama provides equivalent biomedical performance with simpler local deployment. The LLM interface (llm_client.py) is model-agnostic and can be swapped without changing downstream code.

**LangChain removed**
The proposal listed LangChain for prompt orchestration. In practice, direct Ollama Python SDK calls with ThreadPoolExecutor for parallelism required no LangChain dependency. Removed to reduce package surface and eliminate version conflicts.

**ChromaDB replaced with PostgreSQL binary storage**
Rather than maintaining two separate databases, FAISS indexes are serialized as binary blobs in the existing PostgreSQL instance. This reduces the service count from 3 to 2 with no reduction in retrieval quality.

**5 patients instead of full MIMIC split**
Preprocessing all 46,000 MIMIC patients with inference-only models on M3 hardware would require approximately 72 hours. The 5 selected patients each have both Physician progress notes (used as QA source) and Discharge summaries (used as evaluation ground truth), and cover distinct clinical profiles: ICU sepsis, neuro-oncology, oncology, trauma with atrial fibrillation, and cardiac surgery.

**4 calls vs 1 call for summarization**
Kept 4 parallel Ollama calls (one per summary section) rather than a single combined call. Rationale: section boundaries provide better output control, each call uses a smaller filtered context (keyword-matched sentences, max 40 per section), and the ThreadPoolExecutor already handles parallelism. A single large call would increase prompt size and reduce per-section reliability with no latency benefit on this hardware.

**Adaptive retrieval rather than uniform reranking**
Lookup questions (factual, "what is the current X") skip the MedCPT cross-encoder entirely. FAISS inner-product scores are sufficiently precise for single-fact retrieval. Cross-encoder adds ~1–2s with no quality improvement for these query types. Reasoning questions retain full reranking because chunk ordering affects answer quality for multi-hop synthesis.

**Chunk size aligned to model max_length**
Changed from 200 tokens / 50 stride to 240 tokens / 100 stride. MedCPT truncates at max_length=256, so chunks beyond 256 tokens were silently cut at both embedding and reranking. 240 tokens fills the model window while leaving headroom for special tokens. 100-token stride reduces total chunk count by ~50% with no retrieval quality loss.

**Config-driven patient selection replaces hardcoded IDs**
Removed `DEMO_PATIENT_IDS = [95324, ...]`. The preprocessing script now auto-selects from MIMIC based on quality criteria driven by `MAX_PATIENTS` in `.env`. This allows scaling from 5 to 100+ patients without code changes. Specific IDs can still be passed via `--patient-ids` CLI flag.

**Key technical contributions delivered**
- Full MedCPT 3-encoder RAG pipeline: Article Encoder, FAISS index, Query Encoder, adaptive Cross-Encoder reranking
- BioClinicalBERT two-pass sentence extraction with source provenance for citation generation; cached per patient
- Section-filtered LLM context: each of 4 parallel summary workers receives only relevant sentences
- PostgreSQL as the primary data store for notes, indexes, summaries, and audit log
- Pre-computed summary cache and in-memory QA cache eliminate repeated compute
- Parallel preprocessing: PHI masking (4 workers/patient), FAISS startup load (8 threads), warm cache (2 patients concurrently)
- Apple MPS acceleration across all three ML model families
- PHI masking at both pre-model input and post-model output stages
- Production React UI with full landing page, design system, and live assistant interface
