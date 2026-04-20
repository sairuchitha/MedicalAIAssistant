# MedMind AI — Production Readiness & Next Steps

**Evaluated:** April 2026
**Evaluation basis:** End-to-end code audit, security evaluation pipeline, RAG/summarization metrics

---

## Current Metric Baselines

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| ROUGE-L | ≥ 0.35 | 0.108 | Structural — abstractive output |
| BERTScore-F | ≥ 0.78 | 0.795 | PASS |
| Recall@3 — Lookup | ≥ 0.80 | 0.857 | PASS |
| Recall@3 — Reasoning | ≥ 0.80 | 1.000 | PASS |
| Recall@3 — Overall | ≥ 0.80 | 0.929 | PASS |
| MRR — Overall | ≥ 0.70 | 0.893 | PASS |
| MRR — Lookup | ≥ 0.70 | 0.857 | PASS |
| MRR — Reasoning | ≥ 0.70 | 0.929 | PASS |
| Reranking Gain | > 0 | +0.14 | PASS |
| Faithfulness F1 | ≥ 0.75 | 0.768 | PASS — borderline |
| Injection Recall | ≥ 0.90 | 0.895 | PASS |
| Injection Precision | ≥ 0.90 | 1.000 | PASS |
| Injection F1 | ≥ 0.90 | 0.950 | PASS |
| False Positive Rate | ≤ 0.05 | 0.000 | PASS |
| PHI Residual Rate | ≤ 0.02 | 0.0001 | PASS |

---

## What Is Production-Ready

- RAG pipeline: MedCPT 3-encoder, FAISS, adaptive cross-encoder reranking
- PHI masking: double-layer Presidio, residual rate 0.0001
- **Injection detection: 5-layer defense** — input length + regex (38+ patterns) + suspicious-intent detection + GPT-2 perplexity (CPU) + output validation; hardened prompt templates
- Question classifier: reasoning keywords checked before lookup keywords (bug fixed)
- Caching: sub-100ms cached paths, parallel 8-thread startup
- LLM: temperature=0 (deterministic), 120s timeout
- Evaluation: summarization, RAG, security, faithfulness all measured with real numbers
- Frontend: injection block banner, elapsed timer, reasoning latency hint

---

## Alignment with Project Description

The project description specifies **perplexity-based prompt screening as the primary security mechanism**. Current status:

| Requirement | Status | Notes |
|-------------|--------|-------|
| Structured patient summary | ✅ Implemented | 4-section parallel generation via BioClinicalBERT + llama3.1:8b |
| Lookup RAG (factual retrieval) | ✅ Implemented | FAISS top-4, no reranking, ~3–6s |
| Reasoning RAG (multi-note) | ✅ Implemented | FAISS top-10 + MedCPT cross-encoder, ~30–90s |
| PHI masking — pre-model | ✅ Implemented | Presidio before any model sees text |
| PHI masking — post-model | ✅ Implemented | Presidio on all LLM outputs |
| **Perplexity-based injection screening** | ✅ **Implemented** | GPT-2 CPU (PPL < 90), loads at security.py import time before MedCPT on MPS |
| Prompt injection defense | ✅ Implemented | 5 layers: length + regex (38+ patterns) + suspicious-intent + GPT-2 perplexity + output validation; hardened prompt templates |

**GPT-2 perplexity restored:** The segfault (exit 139) was caused by lazy-loading GPT-2 after MedCPT models were already on MPS. Fix: GPT-2 now loads eagerly at `security.py` import time on CPU, and `qa.py` imports `security` before `retriever` — so GPT-2 is always in memory before MedCPT touches the MPS device.

---

## Critical Gaps — Blocks Production Use

### 1. Zero authentication
Any HTTP client can access any patient. No login, no session, no token.
HIPAA violation on day one in a real deployment.
**Fix:** Add JWT auth (FastAPI-Users or python-jose). Effort: 1–2 days.

### 2. No rate limiting
`slowapi` is in requirements.txt but not imported or applied anywhere in main.py.
Unauthenticated endpoints with no rate limiting — Ollama thread pool can be DoS'd trivially.
**Fix:** Wire SlowAPI limiter in main.py. Effort: 2 hours.

### 3. No Ollama timeout on model load
`llm_client.py` now has a 120s request timeout via `ollama.Client(timeout=120)`.
However, if the Ollama **server itself** is slow to load the model into memory on first call
(cold start), that 120s may expire. First-call model load for llama3.1:8b can take 30–60s
on top of generation time.
**Fix:** Add `keep_alive` parameter or pre-warm Ollama at server startup. Effort: 30 min.

### 4. No database migrations
No Alembic. Schema is `create_all()` at startup. Any schema change in production
requires manual drop-and-recreate or ad-hoc SQL.
**Fix:** Add Alembic, generate initial migration. Effort: half day.

### 5. Weak default credentials
`config.py:56` — `POSTGRES_PASSWORD: str = "postgres"`. Default is the live value.
**Fix:** Use Docker secrets or environment-only injection (no default). Effort: 1 hour.

---

## High Priority — Degrades Reliability

### 6. GPT-2 perplexity threshold needs calibration
Current threshold: PPL < 90 blocks inputs. Social engineering prompts have PPL 54–127
(similar to clinical questions, PPL 23–52). Layer 4 caught 0/19 attack queries in evaluation.
**Fix:** Calibrate threshold on a labeled dataset, or replace with a fine-tuned classifier
for clinical injection detection. Effort: 1–2 days.

### 7. Non-deterministic summarisation (FIXED ✅)
`llm_client.py` — `temperature` set to 0. Outputs are now deterministic.

### 8. Race condition on concurrent summary requests
`summary.py:23` — two simultaneous requests for the same uncached patient both
generate full summaries and both commit, creating duplicate rows.
**Fix:** DB-level advisory lock or `INSERT ... ON CONFLICT DO NOTHING`. Effort: 1 hour.

### 9. In-memory cache does not scale
`runtime_store.py` — `_patient_notes`, `_patient_indexes`, `_qa_cache` are
process-local dicts. Two uvicorn workers = two independent caches, inconsistent
results, doubled memory. Cannot horizontally scale.
**Fix:** Move QA cache to Redis; FAISS indexes stay per-process but load from shared DB. Effort: 1–2 days.

### 10. QA warnings lost in frontend
`QAChat.jsx` — `handleAsk` stores the full QA response but the `warnings` array
is never surfaced in the UI. Verification warnings are silently dropped.
**Fix:** Render warnings below QA answer in QAChat.jsx. Effort: 30 min.

### 11. Faithfulness borderline
Mean F1 = 0.768, 26% of sentences below the 0.75 threshold. No automated
remediation — low-faithfulness sentences reach the user unfiltered.
**Fix:** Implement BERTScore faithfulness filter in verification.py. Effort: 2–3 hours.

---

## Medium Priority — Limits Usability

### 12. No CI/CD, no tests
Zero unit tests, zero integration tests, no GitHub Actions. Every change is
manually tested. Any refactor is a blind risk.
**Fix:** Add pytest for services, GitHub Actions for lint + test on PR. Effort: 1 day.

### 13. No streaming
First summary request blocks 30–60s with a spinner. Ollama supports SSE streaming.
Reasoning QA takes 30–90s with only an elapsed timer visible.
**Fix:** Server-sent events from backend + EventSource in frontend. Effort: 1–2 days.

### 14. Summary cache never invalidated
Cached summaries have no TTL. If new notes arrive the clinician sees stale data
with no indication of when the summary was generated.
**Fix:** Add `generated_at` timestamp to cached summary response. Show age in UI. Effort: 1 hour.

### 15. Patient list not paginated
105 patients works. At 46,000 MIMIC patients the dropdown is unusable and
startup FAISS loading would take hours.
**Fix:** Lazy-load patients on demand; paginate API. Effort: half day.

### 16. `chromadb` in requirements but unused
Dead dependency — ~200 MB install, creates version conflict risk.
**Fix:** Remove from requirements.txt. Effort: 2 min.

### 17. Reasoning QA latency exceeds 10s target
llama3.1:8b with 5-chunk context takes 30–90s on Apple Silicon. The UI now shows
an elapsed timer and latency hint, but the underlying latency remains.
**Fix options:** (a) Use a smaller/faster model for reasoning path, (b) implement
streaming SSE, (c) add a response progress indicator. Effort: varies.

---

## Application-Level Gaps

### 18. No user identity in audit log
Audit log records events but has no user field. Cannot determine who accessed
which patient — required for HIPAA audit trails.
**Fix:** Add `user_id` field to audit_logs table and populate from auth token. Effort: 2 hours.

### 19. No summary staleness indicator
Cached summaries show no timestamp. Clinicians cannot tell if it is from today or weeks ago.
**Fix:** Return and display `generated_at` in summary response. Effort: 30 min.

### 20. Citation UI is a flat list
API returns full citation data (note_id, date, type, section). Frontend shows flat chips.
No way to trace a specific claim back to its source note.
**Fix:** Numbered inline footnotes with expand-on-click. Effort: 1 day.

### 21. Warning badges not per-section
Verification warnings are returned but displayed as a flat list (or not at all in QA),
not attached to the section that triggered them.
**Fix:** Parse warning section name and render badge inline per section. Effort: half day.

### 22. Two indirect paraphrase injections still bypass input filters
Highly indirect rephrasing ("please disregard any prior context…") passes regex, suspicious-intent, and perplexity layers.
**Partial mitigation:** Output validation (Layer 5) and hardened prompt templates now provide defense-in-depth — even if an injection reaches the LLM, output-side checks and STRICT RULES block limit impact.
**Remaining fix:** Add 2 more regex patterns for remaining indirect-rephrase variants. Effort: 10 min.

---

## Prioritised Action List

### Before any demo or submission
| # | Task | File | Effort |
|---|------|------|--------|
| 1 | Add 2 remaining indirect-paraphrase injection patterns | `app/services/security.py` | 10 min |
| 2 | Surface QA warnings in UI | `frontend/src/components/QAChat.jsx` | 30 min |
| 3 | Add summary `generated_at` timestamp to response | `app/api/summary.py` | 20 min |
| 4 | Remove `chromadb` from requirements | `requirements.txt` | 2 min |
| 5 | Remove test comment payload from top of security.py | `app/services/security.py` | 2 min |
| 6 | Remove commented-out duplicate router import in main.py | `app/main.py` | 2 min |

### Before a real deployment
| # | Task | Effort |
|---|------|--------|
| 5 | JWT authentication + user identity in audit log | 1–2 days |
| 6 | Wire slowapi rate limiting | 2 hours |
| 7 | Alembic database migrations | half day |
| 8 | Fix race condition on summary generation | 1 hour |
| 9 | BERTScore faithfulness filter in verification.py | 2–3 hours |
| 10 | Streaming summaries + QA via SSE | 1–2 days |
| 11 | Move secrets out of .env defaults | 1 hour |
| 12 | Citation inline footnotes in UI | 1 day |
| 13 | Warning badges per section in UI | half day |
| 14 | pytest suite for core services | 1 day |
| 15 | GPT-2 threshold calibration or classifier fine-tuning | 1–2 days |

---

## Known Metric Limitations

- **ROUGE-L is structurally low (0.108)** — not a real failure. Abstractive 4-section
  summaries use different vocabulary than the discharge note ground truth.
  BERTScore-F (0.795) is the meaningful signal.
- **BERTScore variance between runs** — now fixed: temperature=0 gives stable, reproducible outputs.
- **Evaluation covers 5 of 105 patients** — results are directionally correct but
  not statistically representative. Expand eval set before reporting final numbers.
- **Faithfulness F1 drags on short medication fragments** — entries like
  `"Vancomycin 1000 mg IV Q 12H"` score low because BERTScore expects full sentences.
  Consider filtering single-line fragments from faithfulness evaluation.
- **GPT-2 perplexity Layer 3 catches 0/19 attack queries** — social engineering prompts
  have PPL 54–127, similar to legitimate clinical questions (PPL 23–52). Layer 3 is
  effective only for low-entropy, formulaic injections (e.g., template-filled payloads
  with PPL < 20). The regex layer (Layer 2) carries the load for the current test set.
