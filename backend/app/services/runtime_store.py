"""
In-memory cache of per-patient notes and FAISS indexes.

Loaded once at server startup from PostgreSQL (populated by scripts/preprocess.py).
All three API endpoints read from this cache — zero CSV reads at request time.
"""

import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.db.postgres import FaissIndex, Note, Patient, SessionLocal

# {patient_id: [{"masked_text", "row_id", "chart_date", "category", "description"}]}
_patient_notes: Dict[int, List[dict]] = defaultdict(list)

# {patient_id: (faiss_index, chunk_meta_list)}
_patient_indexes: Dict[int, Tuple[object, List[dict]]] = {}

# {patient_id: [{"sentence", "note_id", "note_date", "note_type"}]}
# Populated after first sentence extraction so BioClinicalBERT only runs once per patient.
_sentence_cache: Dict[int, List[dict]] = {}

# {cache_key: {"answer": str, "citations": [...], "question_type": str, "warnings": []}}
# cache_key = sha256(f"{patient_id}:{question_lower}")
# Avoids re-running retrieval + Ollama for identical questions.
_qa_cache: Dict[str, dict] = {}


def _load_one_patient(pid: int) -> tuple:
    """Load notes and FAISS index for a single patient.  Runs in a thread."""
    db = SessionLocal()
    try:
        notes = db.query(Note).filter(Note.subject_id == pid).all()
        note_dicts = [
            {
                "masked_text": n.masked_text,
                "row_id":      n.row_id or "",
                "chart_date":  n.chart_date or "",
                "category":    n.category or "",
                "description": n.description or "",
            }
            for n in notes
        ]

        faiss_row = db.query(FaissIndex).filter(FaissIndex.subject_id == pid).first()
        if faiss_row:
            index_bytes = np.frombuffer(faiss_row.index_data, dtype=np.uint8)
            index = faiss.deserialize_index(index_bytes)
            return pid, note_dicts, (index, faiss_row.chunk_meta)
        else:
            return pid, note_dicts, None
    finally:
        db.close()


def initialize_runtime():
    """Load all patients, notes, and FAISS indexes from Postgres into RAM.

    Patients are loaded in parallel (ThreadPoolExecutor) so startup time
    scales sub-linearly — 100 patients load roughly as fast as ~20 sequential.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    global _patient_notes, _patient_indexes
    _patient_notes = defaultdict(list)
    _patient_indexes = {}

    db = SessionLocal()
    try:
        patients = db.query(Patient).all()
        if not patients:
            print("[runtime_store] WARNING: No patients found. Run scripts/preprocess.py first.")
            return
        pids = [p.subject_id for p in patients]
    finally:
        db.close()

    print(f"[runtime_store] Loading {len(pids)} patients in parallel ...")
    t0 = __import__("time").time()

    # Cap workers at 8 — beyond that PostgreSQL connection overhead dominates
    workers = min(8, len(pids))
    loaded = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_load_one_patient, pid): pid for pid in pids}
        for future in as_completed(futures):
            pid, note_dicts, index_data = future.result()
            for nd in note_dicts:
                _patient_notes[pid].append(nd)
            if index_data:
                _patient_indexes[pid] = index_data
            loaded += 1
            print(f"[runtime_store]   {loaded}/{len(pids)} patients loaded\r", end="", flush=True)

    elapsed = __import__("time").time() - t0
    print(f"\n[runtime_store] Ready — {len(_patient_notes)} patients, "
          f"{sum(len(v) for v in _patient_notes.values())} notes loaded in {elapsed:.1f}s")


def get_patient_notes(patient_id: int) -> List[dict]:
    """Returns list of note dicts: {masked_text, row_id, chart_date, category, description}"""
    return _patient_notes.get(patient_id, [])


def get_patient_index(patient_id: int) -> Optional[Tuple[object, List[dict]]]:
    """Returns (faiss_index, chunk_meta) or None."""
    return _patient_indexes.get(patient_id)


def get_all_patient_ids() -> List[int]:
    return sorted(_patient_notes.keys())


# ── Sentence embedding cache ─────────────────────────────────────────────────

def get_cached_sentences(patient_id: int) -> Optional[List[dict]]:
    """Return previously extracted sentences for this patient, or None."""
    return _sentence_cache.get(patient_id)


def set_cached_sentences(patient_id: int, sentences: List[dict]) -> None:
    """Store extracted sentences so BioClinicalBERT doesn't re-run for the same patient."""
    _sentence_cache[patient_id] = sentences


# ── QA answer cache ──────────────────────────────────────────────────────────

def _qa_key(patient_id: int, question: str) -> str:
    raw = f"{patient_id}:{question.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached_qa(patient_id: int, question: str) -> Optional[dict]:
    """Return a cached QA result or None if not seen before."""
    return _qa_cache.get(_qa_key(patient_id, question))


def set_cached_qa(patient_id: int, question: str, result: dict) -> None:
    """Cache a QA result keyed by (patient_id, normalized question)."""
    _qa_cache[_qa_key(patient_id, question)] = result
