"""
In-memory cache of per-patient notes and FAISS indexes.

Loaded once at server startup from PostgreSQL (populated by scripts/preprocess.py).
All three API endpoints read from this cache — zero CSV reads at request time.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.db.postgres import FaissIndex, Note, Patient, SessionLocal

# {patient_id: [{"masked_text", "row_id", "chart_date", "category", "description"}]}
_patient_notes: Dict[int, List[dict]] = defaultdict(list)

# {patient_id: (faiss_index, chunk_meta_list)}
_patient_indexes: Dict[int, Tuple[object, List[dict]]] = {}


def initialize_runtime():
    """Load all patients, notes, and FAISS indexes from Postgres into RAM."""
    global _patient_notes, _patient_indexes
    _patient_notes = defaultdict(list)
    _patient_indexes = {}

    db = SessionLocal()
    try:
        patients = db.query(Patient).all()
        if not patients:
            print("[runtime_store] WARNING: No patients found in database. Run scripts/preprocess.py first.")
            return

        print(f"[runtime_store] Loading {len(patients)} patients from Postgres...")

        for patient in patients:
            pid = patient.subject_id

            # Load masked notes with metadata
            notes = db.query(Note).filter(Note.subject_id == pid).all()
            for note in notes:
                _patient_notes[pid].append({
                    "masked_text": note.masked_text,
                    "row_id": note.row_id or "",
                    "chart_date": note.chart_date or "",
                    "category": note.category or "",
                    "description": note.description or "",
                })

            # Load and deserialize FAISS index
            faiss_row = db.query(FaissIndex).filter(FaissIndex.subject_id == pid).first()
            if faiss_row:
                index_bytes = np.frombuffer(faiss_row.index_data, dtype=np.uint8)
                index = faiss.deserialize_index(index_bytes)
                _patient_indexes[pid] = (index, faiss_row.chunk_meta)
                print(f"[runtime_store]   Patient {pid}: {len(notes)} notes, {len(faiss_row.chunk_meta)} chunks loaded")
            else:
                print(f"[runtime_store]   Patient {pid}: notes loaded but no FAISS index found")

        print(f"[runtime_store] Ready. {len(_patient_notes)} patients cached.")

    finally:
        db.close()


def get_patient_notes(patient_id: int) -> List[dict]:
    """Returns list of note dicts: {masked_text, row_id, chart_date, category, description}"""
    return _patient_notes.get(patient_id, [])


def get_patient_index(patient_id: int) -> Optional[Tuple[object, List[dict]]]:
    """Returns (faiss_index, chunk_meta) or None."""
    return _patient_indexes.get(patient_id)


def get_all_patient_ids() -> List[int]:
    return sorted(_patient_notes.keys())
