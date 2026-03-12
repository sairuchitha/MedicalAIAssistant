from collections import defaultdict
from typing import Dict, List, Tuple

from app.services.chunker import chunk_note
from app.services.dedup import deduplicate_notes
from app.services.medcpt_indexer import build_faiss_index
from app.services.mimic_loader import load_mvp_patient_notes
from app.services.phi_masking import mask_phi

_patient_notes: Dict[int, List[str]] = defaultdict(list)
_patient_indexes: Dict[int, Tuple[object, List[dict]]] = {}


def initialize_runtime(limit_patients: int = 5):
    global _patient_notes, _patient_indexes
    df = load_mvp_patient_notes(max_patients=limit_patients)
    df = deduplicate_notes(df)
    all_chunks_by_patient = defaultdict(list)
    for _, row in df.iterrows():
        pid = int(row["SUBJECT_ID"])
        text = mask_phi(str(row["TEXT"]))
        _patient_notes[pid].append(text)
        chunks = chunk_note(
            text=text,
            patient_id=pid,
            note_id=str(row.get("ROW_ID", "")),
            note_date=str(row.get("CHARTDATE", "")),
            note_type=str(row.get("CATEGORY", "")),
        )
        all_chunks_by_patient[pid].extend(chunks)
    for pid, chunks in all_chunks_by_patient.items():
        if chunks:
            index, chunk_meta = build_faiss_index(chunks)
            _patient_indexes[pid] = (index, chunk_meta)


def get_patient_notes(patient_id: int):
    return _patient_notes.get(patient_id, [])


def get_patient_index(patient_id: int):
    return _patient_indexes.get(patient_id)


def get_all_patient_ids():
    return sorted(_patient_notes.keys())
