from typing import Dict, List
from app.services.section_parser import split_sections


def token_window_chunks(words: List[str], chunk_size: int = 240, stride: int = 100) -> List[str]:
    chunks = []
    step = max(1, chunk_size - stride)
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
    return chunks


def chunk_note(text: str, patient_id: int, note_id: str, note_date: str, note_type: str) -> List[Dict]:
    sections = split_sections(text)
    chunks = []
    for section_name, section_text in sections.items():
        if len(section_text.split()) < 30:
            continue
        if section_name == "full_text":
            for idx, token_chunk in enumerate(token_window_chunks(section_text.split())):
                chunks.append({
                    "patient_id": patient_id,
                    "note_id": note_id,
                    "date": str(note_date),
                    "note_type": str(note_type),
                    "section_name": f"fallback_chunk_{idx}",
                    "text": token_chunk,
                })
        else:
            chunks.append({
                "patient_id": patient_id,
                "note_id": note_id,
                "date": str(note_date),
                "note_type": str(note_type),
                "section_name": section_name,
                "text": section_text,
            })
    return chunks
