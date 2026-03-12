from __future__ import annotations

import re
from typing import Dict, List, Tuple
import spacy

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"])
    return _NLP


SECTION_PATTERNS = {
    "chief_complaint": [r"^\s*chief complaint\s*[:\-]?\s*$", r"^\s*cc\s*[:\-]?\s*$"],
    "history_present_illness": [r"^\s*history of present illness\s*[:\-]?\s*$", r"^\s*hpi\s*[:\-]?\s*$"],
    "past_medical_history": [r"^\s*past medical history\s*[:\-]?\s*$", r"^\s*pmh\s*[:\-]?\s*$"],
    "medications": [r"^\s*medications\s*[:\-]?\s*$", r"^\s*current medications\s*[:\-]?\s*$", r"^\s*meds\s*[:\-]?\s*$"],
    "allergies": [r"^\s*allergies\s*[:\-]?\s*$"],
    "assessment": [r"^\s*assessment\s*[:\-]?\s*$"],
    "plan": [r"^\s*plan\s*[:\-]?\s*$", r"^\s*assessment\s+and\s+plan\s*[:\-]?\s*$", r"^\s*a/p\s*[:\-]?\s*$"],
    "brief_hospital_course": [r"^\s*brief hospital course\s*[:\-]?\s*$", r"^\s*hospital course\s*[:\-]?\s*$"],
    "impression": [r"^\s*impression\s*[:\-]?\s*$"],
}


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def find_section_headers(text: str) -> List[Tuple[int, str]]:
    get_nlp()
    headers: List[Tuple[int, str]] = []
    lines = text.splitlines()
    offset = 0
    for line in lines:
        normalized = _normalize_line(line)
        for section_name, patterns in SECTION_PATTERNS.items():
            if any(re.match(p, normalized, flags=re.IGNORECASE) for p in patterns):
                headers.append((offset, section_name))
                break
        offset += len(line) + 1
    return headers


def split_sections(text: str) -> Dict[str, str]:
    if not text or not text.strip():
        return {}
    headers = find_section_headers(text)
    if not headers:
        return {"full_text": text.strip()}
    headers = sorted(headers, key=lambda x: x[0])
    sections: Dict[str, str] = {}
    for i, (start_idx, section_name) in enumerate(headers):
        end_idx = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        chunk = text[start_idx:end_idx].strip()
        first_newline = chunk.find("\n")
        body = chunk[first_newline + 1:].strip() if first_newline != -1 else chunk
        if body:
            sections[section_name] = sections.get(section_name, "") + ("\n" if section_name in sections else "") + body
    return sections if sections else {"full_text": text.strip()}
