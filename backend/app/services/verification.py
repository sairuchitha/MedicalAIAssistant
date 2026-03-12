import re
from typing import Dict, List

ENTITY_PATTERNS = {
    "medications": r"\b(?:mg|mcg|tablet|capsule|insulin|metformin|aspirin)\b",
    "dates": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    "labs": r"\b(?:A1c|HbA1c|WBC|Hgb|creatinine|sodium|potassium)\b",
}


def verify_summary(summary: Dict[str, str], evidence: List[str]) -> List[str]:
    warnings = []
    evidence_blob = " ".join(evidence).lower()
    for section_name, section_text in summary.items():
        normalized = (section_text or "").lower()
        for entity_type, pattern in ENTITY_PATTERNS.items():
            matches = re.findall(pattern, normalized, flags=re.IGNORECASE)
            if matches:
                missing = [m for m in matches if m.lower() not in evidence_blob]
                if missing:
                    warnings.append(f"Unverified {entity_type} mention(s) in {section_name}: {', '.join(sorted(set(missing)))}")
    return warnings
