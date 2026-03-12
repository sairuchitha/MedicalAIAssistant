from __future__ import annotations

from typing import Dict, List, Optional

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


_ANALYZER = None
_ANONYMIZER = None


def get_analyzer() -> AnalyzerEngine:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = AnalyzerEngine()
    return _ANALYZER


def get_anonymizer() -> AnonymizerEngine:
    global _ANONYMIZER
    if _ANONYMIZER is None:
        _ANONYMIZER = AnonymizerEngine()
    return _ANONYMIZER


DEFAULT_ENTITY_MAP: Dict[str, str] = {
    "PERSON": "[PATIENT_NAME]",
    "DATE_TIME": "[DATE]",
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "US_SSN": "[SSN]",
    "MEDICAL_LICENSE": "[LICENSE]",
    "URL": "[URL]",
    "LOCATION": "[LOCATION]",
    "IP_ADDRESS": "[IP]",
    "NRP": "[ID]",
}


def mask_phi(text: str, entities: Optional[List[str]] = None, score_threshold: float = 0.35) -> str:
    if not text or not text.strip():
        return ""

    analyzer = get_analyzer()
    anonymizer = get_anonymizer()
    results = analyzer.analyze(text=text, entities=entities, language="en", score_threshold=score_threshold)
    operators = {entity: OperatorConfig("replace", {"new_value": token}) for entity, token in DEFAULT_ENTITY_MAP.items()}
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
    return anonymized.text
