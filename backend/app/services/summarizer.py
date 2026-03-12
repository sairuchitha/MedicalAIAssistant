from typing import Dict, List

from app.services.biomistral_client import generate_with_biomistral


SUMMARY_SECTIONS = [
    "Chief Complaint",
    "Active Diagnoses",
    "Current Medications",
    "Recent History and Care Plan",
]


def build_section_prompt(section_name: str, evidence: List[str]) -> str:
    joined = "\n".join(f"- {x}" for x in evidence[:60])

    return f"""
Task: Write only the section named "{section_name}" from the clinical evidence below.

Rules:
- Use only the evidence provided
- Do not copy the instruction text
- Do not output separators like <|im_sep|>
- Do not invent facts
- If information is missing, output exactly: Not documented
- Output only plain clinical text for that section
- Keep it concise, 2 to 5 lines maximum

Clinical Evidence:
{joined}

Output:
""".strip()


def clean_output(text: str) -> str:
    if not text:
        return "Not documented"

    bad_patterns = [
        "<|im_sep|>",
        "You are a clinical note assistant.",
        "Task:",
        "Output:",
        "Clinical Evidence:",
    ]

    cleaned = text
    for pattern in bad_patterns:
        cleaned = cleaned.replace(pattern, "")

    cleaned = cleaned.strip()

    if not cleaned:
        return "Not documented"

    return cleaned


def generate_structured_summary(extracted_sentences: List[str]) -> Dict[str, str]:
    result = {}

    for section in SUMMARY_SECTIONS:
        prompt = build_section_prompt(section, extracted_sentences)
        raw = generate_with_biomistral(prompt)
        result[section] = clean_output(raw)

    return result