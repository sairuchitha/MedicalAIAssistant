from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from app.services.llm_client import generate_with_llm


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


def clean_output(text: str, section_name: str = "") -> str:
    if not text:
        return "Not documented"
    bad_patterns = [
        "<|im_sep|>",
        "You are a clinical note assistant.",
        "Task:",
        "Output:",
        "Clinical Evidence:",
        "Rules:",
    ]
    cleaned = text
    for pattern in bad_patterns:
        cleaned = cleaned.replace(pattern, "")
    # Strip section name prefix if the LLM echoes it (e.g. "Chief Complaint: ...")
    if section_name:
        stripped = cleaned.lstrip()
        prefix = f"{section_name}:"
        if stripped.lower().startswith(prefix.lower()):
            cleaned = stripped[len(prefix):].lstrip()
    cleaned = cleaned.strip()
    return cleaned if cleaned else "Not documented"


def _generate_section(section_name: str, evidence: List[str]) -> Tuple[str, str]:
    """Returns (section_name, generated_text). Runs in a thread pool."""
    prompt = build_section_prompt(section_name, evidence)
    raw = generate_with_llm(prompt)
    return section_name, clean_output(raw, section_name)


def build_summary_citations(extracted: List[Dict]) -> List[Dict]:
    """
    Derive unique source citations from the extracted sentences.
    Returns list of {note_id, note_date, note_type} deduplicated and sorted by date.
    """
    seen = set()
    citations = []
    for item in extracted:
        key = (item["note_id"], item["note_date"], item["note_type"])
        if key not in seen:
            seen.add(key)
            citations.append({
                "note_id": item["note_id"],
                "note_date": item["note_date"],
                "note_type": item["note_type"],
            })
    # Sort by date ascending
    citations.sort(key=lambda x: x["note_date"] or "")
    return citations


def generate_structured_summary(extracted: List[Dict]) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Run all 4 section prompts in parallel via ThreadPoolExecutor.
    Returns (summary_dict, citations_list).
    """
    sentences = [item["sentence"] for item in extracted]
    citations = build_summary_citations(extracted)

    result = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_generate_section, section, sentences): section
            for section in SUMMARY_SECTIONS
        }
        for future in as_completed(futures):
            section_name, text = future.result()
            result[section_name] = text

    # Ensure consistent ordering
    ordered = {s: result.get(s, "Not documented") for s in SUMMARY_SECTIONS}
    return ordered, citations
