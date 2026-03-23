import re
from typing import List, Tuple

BLOCK_PATTERNS = [
    r"ignore all previous instructions",
    r"reveal patient name",
    r"print the raw note",
    r"show hidden prompt",
    r"output social security",
    r"leak patient identifier",
    # New patterns added by dev-moni
    r"you are now",
    r"new system prompt",
    r"disregard your instructions",
    r"act as if",
    r"forget everything",
    r"override instructions",
    r"pretend you are",
    r"bypass security",
    r"print all patient",
    r"show me all notes",
    r"dump the database",
    r"reveal all records",
]


def check_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    lowered = (text or "").lower()
    hits = [pattern for pattern in BLOCK_PATTERNS if re.search(pattern, lowered)]
    return (len(hits) > 0, hits)
