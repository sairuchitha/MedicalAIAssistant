import re
from typing import List, Tuple

BLOCK_PATTERNS = [
    r"ignore all previous instructions",
    r"reveal patient name",
    r"print the raw note",
    r"show hidden prompt",
    r"output social security",
    r"leak patient identifier",
]


def check_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    lowered = (text or "").lower()
    hits = [pattern for pattern in BLOCK_PATTERNS if re.search(pattern, lowered)]
    return (len(hits) > 0, hits)
