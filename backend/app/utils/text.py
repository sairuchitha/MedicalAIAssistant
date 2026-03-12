import re


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def word_count(text: str) -> int:
    return len(normalize_space(text).split())
