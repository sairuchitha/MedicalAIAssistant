import hashlib
import pandas as pd


def hash_text(text: str) -> str:
    return hashlib.md5((text or "").strip().encode("utf-8")).hexdigest()


def deduplicate_notes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_hash"] = df["TEXT"].apply(hash_text)
    df = df.drop_duplicates(subset=["SUBJECT_ID", "CATEGORY", "text_hash"])
    return df.drop(columns=["text_hash"])
