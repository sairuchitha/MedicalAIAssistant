from pathlib import Path
from typing import List, Optional
import pandas as pd

from app.config import settings
from app.utils.constants import IMPORTANT_NOTE_CATEGORIES

KEEP_COLUMNS = [
    "ROW_ID",
    "SUBJECT_ID",
    "HADM_ID",
    "CHARTDATE",
    "CHARTTIME",
    "STORETIME",
    "CATEGORY",
    "DESCRIPTION",
    "TEXT",
]

CHUNK_SIZE = 50000


def load_noteevents_csv(csv_path: Optional[str] = None) -> pd.DataFrame:

    path = Path(csv_path or settings.MIMIC_NOTEEVENTS_CSV)

    if not path.exists():
        raise FileNotFoundError(f"NOTEEVENTS file not found: {path}")

    chunks = []

    for chunk in pd.read_csv(
        path,
        usecols=lambda c: c in KEEP_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        chunk = chunk.dropna(subset=["SUBJECT_ID", "TEXT"])
        chunk["TEXT"] = chunk["TEXT"].astype(str).str.strip()
        chunk = chunk[chunk["TEXT"] != ""]

        chunk = chunk[chunk["TEXT"].str.split().str.len() >= settings.MIN_NOTE_WORDS]

        if "CATEGORY" in chunk.columns:
            chunk = chunk[chunk["CATEGORY"].isin(IMPORTANT_NOTE_CATEGORIES)]

        chunks.append(chunk)

        # stop early once enough rows are collected
        if len(chunks) > 10:
            break

    df = pd.concat(chunks)

    for col in ["CHARTDATE", "CHARTTIME", "STORETIME"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def select_top_patients(df: pd.DataFrame, max_patients: Optional[int] = None) -> List[int]:

    limit = max_patients or settings.MAX_PATIENTS

    patient_counts = df.groupby("SUBJECT_ID").size().sort_values(ascending=False)

    return patient_counts.head(limit).index.astype(int).tolist()


def limit_notes_per_patient(
    df: pd.DataFrame,
    patient_ids: List[int],
    max_notes_per_patient: Optional[int] = None,
) -> pd.DataFrame:

    note_limit = max_notes_per_patient or settings.MAX_NOTES_PER_PATIENT

    subset = df[df["SUBJECT_ID"].isin(patient_ids)].copy()

    subset = subset.sort_values(["SUBJECT_ID", "CHARTDATE"])

    return subset.groupby("SUBJECT_ID").tail(note_limit)


def load_mvp_patient_notes(
    max_patients: Optional[int] = None,
    max_notes_per_patient: Optional[int] = None,
) -> pd.DataFrame:

    df = load_noteevents_csv()

    patient_ids = select_top_patients(df, max_patients)

    return limit_notes_per_patient(df, patient_ids, max_notes_per_patient)

def get_patient_ids(df: pd.DataFrame) -> List[int]:
    return sorted(df["SUBJECT_ID"].dropna().astype(int).unique().tolist())