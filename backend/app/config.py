from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "MedMind AI Backend"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    MIMIC_NOTEEVENTS_CSV: str = "data/raw/NOTEEVENTS.csv"
    PROCESSED_DIR: str = "data/processed"
    PATIENT_NOTES_DIR: str = "data/processed/patient_notes"
    SUMMARIES_DIR: str = "data/processed/summaries"
    INDEX_DIR: str = "data/processed/indexes"
    CHROMA_DIR: str = "data/processed/indexes/chroma"

    MAX_PATIENTS: int = 5
    MAX_NOTES_PER_PATIENT: int = 300
    MIN_NOTE_WORDS: int = 100

    BIOCLINICALBERT_MODEL: str = "emilyalsentzer/Bio_ClinicalBERT"
    MEDCPT_ARTICLE_ENCODER: str = "ncbi/MedCPT-Article-Encoder"
    MEDCPT_QUERY_ENCODER: str = "ncbi/MedCPT-Query-Encoder"
    MEDCPT_CROSS_ENCODER: str = "ncbi/MedCPT-Cross-Encoder"

    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # Chunking — 240 tokens fits within MedCPT's 256-token max_length with headroom
    # Stride 100 (vs 50) halves total chunk count, reducing index size and embedding time
    CHUNK_TOKENS: int = 240
    CHUNK_STRIDE: int = 100

    # Retrieval candidate pools — separate from answer top-k (LOOKUP_TOP_K / REASONING_TOP_K)
    # Lookup: small FAISS pool, NO cross-encoder reranking (saves ~1-2s per call)
    # Reasoning: larger pool, full cross-encoder reranking for precision
    LOOKUP_TOP_K_RETRIEVE: int = 4
    REASONING_TOP_K_RETRIEVE: int = 10

    # How many chunks to pass into the LLM prompt after retrieval/reranking
    LOOKUP_TOP_K: int = 3
    REASONING_TOP_K: int = 5

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "vision_ai"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    @property
    def POSTGRES_URL(self) -> str:
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()
