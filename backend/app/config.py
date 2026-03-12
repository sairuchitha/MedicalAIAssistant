from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "Vision AI Backend"
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

    CHUNK_TOKENS: int = 200
    CHUNK_STRIDE: int = 50
    RETRIEVAL_TOP_K: int = 10
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
