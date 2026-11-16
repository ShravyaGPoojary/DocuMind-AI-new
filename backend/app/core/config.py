from pathlib import Path
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuMind AI"
    VERSION: str = "1.0.0"
    STAGE: str = "Stage 5 - Intelligent Document Summarization"
    DESCRIPTION: str = (
        "DocuMind AI is an Intelligent Multi-Format Document Understanding Platform "
        "powered by Retrieval-Augmented Generation (RAG) and Intelligent Document Summarization."
    )
    ENVIRONMENT: str = "development"
    API_PREFIX: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: Union[str, List[str]] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Database and Storage Configuration
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data"
    UPLOAD_DIR: Path = Path(__file__).resolve().parent.parent.parent / "uploads"
    CHROMA_PERSIST_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data" / "chroma"
    DATABASE_NAME: str = "documind.db"
    DATABASE_URL: Optional[str] = None

    # ChromaDB & Embedding Model Settings
    CHROMA_COLLECTION_NAME: str = "documind_chunks"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM Provider Configuration
    LLM_PROVIDER: str = "groq"

    # Groq Cloud LLM Settings
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_TIMEOUT: int = 60  # Seconds
    GROQ_REQUEST_TOKEN_BUDGET: int = 3200  # Conservative target request budget (~3,000–3,500 tokens)
    GROQ_TPM_LIMIT: int = 8000  # Groq Free Tier TPM limit
    GROQ_RPM_LIMIT: int = 30  # Groq Free Tier RPM limit
    GROQ_TOKEN_SAFETY_MARGIN: int = 400  # Token safety margin below target budget

    # Stage 5.1 Centralized Summarization Token Budget Settings
    SUMMARIZATION_TARGET_TOKENS: int = 3200  # Target estimated request budget (3,000–3,500 tokens)
    SUMMARIZATION_MAP_OUTPUT_TOKENS: int = 300  # Compact intermediate section digest output cap
    SUMMARIZATION_RESERVED_OUTPUT_TOKENS: int = 1024  # Default reserved completion for reduce synthesis
    SUMMARIZATION_TOKEN_SAFETY_MARGIN: int = 400  # Safety buffer against tokenizer estimation variance

    # Ollama Local LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    OLLAMA_TIMEOUT: int = 180  # Seconds (configured for local CPU inference)

    # Authentication & Security Settings (Stage 8)
    JWT_SECRET_KEY: str = "documind-ai-super-secret-key-change-in-production-stage8-2026"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    GOOGLE_CLIENT_ID: Optional[str] = None

    # Upload and Document Processing Settings
    MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".pptx", ".txt", ".md", ".csv"}
    DEFAULT_CHUNK_SIZE: int = 500  # Words
    DEFAULT_CHUNK_OVERLAP: int = 100  # Words

    @property
    def sync_database_url(self) -> str:
        """Returns the synchronous SQLite database connection URL."""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        db_path = (self.DATA_DIR / self.DATABASE_NAME).resolve().as_posix()
        return f"sqlite:///{db_path}"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]

    model_config = SettingsConfigDict(
        env_file=(
            str(Path(__file__).resolve().parent.parent.parent / ".env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
