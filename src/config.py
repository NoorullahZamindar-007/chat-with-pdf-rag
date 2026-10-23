import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


@dataclass(frozen=True)
class Settings:
    """Environment-backed settings used across the RAG app."""

    base_dir: Path
    upload_dir: Path
    processed_dir: Path
    vector_db_dir: Path
    chroma_persist_dir: Path
    max_pdf_mb: int
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    top_k: int
    llm_provider: str
    llm_max_tokens: int
    llm_timeout: int
    groq_api_key: str
    groq_model: str
    ollama_llm_model: str
    ollama_base_url: str
    embedding_provider: str
    sentence_transformer_model: str
    ollama_embed_model: str
    embedding_batch_size: int

    @property
    def embedding_model(self) -> str:
        """Backward-compatible name used by the existing Streamlit app."""
        return self.sentence_transformer_model

    @property
    def ollama_embedding_model(self) -> str:
        """Backward-compatible name used by the existing Streamlit app."""
        return self.ollama_embed_model

    @property
    def ollama_model(self) -> str:
        """Backward-compatible name for the Ollama chat model."""
        return self.ollama_llm_model


def _validate_settings(chunk_size: int, chunk_overlap: int, max_pdf_mb: int, llm_timeout: int) -> None:
    """Fail fast on inconsistent chunking/upload limits from the environment."""
    if chunk_size <= 0:
        raise ValueError(f"CHUNK_SIZE must be positive, got {chunk_size}.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"CHUNK_OVERLAP must satisfy 0 <= overlap < CHUNK_SIZE, "
            f"got overlap={chunk_overlap} size={chunk_size}."
        )
    if max_pdf_mb <= 0:
        raise ValueError(f"MAX_PDF_MB must be positive, got {max_pdf_mb}.")
    if llm_timeout <= 0:
        raise ValueError(f"LLM_TIMEOUT must be positive, got {llm_timeout}.")


def get_settings() -> Settings:
    """Read .env values each time so tests and local changes are picked up."""
    chroma_dir = _path_env("CHROMA_PERSIST_DIR", BASE_DIR / "vector_db")
    top_k = _int_env("TOP_K", _int_env("RETRIEVAL_K", 8))
    chunk_size = _int_env("CHUNK_SIZE", 900)
    chunk_overlap = _int_env("CHUNK_OVERLAP", 180)
    max_pdf_mb = _int_env("MAX_PDF_MB", 25)
    llm_timeout = _int_env("LLM_TIMEOUT", 60)
    _validate_settings(chunk_size, chunk_overlap, max_pdf_mb, llm_timeout)

    return Settings(
        base_dir=BASE_DIR,
        upload_dir=BASE_DIR / "data" / "uploads",
        processed_dir=BASE_DIR / "data" / "processed",
        vector_db_dir=chroma_dir,
        chroma_persist_dir=chroma_dir,
        max_pdf_mb=max_pdf_mb,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_k=top_k,
        top_k=top_k,
        llm_provider=os.getenv("LLM_PROVIDER", "groq"),
        llm_max_tokens=_int_env("LLM_MAX_TOKENS", 1000),
        llm_timeout=llm_timeout,
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        ollama_llm_model=os.getenv(
            "OLLAMA_LLM_MODEL",
            os.getenv("OLLAMA_MODEL", "llama3.1"),
        ),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"),
        sentence_transformer_model=os.getenv(
            "SENTENCE_TRANSFORMER_MODEL",
            os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        ),
        ollama_embed_model=os.getenv(
            "OLLAMA_EMBED_MODEL",
            os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        ),
        embedding_batch_size=_int_env("EMBEDDING_BATCH_SIZE", 8),
    )

