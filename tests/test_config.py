import pytest

from src.config import get_settings


def test_get_settings_warns_on_invalid_integer_env(monkeypatch, caplog):
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    with caplog.at_level("WARNING", logger="src.config"):
        settings = get_settings()
    assert settings.chunk_size == 900
    assert "CHUNK_SIZE" in caplog.text
    assert "not-a-number" in caplog.text


def test_get_settings_defaults_are_valid():
    settings = get_settings()
    assert settings.chunk_size == 900
    assert settings.chunk_overlap == 180
    assert settings.max_pdf_mb == 25


def test_get_settings_rejects_overlap_not_smaller_than_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        get_settings()


def test_get_settings_rejects_non_positive_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "0")
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        get_settings()


def test_get_settings_rejects_non_positive_max_pdf_mb(monkeypatch):
    monkeypatch.setenv("MAX_PDF_MB", "-1")
    with pytest.raises(ValueError, match="MAX_PDF_MB"):
        get_settings()


def test_llm_timeout_defaults_to_60():
    assert get_settings().llm_timeout == 60


def test_llm_timeout_env_override(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT", "15")
    assert get_settings().llm_timeout == 15


def test_llm_timeout_rejects_non_positive(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT", "0")
    with pytest.raises(ValueError, match="LLM_TIMEOUT"):
        get_settings()


def test_llm_max_tokens_rejects_non_positive(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "0")
    with pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
        get_settings()


def test_embedding_batch_size_rejects_non_positive(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "-1")
    with pytest.raises(ValueError, match="EMBEDDING_BATCH_SIZE"):
        get_settings()


def test_top_k_rejects_non_positive(monkeypatch):
    monkeypatch.setenv("TOP_K", "0")
    with pytest.raises(ValueError, match="TOP_K"):
        get_settings()


def test_llm_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "grop")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        get_settings()


def test_llm_provider_accepts_case_variant(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Ollama")
    assert get_settings().llm_provider == "Ollama"


def test_string_env_values_are_stripped(monkeypatch):
    # .env values pasted with surrounding spaces (" groq ") used to
    # fail startup validation even though the intent is obvious.
    monkeypatch.setenv("LLM_PROVIDER", " groq ")
    monkeypatch.setenv("EMBEDDING_PROVIDER", " ollama ")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434 ")
    monkeypatch.setenv("GROQ_MODEL", " llama-3.1-8b-instant ")
    monkeypatch.setenv("GROQ_API_KEY", "  gsk_test  ")
    settings = get_settings()
    assert settings.llm_provider == "groq"
    assert settings.embedding_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.groq_model == "llama-3.1-8b-instant"
    assert settings.groq_api_key == "gsk_test"


def test_empty_string_env_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("OLLAMA_BASE_URL", "   ")
    settings = get_settings()
    assert settings.llm_provider == "groq"
    assert settings.ollama_base_url == "http://localhost:11434"


def test_chroma_persist_dir_whitespace_is_stripped(monkeypatch):
    from src.config import BASE_DIR

    monkeypatch.setenv("CHROMA_PERSIST_DIR", " vector_db ")
    assert get_settings().chroma_persist_dir == BASE_DIR / "vector_db"


def test_embedding_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        get_settings()


def test_embedding_provider_accepts_underscore_alias(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    assert get_settings().embedding_provider == "sentence_transformers"


def test_retrieval_k_is_used_when_top_k_is_unset(monkeypatch):
    # RETRIEVAL_K is the legacy name for TOP_K: it still sets the
    # retrieval count, but it is undocumented, so a user following an
    # older guide had no way to know it works.
    monkeypatch.delenv("TOP_K", raising=False)
    monkeypatch.setenv("RETRIEVAL_K", "5")
    assert get_settings().top_k == 5


def test_top_k_wins_over_retrieval_k(monkeypatch):
    monkeypatch.setenv("TOP_K", "10")
    monkeypatch.setenv("RETRIEVAL_K", "5")
    assert get_settings().top_k == 10


def test_ollama_model_alias_is_used_when_ollama_llm_model_is_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_LLM_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    assert get_settings().ollama_llm_model == "llama3.2"


def test_embedding_model_alias_is_used_when_sentence_transformer_model_is_unset(monkeypatch):
    monkeypatch.delenv("SENTENCE_TRANSFORMER_MODEL", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    assert get_settings().sentence_transformer_model == "sentence-transformers/all-mpnet-base-v2"


def test_ollama_embedding_model_alias_is_used_when_ollama_embed_model_is_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    assert get_settings().ollama_embed_model == "mxbai-embed-large"
