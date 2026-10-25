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
