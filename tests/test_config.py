import pytest

from src.config import get_settings


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
