import json
import urllib.error
from io import BytesIO
from types import SimpleNamespace

import pytest


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return BytesIO(json.dumps(self.payload).encode("utf-8"))

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_config_reads_ollama_llm_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_LLM_MODEL", "llama3.1:8b")

    from src.config import get_settings

    settings = get_settings()

    assert settings.ollama_llm_model == "llama3.1:8b"
    assert settings.ollama_model == "llama3.1:8b"


def test_get_llm_client_rejects_missing_groq_key():
    from src.llm import MissingApiKeyError, get_llm_client

    settings = SimpleNamespace(
        llm_provider="groq",
        groq_api_key="",
        groq_model="llama-3.1-8b-instant",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.1",
        llm_max_tokens=800,
    )

    with pytest.raises(MissingApiKeyError, match="GROQ_API_KEY"):
        get_llm_client(settings)


def test_generate_answer_uses_groq_payload(monkeypatch):
    from src import llm

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"choices": [{"message": {"content": "answer"}}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    settings = SimpleNamespace(
        llm_provider="groq",
        groq_api_key="secret-key",
        groq_model="llama-3.1-8b-instant",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.1",
        llm_max_tokens=800,
    )

    answer = llm.generate_answer("Use only context", settings=settings)

    assert answer == "answer"
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["payload"]["temperature"] == 0
    assert captured["payload"]["max_tokens"] == 800
    assert captured["payload"]["model"] == "llama-3.1-8b-instant"
    assert "secret-key" not in str(captured["payload"])


def test_ollama_connection_error_is_helpful(monkeypatch):
    from src import llm

    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    settings = SimpleNamespace(
        llm_provider="ollama",
        groq_api_key="",
        groq_model="llama-3.1-8b-instant",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.1",
        llm_max_tokens=800,
    )

    with pytest.raises(RuntimeError, match="Ollama server is not running"):
        llm.generate_answer("hello", settings=settings)
