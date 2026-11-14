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
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["accept"] == "application/json"
    assert headers["authorization"] == "Bearer secret-key"
    assert headers["user-agent"] == "chat-with-pdf-rag/1.0"
    assert captured["payload"]["temperature"] == 0
    assert captured["payload"]["max_tokens"] == 800
    assert captured["payload"]["model"] == "llama-3.1-8b-instant"
    assert "secret-key" not in str(captured["payload"])


def test_null_llm_content_is_an_unexpected_response_error(monkeypatch):
    """A null content field must not escape as a bare AttributeError.

    Both providers call .strip() on the reply content; a model reply
    with content=None (filtered/empty) used to leak AttributeError to
    the user instead of the clean "unexpected response" error.
    """
    from src import llm

    def fake_groq_urlopen(request, timeout):
        return FakeHTTPResponse({"choices": [{"message": {"content": None}}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_groq_urlopen)

    groq_settings = SimpleNamespace(
        llm_provider="groq",
        groq_api_key="secret-key",
        groq_model="llama-3.1-8b-instant",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.1",
        llm_max_tokens=800,
    )

    with pytest.raises(RuntimeError, match="unexpected response"):
        llm.generate_answer("hello", settings=groq_settings)

    def fake_ollama_urlopen(request, timeout):
        return FakeHTTPResponse({"message": {"content": None}})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_ollama_urlopen)

    ollama_settings = SimpleNamespace(
        llm_provider="ollama",
        groq_api_key="",
        groq_model="llama-3.1-8b-instant",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.1",
        llm_max_tokens=800,
    )

    with pytest.raises(RuntimeError, match="unexpected response"):
        llm.generate_answer("hello", settings=ollama_settings)


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


def test_truncated_llm_response_is_an_api_error(monkeypatch):
    """A response cut off mid-read must not escape as a bare traceback.

    IncompleteRead (and a reset connection) is not a URLError, so it
    used to leak out of _post_json as a raw http.client/OSError error
    and surface in the app as an "Unknown error" instead of the real
    cause.
    """
    import http.client

    from src import llm

    def fake_urlopen(request, timeout):
        raise http.client.IncompleteRead(b"partial", 100)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    settings = SimpleNamespace(
        llm_provider="groq",
        groq_api_key="secret-key",
        groq_model="llama-3.1-8b-instant",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.1",
        llm_max_tokens=800,
    )

    with pytest.raises(RuntimeError, match="API request failed"):
        llm.generate_answer("hello", settings=settings)


def test_non_utf8_llm_response_is_an_api_error(monkeypatch):
    """A non-UTF-8 response body must not escape as a bare traceback.

    The body is decoded before json.loads is reached, so a
    UnicodeDecodeError used to leak out of _post_json raw and surface
    in the app as an "Unknown error" instead of the real cause.
    """
    from src import llm

    class NonUtf8Response:
        def __enter__(self):
            return BytesIO(b"\xff\xfe not json \x80\x81")

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_urlopen(request, timeout):
        return NonUtf8Response()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    settings = SimpleNamespace(
        llm_provider="groq",
        groq_api_key="secret-key",
        groq_model="llama-3.1-8b-instant",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.1",
        llm_max_tokens=800,
    )

    with pytest.raises(RuntimeError, match="not valid UTF-8/JSON"):
        llm.generate_answer("hello", settings=settings)


def test_groq_rate_limit_is_reported_as_a_clear_error(monkeypatch):
    """A Groq 429 must tell the user to wait and retry, not dump a raw body.

    Groq's free tier rate-limits frequently; the generic
    "API request failed: <raw body>" message gave no hint to wait
    before retrying.
    """
    from src import llm

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            BytesIO(b'{"error": {"message": "Rate limit reached"}}'),
        )

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    settings = SimpleNamespace(
        llm_provider="groq",
        groq_api_key="test-key",
        groq_model="llama-3.1-8b-instant",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.1",
        llm_max_tokens=800,
    )

    with pytest.raises(RuntimeError, match="rate limit"):
        llm.generate_answer("hello", settings=settings)


def test_groq_rate_limit_naming_the_model_is_not_model_not_found(monkeypatch):
    """A Groq 429 names the model in its body; it must still read as a rate limit.

    Groq's real 429 body is "Rate limit reached for model
    llama-3.1-8b-instant in organization org_... on tokens per day
    (TPD): ...". The "model" substring check used to run before the
    429 branch, so this was reported as "Model not found" and told
    the user to check the model instead of waiting and retrying. A
    plain "rate limit" match cannot tell the two apart (the wrong
    message contains the body too), so assert the negative as well.
    """
    from src import llm

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            BytesIO(
                b'{"error": {"message": "Rate limit reached for model llama-3.1-8b-instant '
                b'in organization org_123 on tokens per day (TPD): Limit 100000, Used 100123"}}'
            ),
        )

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    settings = SimpleNamespace(
        llm_provider="groq",
        groq_api_key="test-key",
        groq_model="llama-3.1-8b-instant",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.1",
        llm_max_tokens=800,
    )

    with pytest.raises(RuntimeError) as excinfo:
        llm.generate_answer("hello", settings=settings)
    assert "rate limit" in str(excinfo.value).lower()
    assert "Model not found" not in str(excinfo.value)


def test_groq_invalid_api_key_is_reported_as_a_clear_error(monkeypatch):
    """A Groq 401 must name the fix (check the key), not dump a raw body.

    A wrong or revoked GROQ_API_KEY used to fall through to the generic
    "API request failed: <raw body>" branch, like the 403/1010 and 429
    cases did before they got their own actionable messages.
    """
    from src import llm

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            BytesIO(b'{"error": {"message": "Invalid API key provided"}}'),
        )

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    settings = SimpleNamespace(
        llm_provider="groq",
        groq_api_key="bad-key",
        groq_model="llama-3.1-8b-instant",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.1",
        llm_max_tokens=800,
    )

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.generate_answer("hello", settings=settings)


def test_generate_answer_uses_configured_llm_timeout(monkeypatch):
    from types import SimpleNamespace

    from src import llm

    captured = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        return FakeHTTPResponse({"choices": [{"message": {"content": "answer"}}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    settings = SimpleNamespace(
        llm_provider="groq",
        groq_api_key="test-key",
        groq_model="llama-3.1-8b-instant",
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.1",
        llm_max_tokens=800,
        llm_timeout=7,
    )

    assert llm.generate_answer("hello", settings=settings) == "answer"
    assert captured["timeout"] == 7
