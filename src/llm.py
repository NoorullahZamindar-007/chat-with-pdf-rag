import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.config import Settings, get_settings
from src.logger import get_logger


logger = get_logger(__name__)


class MissingApiKeyError(RuntimeError):
    """Raised when a cloud LLM is selected without its required API key."""


@dataclass(frozen=True)
class LLMClient:
    provider: str
    model: str
    base_url: str
    api_key: str = ""
    max_tokens: int = 800
    timeout: int = 60

    def generate(self, prompt: str) -> str:
        if self.provider == "groq":
            return _generate_with_groq(self, prompt)
        if self.provider == "ollama":
            return _generate_with_ollama(self, prompt)
        raise ValueError("Unsupported LLM provider. Use 'groq' or 'ollama'.")


@dataclass(frozen=True)
class _LLMResponse:
    content: str


class _CompatLLM:
    """Tiny LangChain-like wrapper so the existing RAG chain can call .invoke()."""

    def __init__(self, client: LLMClient):
        self.client = client

    def invoke(self, messages: list[Any]) -> _LLMResponse:
        prompt = "\n\n".join(getattr(message, "content", str(message)) for message in messages)
        return _LLMResponse(content=self.client.generate(prompt))


def get_llm_client(settings: Settings | None = None, provider: str | None = None) -> LLMClient:
    """Build the configured LLM client without hardcoding or printing secrets."""
    settings = settings or get_settings()
    selected = (provider or settings.llm_provider).lower()
    max_tokens = getattr(settings, "llm_max_tokens", 800)

    if selected == "groq":
        # Groq is cloud-based and fast. It needs an API key, which must stay secret.
        if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
            raise MissingApiKeyError("GROQ_API_KEY is missing. Add it to .env or choose Ollama.")
        return LLMClient(
            provider="groq",
            model=settings.groq_model,
            base_url="https://api.groq.com/openai/v1/chat/completions",
            api_key=settings.groq_api_key,
            max_tokens=max_tokens,
        )

    if selected == "ollama":
        # Ollama runs locally, so prompts stay private on your machine.
        return LLMClient(
            provider="ollama",
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url.rstrip("/"),
            max_tokens=max_tokens,
        )

    raise ValueError("Unsupported LLM provider. Use 'groq' or 'ollama'.")


def generate_answer(prompt: str, settings: Settings | None = None) -> str:
    """Generate one answer with the configured provider."""
    return get_llm_client(settings).generate(prompt)


def get_llm(settings: Settings, provider: str | None = None):
    """Compatibility helper for the existing Streamlit RAG chain."""
    return _CompatLLM(get_llm_client(settings, provider=provider))


def _post_json(url: str, payload: dict, headers: dict | None, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = _extract_error_message(body) or exc.reason
        if exc.code == 404 or "model" in message.lower():
            raise RuntimeError(f"Model not found or unavailable: {message}") from exc
        raise RuntimeError(f"API request failed: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("API request failed: response was not valid JSON.") from exc


def _extract_error_message(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()
    error = payload.get("error", payload)
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(error)


def _generate_with_groq(client: LLMClient, prompt: str) -> str:
    logger.info("LLM answer started: provider=groq model=%s", client.model)
    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": prompt}],
        # temperature=0 reduces random answers, which is safer for RAG.
        "temperature": 0,
        # Keep output bounded so answers stay concise and API cost stays predictable.
        "max_tokens": client.max_tokens,
    }
    try:
        response = _post_json(
            client.base_url,
            payload,
            {"Authorization": f"Bearer {client.api_key}"},
            client.timeout,
        )
        answer = response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        logger.exception("LLM answer failed")
        raise RuntimeError("API request failed: Groq returned an unexpected response.") from exc
    except Exception:
        logger.exception("LLM answer failed")
        raise
    logger.info("LLM answer completed: provider=groq model=%s", client.model)
    return answer


def _generate_with_ollama(client: LLMClient, prompt: str) -> str:
    logger.info("LLM answer started: provider=ollama model=%s", client.model)
    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": client.max_tokens,
        },
    }
    try:
        response = _post_json(f"{client.base_url}/api/chat", payload, None, client.timeout)
    except RuntimeError as exc:
        logger.exception("LLM answer failed")
        message = str(exc)
        if "connection refused" in message.lower() or "actively refused" in message.lower():
            raise RuntimeError(
                f"Ollama server is not running at {client.base_url}. Start it with: ollama serve"
            ) from exc
        if "model" in message.lower():
            raise RuntimeError(f"Ollama model not found. Pull it with: ollama pull {client.model}") from exc
        raise

    try:
        answer = response["message"]["content"].strip()
    except (KeyError, TypeError) as exc:
        logger.exception("LLM answer failed")
        raise RuntimeError("API request failed: Ollama returned an unexpected response.") from exc
    logger.info("LLM answer completed: provider=ollama model=%s", client.model)
    return answer
