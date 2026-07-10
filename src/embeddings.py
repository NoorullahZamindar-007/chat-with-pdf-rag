from src.config import Settings, get_settings


def get_embedding_function(settings: Settings | None = None, provider: str | None = None):
    """Create the embedding model selected in .env.

    Embeddings turn text chunks into numeric vectors. Similar meanings land near
    each other, which lets Chroma retrieve PDF chunks relevant to a question.
    """
    settings = settings or get_settings()
    selected = (provider or settings.embedding_provider).lower().replace("_", "-")

    if selected in {"sentence-transformers", "huggingface"}:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformer embeddings need langchain-huggingface. "
                "Run: pip install -r requirements.txt"
            ) from exc

        return HuggingFaceEmbeddings(model_name=settings.sentence_transformer_model)

    if selected == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "Ollama embeddings need langchain-ollama. Run: pip install -r requirements.txt"
            ) from exc

        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(
        "Unsupported embedding provider. Use 'sentence-transformers' or 'ollama'."
    )
