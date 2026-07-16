import sys
import types
from types import SimpleNamespace

import pytest


def test_get_settings_reads_embedding_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("SENTENCE_TRANSFORMER_MODEL", "sentence-model")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text") 
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "custom_chroma")
    monkeypatch.setenv("TOP_K", "7")

    from src.config import get_settings

    settings = get_settings()

    assert settings.embedding_provider == "ollama"
    assert settings.sentence_transformer_model == "sentence-model"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_embed_model == "nomic-embed-text"
    assert str(settings.chroma_persist_dir).endswith("custom_chroma")
    assert settings.top_k == 7


def test_embedding_key_includes_chunk_settings():
    from app import embedding_key

    base = SimpleNamespace(
        embedding_provider="sentence-transformers",
        sentence_transformer_model="sentence-model",
        ollama_embed_model="nomic-embed-text",
        chunk_size=900,
        chunk_overlap=180,
    )
    changed = SimpleNamespace(**{**base.__dict__, "chunk_overlap": 90})

    assert embedding_key(base) == "sentence-transformers:sentence-model:chunk-900:overlap-180"
    assert embedding_key(base) != embedding_key(changed)

def test_get_embedding_function_uses_huggingface(monkeypatch):
    created = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, model_name):
            created["model_name"] = model_name

    module = types.ModuleType("langchain_huggingface")
    module.HuggingFaceEmbeddings = FakeHuggingFaceEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_huggingface", module) 

    from src.embeddings import get_embedding_function

    settings = SimpleNamespace(
        embedding_provider="sentence-transformers",
        sentence_transformer_model="sentence-transformers/all-MiniLM-L6-v2",
        ollama_base_url="http://localhost:11434",
        ollama_embed_model="nomic-embed-text",
    )

    get_embedding_function(settings)

    assert created["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_get_embedding_function_rejects_unknown_provider():
    from src.embeddings import get_embedding_function

    settings = SimpleNamespace(embedding_provider="bad-provider")

    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embedding_function(settings)


def test_add_chunks_to_vector_store_skips_duplicate_ids(monkeypatch):
    from src import vector_store

    class FakeStore:
        def __init__(self):
            self.added = []
            self.persisted = False

        def get(self, ids):
            return {"ids": ["doc_page_1_chunk_0"]}

        def add_texts(self, texts, metadatas, ids):
            self.added.append((texts, metadatas, ids))

        def persist(self):
            self.persisted = True

    store = FakeStore()
    monkeypatch.setattr(vector_store, "get_vector_store", lambda collection_name: store)

    chunks = [
        SimpleNamespace(
            chunk_id="doc_page_1_chunk_0",
            text="duplicate",
            metadata={"document_id": "doc"},
        ),
        SimpleNamespace(
            chunk_id="doc_page_1_chunk_1",
            text="new text",
            metadata={"document_id": "doc"},
        ),
    ]

    added_ids = vector_store.add_chunks_to_vector_store(chunks, "pdf_docs")

    assert added_ids == ["doc_page_1_chunk_1"]
    assert store.added[0][0] == ["new text"]
    assert store.added[0][2] == ["doc_page_1_chunk_1"]
    assert store.persisted is True


def test_retrieve_relevant_chunks_filters_by_document_id(monkeypatch):
    from src import retriever

    class FakeStore:
        def similarity_search_with_relevance_scores(self, question, k, filter):
            self.question = question
            self.k = k
            self.filter = filter
            doc = SimpleNamespace(
                page_content="answer text",
                metadata={"document_id": "doc", "page_number": 4, "chunk_id": "c1"},
            )
            return [(doc, 0.82)]

    store = FakeStore()
    monkeypatch.setattr(retriever, "get_vector_store", lambda collection_name: store)

    results = retriever.retrieve_relevant_chunks(
        question="What is covered?",
        collection_name="pdf_docs",
        document_id="doc",
        top_k=3,
    )

    assert store.filter == {"document_id": "doc"}
    assert store.k == 3 
    assert results == [
        {
            "text": "answer text",
            "page_number": 4, 
            "similarity_score": 0.82,
            "metadata": {"document_id": "doc", "page_number": 4, "chunk_id": "c1"},
        }
    ]
