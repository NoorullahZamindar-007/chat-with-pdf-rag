
import pytest

from src import retriever


class FakeDoc:
    def __init__(self, text, metadata):
        self.page_content = text
        self.metadata = metadata


class FakeStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_relevance_scores(self, question, k, filter):
        self.calls.append((question, k, filter))
        return [(FakeDoc("chunk text", {"page_number": 2}), 0.9)]


def install_fake_store(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(retriever, "get_vector_store", lambda name: store)
    return store


def test_retrieve_relevant_chunks_passes_filter_and_score(monkeypatch):
    store = install_fake_store(monkeypatch)

    results = retriever.retrieve_relevant_chunks("What is this?", "pdf_docs", "doc-abc")

    assert results[0]["text"] == "chunk text"
    assert results[0]["page_number"] == 2
    assert results[0]["similarity_score"] == 0.9
    assert store.calls[0][2] == {"document_id": "doc-abc"}


def test_retrieve_relevant_chunks_rejects_blank_question(monkeypatch):
    install_fake_store(monkeypatch)

    with pytest.raises(ValueError, match="question"):
        retriever.retrieve_relevant_chunks("   ", "pdf_docs", "doc-abc")


def test_retrieve_relevant_chunks_rejects_non_positive_top_k(monkeypatch):
    store = install_fake_store(monkeypatch)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve_relevant_chunks("What is this?", "pdf_docs", "doc-abc", top_k=0)

    assert store.calls == []


class _SearchStore:
    """Fake store recording the filter passed to similarity_search."""

    def __init__(self):
        self.calls = []

    def similarity_search(self, question, k, filter=None):
        self.calls.append((question, k, filter))
        return [FakeDoc("chunk text", {"page_number": 2})]


def test_retrieve_documents_passes_document_filter(monkeypatch):
    store = _SearchStore()

    docs = retriever.retrieve_documents(store, "What is this?", document_id="doc-abc")

    assert docs[0].page_content == "chunk text"
    assert store.calls[0][2] == {"document_id": "doc-abc"}


def test_retrieve_documents_without_document_id_searches_whole_store(monkeypatch):
    store = _SearchStore()

    retriever.retrieve_documents(store, "What is this?")

    assert store.calls[0][2] is None
