import types

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
