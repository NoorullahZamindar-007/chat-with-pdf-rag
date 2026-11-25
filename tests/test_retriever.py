
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


def test_retrieve_relevant_chunks_rejects_blank_document_id(monkeypatch):
    store = install_fake_store(monkeypatch)

    # A blank id used to filter on an empty id, match nothing and
    # return [] with no error — the generic "not found" answer instead
    # of a clear failure.
    with pytest.raises(ValueError, match="document_id"):
        retriever.retrieve_relevant_chunks("What is this?", "pdf_docs", "   ")

    assert store.calls == []


def test_retrieve_documents_without_document_id_searches_whole_store(monkeypatch):
    store = _SearchStore()

    retriever.retrieve_documents(store, "What is this?")

    assert store.calls[0][2] is None


def test_retrieve_documents_rejects_blank_document_id(monkeypatch):
    store = _SearchStore()

    # A blank id is falsy, so the filter below became None and the
    # helper silently searched the whole collection — chunks from
    # other PDFs leaking into the answer. Only an omitted (None) id
    # means "search everything".
    with pytest.raises(ValueError, match="document_id"):
        retriever.retrieve_documents(store, "What is this?", document_id="  ")

    assert store.calls == []


def test_retrieve_relevant_chunks_tolerates_non_dict_metadata(monkeypatch):
    # A retrieved chunk can carry a non-dict metadata value (a stray
    # string from a JSON retriever, for example); dict() on it used to
    # raise ValueError/TypeError and lose the whole retrieval instead
    # of returning the chunk with empty metadata.
    class StringMetadataStore:
        def similarity_search_with_relevance_scores(self, question, k, filter):
            return [
                (FakeDoc("chunk text", "page 3"), 0.9),
                (FakeDoc("other text", 123), 0.8),
            ]

    monkeypatch.setattr(retriever, "get_vector_store", lambda name: StringMetadataStore())

    results = retriever.retrieve_relevant_chunks("What is this?", "pdf_docs", "doc-abc")

    assert results[0]["text"] == "chunk text"
    assert results[0]["metadata"] == {}
    assert results[0]["page_number"] is None
    assert results[1]["metadata"] == {}
