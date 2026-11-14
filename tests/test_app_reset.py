"""Regression tests for the Streamlit app's document-reset button.

app.py imports streamlit at module level, and streamlit (like the
langchain packages) is not installed in the test environment, so both
are stubbed here before importing app — the same approach the chunker
tests use for langchain_text_splitters.
"""

import sys
import types
from contextlib import contextmanager

import pytest


class _SessionState(dict):
    """A dict with attribute access, like Streamlit's session state."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item) from None

    def __setattr__(self, item, value):
        self[item] = value


class _FakeStreamlit(types.ModuleType):
    """Only the Streamlit surface reset_current_document() touches."""

    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.calls = []

    def warning(self, *args, **kwargs):
        self.calls.append(("warning", args))

    def error(self, *args, **kwargs):
        self.calls.append(("error", args))

    def success(self, *args, **kwargs):
        self.calls.append(("success", args))

    @contextmanager
    def spinner(self, *args, **kwargs):
        yield


def _install_stubs():
    if "streamlit" not in sys.modules:
        sys.modules["streamlit"] = _FakeStreamlit()
    if "langchain_text_splitters" not in sys.modules:
        module = types.ModuleType("langchain_text_splitters")

        class _FakeSplitter:
            # Same step-slicing behaviour as the fake splitter in
            # tests/test_chunker.py: src.chunker binds whichever
            # splitter class is installed when it is first imported,
            # so this stub must chunk the same way or the chunker
            # tests fail when this module runs first.
            def __init__(self, chunk_size, chunk_overlap, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text):
                step = self.chunk_size - self.chunk_overlap
                return [
                    text[index : index + self.chunk_size]
                    for index in range(0, len(text), step)
                ]

        module.RecursiveCharacterTextSplitter = _FakeSplitter
        sys.modules["langchain_text_splitters"] = module


_install_stubs()

import app  # noqa: E402


@pytest.fixture
def indexed_state():
    state = app.st.session_state
    state.clear()
    app.st.calls.clear()
    state.update(
        {
            "document_id": "doc-123",
            "collection_name": "pdf_abc",
            "vector_store": object(),
            "page_count": 3,
            "chunk_count": 10,
            "file_name": "guide.pdf",
            "indexed_upload": "signature",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    return state


class _FakeStore:
    def __init__(self):
        self.deleted = []

    def delete(self, where=None, **kwargs):
        self.deleted.append(where)


def test_reset_deletes_only_this_document(monkeypatch, indexed_state):
    # Yesterday's public-delete refactor dropped the `store = ...`
    # assignment, so this always failed with a NameError that the
    # generic handler swallowed as "Vector database reset failed".
    store = _FakeStore()
    monkeypatch.setattr(app, "get_vector_store", lambda name: store)

    app.reset_current_document()

    assert store.deleted == [{"document_id": "doc-123"}]
    assert indexed_state["document_id"] is None
    assert indexed_state["collection_name"] is None
    assert indexed_state["vector_store"] is None
    assert indexed_state["page_count"] == 0
    assert indexed_state["chunk_count"] == 0
    assert indexed_state["file_name"] is None
    assert indexed_state["indexed_upload"] is None
    assert indexed_state["messages"] == []
    assert ("success",) in [(name,) for name, _ in app.st.calls]


def test_reset_without_document_warns_and_opens_no_store(monkeypatch):
    app.st.session_state.clear()
    app.st.calls.clear()
    app.st.session_state.update({"document_id": None, "collection_name": None})
    monkeypatch.setattr(
        app,
        "get_vector_store",
        lambda name: (_ for _ in ()).throw(AssertionError("must not open a store")),
    )

    app.reset_current_document()

    assert any(name == "warning" for name, _ in app.st.calls)


def test_reset_failure_keeps_document_state(monkeypatch, indexed_state):
    class _FailingStore:
        def delete(self, where=None, **kwargs):
            raise RuntimeError("Chroma is down")

    monkeypatch.setattr(app, "get_vector_store", lambda name: _FailingStore())

    app.reset_current_document()

    # The document must stay indexed when the delete itself fails;
    # silently clearing the session state would orphan the chunks.
    assert indexed_state["document_id"] == "doc-123"
    assert indexed_state["collection_name"] == "pdf_abc"
    assert any(name == "error" for name, _ in app.st.calls)


def test_drop_superseded_collection_deletes_the_old_one(monkeypatch):
    deleted = []
    monkeypatch.setattr(app, "delete_collection", lambda name: deleted.append(name))

    app._drop_superseded_collection("pdf_aaa", "pdf_bbb")

    assert deleted == ["pdf_aaa"]


def test_drop_superseded_collection_skips_same_or_missing_collection(monkeypatch):
    deleted = []
    monkeypatch.setattr(app, "delete_collection", lambda name: deleted.append(name))

    # Re-uploading the same PDF yields the same collection name; deleting
    # it would wipe the index that is still in use.
    app._drop_superseded_collection("pdf_aaa", "pdf_aaa")
    app._drop_superseded_collection(None, "pdf_bbb")
    app._drop_superseded_collection("", "pdf_bbb")

    assert deleted == []


def test_drop_superseded_collection_failure_does_not_raise(monkeypatch):
    # Cleanup must never block the new document: a backend failure here
    # just leaves the old collection on disk until the next upload.
    def failing_delete(name):
        raise RuntimeError("Chroma is down")

    monkeypatch.setattr(app, "delete_collection", failing_delete)

    app._drop_superseded_collection("pdf_aaa", "pdf_bbb")
