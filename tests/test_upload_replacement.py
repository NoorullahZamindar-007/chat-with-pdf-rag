"""Regression tests for replacing the indexed PDF in render_upload_area.

Uploading a second PDF used to orphan the first document's Chroma
collection forever: session state just pointed at the new collection
and nothing ever deleted the old one. These tests prove the old
collection is dropped when a new document replaces it.

app.py imports streamlit at module level, and streamlit is not
installed in the test environment, so it is stubbed here — the same
approach as tests/test_app_reset.py, extended with the Streamlit
surface render_upload_area touches.
"""

import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

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


class _FakeColumn:
    def __init__(self, calls):
        self._calls = calls

    def metric(self, *args, **kwargs):
        self._calls.append(("metric", args))


class _FakeStreamlit(types.ModuleType):
    """Only the Streamlit surface render_upload_area touches."""

    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.calls = []
        self.uploaded_file = None

    def markdown(self, *args, **kwargs):
        self.calls.append(("markdown", args))

    def file_uploader(self, *args, **kwargs):
        return self.uploaded_file

    def info(self, *args, **kwargs):
        self.calls.append(("info", args))

    def warning(self, *args, **kwargs):
        self.calls.append(("warning", args))

    def error(self, *args, **kwargs):
        self.calls.append(("error", args))

    def success(self, *args, **kwargs):
        self.calls.append(("success", args))

    def columns(self, count):
        return tuple(_FakeColumn(self.calls) for _ in range(count))

    @contextmanager
    def spinner(self, *args, **kwargs):
        yield


def _install_stubs():
    if "streamlit" not in sys.modules:
        sys.modules["streamlit"] = _FakeStreamlit()
    else:
        # test_app_reset.py installs a smaller fake first when the full
        # suite runs alphabetically; extend that same object (app is
        # already bound to it) with the surface this module needs,
        # instead of replacing it.
        st = sys.modules["streamlit"]
        if not hasattr(st, "file_uploader"):
            st.uploaded_file = None
            st.markdown = lambda *a, **k: st.calls.append(("markdown", a))
            st.file_uploader = lambda *a, **k: st.uploaded_file
            st.info = lambda *a, **k: st.calls.append(("info", a))
            st.columns = lambda count: tuple(_FakeColumn(st.calls) for _ in range(count))
    if "langchain_text_splitters" not in sys.modules:
        module = types.ModuleType("langchain_text_splitters")

        class _FakeSplitter:
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


class _FakeUpload:
    def __init__(self, name, data=b"fake-pdf-bytes"):
        self.name = name
        self.size = len(data)
        self._data = data

    def getvalue(self):
        return self._data


@pytest.fixture
def indexed_state():
    state = app.st.session_state
    state.clear()
    app.st.calls.clear()
    state.update(
        {
            "document_id": "doc-aaa",
            "collection_name": "pdf_aaa",
            "vector_store": object(),
            "page_count": 3,
            "chunk_count": 10,
            "file_name": "a.pdf",
            "indexed_upload": "sig-a",
            "failed_upload": None,
            "messages": [],
        }
    )
    return state


def _result_for(collection_name, document_id="doc-bbb"):
    return {
        "document_id": document_id,
        "collection_name": collection_name,
        "vector_store": object(),
        "page_count": 2,
        "chunk_count": 5,
        "added_count": 5,
        "file_name": "b.pdf",
    }


def test_replacing_the_pdf_deletes_the_old_collection(monkeypatch, indexed_state):
    deleted = []
    monkeypatch.setattr(app, "delete_collection", lambda name: deleted.append(name))
    monkeypatch.setattr(
        app, "save_and_index_pdf", lambda f, s: _result_for("pdf_bbb")
    )
    monkeypatch.setattr(app, "_upload_signature", lambda f, s: "sig-b")
    app.st.uploaded_file = _FakeUpload("b.pdf")

    app.render_upload_area(None, SimpleNamespace())

    assert deleted == ["pdf_aaa"]
    assert indexed_state["collection_name"] == "pdf_bbb"
    assert indexed_state["document_id"] == "doc-bbb"
    assert indexed_state["indexed_upload"] == "sig-b"


def test_replacing_the_pdf_still_indexes_when_cleanup_fails(monkeypatch, indexed_state):
    # A cleanup failure must never block the new document: the old
    # collection just stays on disk until the next upload tries again.
    def failing_delete(name):
        raise RuntimeError("Chroma is down")

    monkeypatch.setattr(app, "delete_collection", failing_delete)
    monkeypatch.setattr(
        app, "save_and_index_pdf", lambda f, s: _result_for("pdf_bbb")
    )
    monkeypatch.setattr(app, "_upload_signature", lambda f, s: "sig-b")
    app.st.uploaded_file = _FakeUpload("b.pdf")

    app.render_upload_area(None, SimpleNamespace())

    assert indexed_state["collection_name"] == "pdf_bbb"
    assert indexed_state["document_id"] == "doc-bbb"


def test_reuploading_the_same_pdf_deletes_nothing(monkeypatch, indexed_state):
    deleted = []
    monkeypatch.setattr(app, "delete_collection", lambda name: deleted.append(name))
    monkeypatch.setattr(
        app,
        "save_and_index_pdf",
        lambda f, s: (_ for _ in ()).throw(AssertionError("must not re-index")),
    )
    monkeypatch.setattr(app, "_upload_signature", lambda f, s: "sig-a")
    app.st.uploaded_file = _FakeUpload("a.pdf")

    app.render_upload_area(None, SimpleNamespace())

    assert deleted == []
    assert indexed_state["collection_name"] == "pdf_aaa"
