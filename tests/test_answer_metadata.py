"""Regression tests for answer_user_question() metadata normalisation.

app.py imports streamlit at module level, and streamlit (like the
langchain packages) is not installed in the test environment, so both
are stubbed here before importing app — the same approach the chunker
tests use for langchain_text_splitters and test_app_reset.py uses for
streamlit. The fake Streamlit below is identical to the one in
test_app_reset.py so either file may run first.
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


class _FakeStreamlit(types.ModuleType):
    """Only the Streamlit surface answer_user_question() touches."""

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

    @contextmanager
    def expander(self, *args, **kwargs):
        yield

    def markdown(self, *args, **kwargs):
        self.calls.append(("markdown", args))

    def write(self, *args, **kwargs):
        self.calls.append(("write", args))


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


class _FakeStore:
    """Vector store returning prebuilt docs from similarity_search."""

    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, question, k=None, filter=None):
        return self.docs


@pytest.fixture
def qa_state(monkeypatch):
    state = app.st.session_state
    state.clear()
    app.st.calls.clear()
    state.update(
        {
            "document_id": "doc-123",
            "collection_name": "pdf_abc",
            "vector_store": None,
            "page_count": 1,
            "chunk_count": 1,
            "file_name": "guide.pdf",
            "messages": [],
        }
    )
    # The LLM is stubbed out: the metadata normalisation under test
    # runs before the LLM call, so the fake only has to return an answer.
    seen = {}

    def _fake_answer(question, docs, llm):
        seen["docs"] = docs
        return {"answer": "an answer", "source_pages": [1]}

    monkeypatch.setattr(app, "answer_question", _fake_answer)
    monkeypatch.setattr(app, "get_llm", lambda settings: object())
    return state, seen


def test_non_dict_metadata_is_normalised(qa_state):
    # A doc whose metadata is a stray string (from a JSON retriever)
    # used to crash answer_user_question on `metadata["page"] = ...`
    # with TypeError and lose the whole chat turn.
    state, seen = qa_state
    doc = SimpleNamespace(page_content="some text", metadata="not-a-dict")
    state["vector_store"] = _FakeStore([doc])
    result = app.answer_user_question("What is this?", SimpleNamespace(top_k=5))
    assert result["answer"] == "an answer"
    assert seen["docs"][0].metadata == {}


def test_none_metadata_is_normalised_and_page_aliased(qa_state):
    state, seen = qa_state
    doc = SimpleNamespace(page_content="some text", metadata=None)
    state["vector_store"] = _FakeStore([doc])
    result = app.answer_user_question("What is this?", SimpleNamespace(top_k=5))
    assert result["answer"] == "an answer"
    assert seen["docs"][0].metadata == {}


def test_dict_metadata_page_number_is_aliased_to_page(qa_state):
    state, seen = qa_state
    doc = SimpleNamespace(
        page_content="some text", metadata={"page_number": 4}
    )
    state["vector_store"] = _FakeStore([doc])
    app.answer_user_question("What is this?", SimpleNamespace(top_k=5))
    assert seen["docs"][0].metadata == {"page_number": 4, "page": 4}
