import sys
import types

import pytest

from src.pdf_loader import PageDocument


class FakeRecursiveCharacterTextSplitter:
    def __init__(self, chunk_size, chunk_overlap, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        step = self.chunk_size - self.chunk_overlap
        return [text[index : index + self.chunk_size] for index in range(0, len(text), step)]


def install_fake_splitter(monkeypatch):
    module = types.ModuleType("langchain_text_splitters")
    module.RecursiveCharacterTextSplitter = FakeRecursiveCharacterTextSplitter
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", module)


def page(text, page_number=2):
    return PageDocument(
        document_id="doc-abc",
        file_name="guide.pdf",
        page_number=page_number,
        text=text,
        metadata={"source": "test"},
    )


def test_chunk_pages_creates_chunks(monkeypatch):
    install_fake_splitter(monkeypatch)
    from src.chunker import chunk_pages

    chunks = chunk_pages([page("abcdefghijklmnopqrstuvwxyz")], chunk_size=10, chunk_overlap=2)

    assert len(chunks) == 4
    assert chunks[0].text == "abcdefghij"
    assert chunks[0].chunk_id == "doc-abc_page_2_chunk_0"


def test_chunk_pages_skips_empty_text(monkeypatch):
    install_fake_splitter(monkeypatch)
    from src.chunker import chunk_pages

    chunks = chunk_pages(
        [page("   "), page("real text", page_number=3)],
        chunk_size=20,
        chunk_overlap=0,
    )

    assert len(chunks) == 1
    assert chunks[0].page_number == 3
    assert chunks[0].text == "real text"


def test_chunk_pages_rejects_overlap_greater_than_or_equal_to_size(monkeypatch):
    install_fake_splitter(monkeypatch)
    from src.chunker import chunk_pages

    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_pages([page("text")], chunk_size=100, chunk_overlap=100)


def test_chunk_pages_raises_when_no_chunks_created(monkeypatch):
    install_fake_splitter(monkeypatch)
    from src.chunker import chunk_pages

    with pytest.raises(ValueError, match="No text chunks"):
        chunk_pages([page("   ")], chunk_size=100, chunk_overlap=10)


def test_chunk_pages_adds_required_metadata(monkeypatch):
    install_fake_splitter(monkeypatch)
    from src.chunker import chunk_pages

    chunk = chunk_pages([page("hello world")], chunk_size=50, chunk_overlap=5)[0]

    assert chunk.metadata == {
        "document_id": "doc-abc",
        "file_name": "guide.pdf",
        "page_number": 2,
        "chunk_index": 0,
        "chunk_id": "doc-abc_page_2_chunk_0",
        "chunk_size": 50,
        "chunk_overlap": 5,
    }
