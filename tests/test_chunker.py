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


def test_chunk_pdf_pages_defaults_match_chunk_pages(monkeypatch):
    install_fake_splitter(monkeypatch)
    # langchain_core is not installed in this env; stub the Document
    # the compatibility wrapper imports.
    from dataclasses import dataclass

    @dataclass
    class FakeDocument:
        page_content: str
        metadata: dict

    core = types.ModuleType("langchain_core")
    documents = types.ModuleType("langchain_core.documents")
    documents.Document = FakeDocument
    core.documents = documents
    monkeypatch.setitem(sys.modules, "langchain_core", core)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", documents)
    from src.chunker import chunk_pages, chunk_pdf_pages

    # chunk_pdf_pages used to default to 1000 while chunk_pages and the
    # config default to 900, so the same PDF chunked differently
    # depending on which entry point a caller used.
    pages = [page("x" * 2500)]
    wrapped = [doc.page_content for doc in chunk_pdf_pages(pages)]
    direct = [chunk.text for chunk in chunk_pages(pages)]
    assert wrapped == direct


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


def test_chunk_pages_rejects_negative_overlap(monkeypatch):
    install_fake_splitter(monkeypatch)
    from src.chunker import chunk_pages

    # Only overlap >= size was checked, so a negative overlap slipped
    # through to the splitter instead of failing with a clear error.
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_pages([page("text")], chunk_size=100, chunk_overlap=-5)


def test_chunk_pages_raises_when_no_chunks_created(monkeypatch):
    install_fake_splitter(monkeypatch)
    from src.chunker import chunk_pages

    with pytest.raises(ValueError, match="No text chunks"):
        chunk_pages([page("   ")], chunk_size=100, chunk_overlap=10)


def test_chunk_pages_tolerates_non_int_page_numbers(monkeypatch):
    install_fake_splitter(monkeypatch)
    from src.chunker import chunk_pages

    # A string page number parses, a float truncates, and a non-numeric
    # value falls back to page 0 instead of dropping the whole document.
    chunks = chunk_pages(
        [
            page("first page", page_number="3"),
            page("second page", page_number=4.7),
            page("third page", page_number="not-a-page"),
        ],
        chunk_size=50,
        chunk_overlap=5,
    )

    assert [chunk.page_number for chunk in chunks] == [3, 4, 0]


def test_chunk_pages_tolerates_non_finite_page_numbers(monkeypatch):
    install_fake_splitter(monkeypatch)
    from src.chunker import chunk_pages

    # int(float("inf")) raises OverflowError, not ValueError, so a
    # non-finite page number used to crash chunking instead of
    # falling back to page 0 like other unparseable values.
    chunks = chunk_pages(
        [
            page("first page", page_number=float("inf")),
            page("second page", page_number=float("nan")),
        ],
        chunk_size=50,
        chunk_overlap=5,
    )

    assert [chunk.page_number for chunk in chunks] == [0, 0]


def test_chunk_pages_tolerates_non_dict_metadata(monkeypatch):
    install_fake_splitter(monkeypatch)
    from src.chunker import chunk_pages

    # A dict page whose metadata is a stray string (not a dict) used to
    # crash chunking on .get with AttributeError; the top-level values
    # are used instead and the page still chunks.
    chunks = chunk_pages(
        [
            {
                "text": "hello world",
                "document_id": "doc-abc",
                "file_name": "guide.pdf",
                "page_number": 2,
                "metadata": "not-a-dict",
            }
        ],
        chunk_size=50,
        chunk_overlap=5,
    )

    assert len(chunks) == 1
    assert chunks[0].page_number == 2
    assert chunks[0].text == "hello world"


def test_chunk_pages_skips_non_string_text(monkeypatch):
    install_fake_splitter(monkeypatch)
    from src.chunker import chunk_pages

    # A page whose text is not a string (an int from a JSON retriever)
    # used to crash clean_text on .replace; it is skipped instead and
    # the other pages still chunk.
    chunks = chunk_pages(
        [
            {"text": 123, "document_id": "d", "file_name": "f.pdf", "page_number": 1},
            page("real text", page_number=3),
        ],
        chunk_size=50,
        chunk_overlap=5,
    )

    assert len(chunks) == 1
    assert chunks[0].page_number == 3


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
