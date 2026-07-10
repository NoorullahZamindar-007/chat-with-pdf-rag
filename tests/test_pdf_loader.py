import importlib.util

import pytest

from src.pdf_loader import extract_pdf_pages, validate_pdf
from src.text_cleaner import clean_text
from src.utils import generate_document_id


def test_validate_pdf_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_pdf(str(tmp_path / "missing.pdf"), max_size_mb=1)


def test_validate_pdf_rejects_non_pdf_file(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("not a pdf")

    with pytest.raises(ValueError, match="PDF"):
        validate_pdf(str(file_path), max_size_mb=1)


def test_validate_pdf_rejects_large_file(tmp_path):
    file_path = tmp_path / "large.pdf"
    file_path.write_bytes(b"0" * 2)

    with pytest.raises(ValueError, match="too large"):
        validate_pdf(str(file_path), max_size_mb=0)


def test_generate_document_id_uses_file_hash(tmp_path):
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"same content")

    assert generate_document_id(str(file_path)) == generate_document_id(str(file_path))


def test_clean_text_normalizes_noise_without_removing_punctuation():
    messy = " Hello,\x00   world! \n\n\n This   stays: yes? "

    assert clean_text(messy) == "Hello, world!\n\nThis stays: yes?"


@pytest.mark.skipif(importlib.util.find_spec("fitz") is None, reason="PyMuPDF not installed")
def test_extract_pdf_pages_reads_text_and_metadata(tmp_path):
    import fitz

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello from page one")
    doc.save(pdf_path)
    doc.close()

    pages = extract_pdf_pages(str(pdf_path), document_id="doc-123")

    assert len(pages) == 1
    assert pages[0].document_id == "doc-123"
    assert pages[0].file_name == "sample.pdf"
    assert pages[0].page_number == 1
    assert "Hello from page one" in pages[0].text
    assert pages[0].metadata == {
        "document_id": "doc-123",
        "file_name": "sample.pdf",
        "page_number": 1,
        "extraction_method": "pymupdf",
    }


@pytest.mark.skipif(importlib.util.find_spec("fitz") is None, reason="PyMuPDF not installed")
def test_extract_pdf_pages_rejects_scanned_or_empty_pdf(tmp_path):
    import fitz

    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    with pytest.raises(ValueError, match="OCR"):
        extract_pdf_pages(str(pdf_path), document_id="doc-123")
