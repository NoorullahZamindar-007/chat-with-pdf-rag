import importlib.util

import pytest

from src.pdf_loader import extract_pdf_pages, validate_pdf
from src.text_cleaner import clean_text
from src.utils import generate_document_id


def test_validate_pdf_rejects_renamed_non_pdf_on_disk(tmp_path):
    """A renamed non-PDF saved to disk must fail even though its name ends in .pdf.

    The upload-time magic-byte peek is skipped for raw streams, so this
    saved-file check is the last clear checkpoint before extraction.
    """
    file_path = tmp_path / "fake.pdf"
    file_path.write_bytes(b"MZ\x90\x00not a pdf")

    with pytest.raises(ValueError, match="does not look like a PDF"):
        validate_pdf(str(file_path), max_size_mb=10)


def test_validate_pdf_accepts_real_pdf_magic_bytes(tmp_path):
    """A file starting with %PDF- passes the magic-byte check."""
    file_path = tmp_path / "real.pdf"
    file_path.write_bytes(b"%PDF-1.7\n%fake content")

    validate_pdf(str(file_path), max_size_mb=10)  # must not raise


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


def test_extract_pdf_pages_rejects_password_protected_pdf(monkeypatch):
    """A locked PDF must fail with a clear message, not the OCR hint."""
    import src.pdf_loader as pdf_loader

    class FakeDoc:
        needs_pass = True
        page_count = 1

        def close(self):
            pass

    class FakeFitz:
        @staticmethod
        def open(path):
            return FakeDoc()

    monkeypatch.setattr(pdf_loader, "fitz", FakeFitz())
    with pytest.raises(ValueError, match="password-protected"):
        pdf_loader.extract_pdf_pages("locked.pdf", document_id="doc-123")


def test_validate_pdf_upload_rejects_non_pdf_magic_bytes():
    """A renamed non-PDF fails validation even with a .pdf extension."""
    import src.pdf_loader as pdf_loader

    class FakeUpload:
        name = "evil.pdf"
        size = 4

        def getvalue(self):
            return b"MZ\x90\x00not a pdf"

    try:
        pdf_loader.validate_pdf_upload(FakeUpload(), max_size_mb=10)
    except pdf_loader.PDFValidationError as exc:
        assert "does not look like a PDF" in str(exc)
    else:
        raise AssertionError("expected PDFValidationError")

    class RealUpload:
        name = "real.pdf"
        size = 8

        def getvalue(self):
            return b"%PDF-1.7..."

    pdf_loader.validate_pdf_upload(RealUpload(), max_size_mb=10)  # must not raise
