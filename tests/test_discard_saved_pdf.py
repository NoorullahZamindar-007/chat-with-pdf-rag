"""Tests for the saved-upload cleanup helper used when indexing fails."""


from src.pdf_loader import discard_saved_pdf


def test_discard_saved_pdf_removes_file(tmp_path):
    junk = tmp_path / "abc123.pdf"
    junk.write_bytes(b"%PDF-1.4 junk")
    discard_saved_pdf(junk)
    assert not junk.exists()


def test_discard_saved_pdf_missing_file_is_safe(tmp_path):
    discard_saved_pdf(tmp_path / "does-not-exist.pdf")
