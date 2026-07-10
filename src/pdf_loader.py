import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

try:
    import fitz  # PyMuPDF reads PDF pages and extracts selectable text.
except ImportError:  # Keeps validation helpers importable before dependencies are installed.
    fitz = None

from src.text_cleaner import clean_text
from src.utils import generate_document_id


class PDFValidationError(ValueError):
    """Raised when the uploaded file is not a usable PDF."""


@dataclass(frozen=True)
class PDFPage:
    """Small compatibility type used by the chunking tests and examples."""

    page_number: int
    text: str


@dataclass(frozen=True)
class PageDocument:
    """One extracted PDF page plus metadata needed later in the RAG pipeline."""

    document_id: str
    file_name: str
    page_number: int
    text: str
    metadata: dict[str, Any]


def validate_pdf(file_path: str, max_size_mb: int) -> None:
    """Validate a PDF path before extraction."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    max_bytes = max_size_mb * 1024 * 1024
    if path.stat().st_size > max_bytes:
        raise ValueError(f"PDF is too large. Max size is {max_size_mb} MB.")


def validate_pdf_upload(uploaded_file, max_size_mb: int) -> None:
    """Validate Streamlit's UploadedFile before saving it."""
    if uploaded_file is None:
        raise PDFValidationError("Please upload a PDF file.")

    filename = getattr(uploaded_file, "name", "")
    if not filename.lower().endswith(".pdf"):
        raise PDFValidationError("Only PDF files are supported.")

    size = getattr(uploaded_file, "size", None)
    if size is None and hasattr(uploaded_file, "getbuffer"):
        size = len(uploaded_file.getbuffer())

    if size == 0:
        raise PDFValidationError("The uploaded PDF is empty.")

    if size and size > max_size_mb * 1024 * 1024:
        raise PDFValidationError(f"PDF is too large. Max size is {max_size_mb} MB.")


def _read_upload_bytes(uploaded_file: BinaryIO) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "getbuffer"):
        return bytes(uploaded_file.getbuffer())
    return uploaded_file.read()


def save_uploaded_pdf(uploaded_file, upload_dir: Path) -> tuple[Path, str]:
    """Save a PDF using its SHA-256 hash so identical PDFs reuse the same path."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    data = _read_upload_bytes(uploaded_file)
    file_hash = hashlib.sha256(data).hexdigest()
    pdf_path = upload_dir / f"{file_hash}.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(data)
    return pdf_path, file_hash


def extract_pdf_pages(file_path: str, document_id: str | None = None) -> list[PageDocument]:
    """Extract cleaned text from each non-empty PDF page."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install -r requirements.txt")

    path = Path(file_path)
    if document_id is None:
        document_id = generate_document_id(str(path))

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise ValueError("Could not open this PDF. The file may be corrupt.") from exc

    pages: list[PageDocument] = []
    try:
        if doc.page_count == 0:
            raise ValueError("The PDF is empty.")

        for page_index, page in enumerate(doc, start=1):
            # Empty pages are skipped, but a fully empty PDF fails below.
            text = clean_text(page.get_text("text"))
            if not text:
                continue

            metadata = {
                "document_id": document_id,
                "file_name": path.name,
                "page_number": page_index,
                "extraction_method": "pymupdf",
            }
            pages.append(
                PageDocument(
                    document_id=document_id,
                    file_name=path.name,
                    page_number=page_index,
                    text=text,
                    metadata=metadata,
                )
            )
    finally:
        doc.close()

    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned and OCR is needed."
        )

    return pages

