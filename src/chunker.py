from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.text_cleaner import clean_text 


@dataclass(frozen=True)     
class TextChunk: 
    """A small piece of page text ready for embedding and retrieval."""   
  
    chunk_id: str   
    document_id: str   
    file_name: str  
    page_number: int  
    chunk_index: int 
    text: str 
    metadata: dict[str, Any] 
 
 
def _page_value(page: Any, name: str, default: Any = None) -> Any:
    """Read values from a dataclass/object page or from its metadata dict."""
    if isinstance(page, dict):
        value = page.get(name, default)
        metadata = page.get("metadata", {}) or {}
    else:
        value = getattr(page, name, default)
        metadata = getattr(page, "metadata", {}) or {}
    return metadata.get(name, value)


def _page_number(page: Any) -> int:
    """Return a page's page number, tolerating string/float/missing values.

    PDF loaders normally emit ints, but a metadata value such as "3",
    3.0, or a stray non-numeric string used to crash chunking with a
    bare ValueError/TypeError and lose the whole document. Fall back to
    page 0 instead so one odd page number cannot drop the rest.
    """
    raw = _page_value(page, "page_number", 0)
    try:
        return int(float(raw))
    except (TypeError, ValueError, OverflowError):
        # int(float("inf")) raises OverflowError, not ValueError: a
        # non-finite page number must fall back to page 0 like any
        # other unparseable value, not crash chunking.
        return 0


def chunk_pages(
    pages: list,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """Split extracted PDF pages into stable, metadata-rich text chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[TextChunk] = []
    for page in pages:
        text = clean_text(_page_value(page, "text", ""))
        if not text:
            continue

        document_id = _page_value(page, "document_id", "unknown_document")
        file_name = _page_value(page, "file_name", "unknown.pdf")
        page_number = _page_number(page)

        # Chunk indexes restart per page, which keeps chunk IDs readable and stable.
        for chunk_index, raw_chunk in enumerate(splitter.split_text(text)):
            chunk_text = clean_text(raw_chunk)
            if not chunk_text:
                continue

            chunk_id = f"{document_id}_page_{page_number}_chunk_{chunk_index}"
            metadata = {
                "document_id": document_id,
                "file_name": file_name,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    file_name=file_name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    metadata=metadata, 
                )  
            )
   
    if not chunks:
        raise ValueError("No text chunks were created. Check that the PDF text is not empty.")

    return chunks

 
def chunk_pdf_pages(pages: list, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Compatibility wrapper for the existing Streamlit/vector-store code."""
    from langchain_core.documents import Document
 
    return [
        Document(
            page_content=chunk.text,
            metadata={**chunk.metadata, "page": chunk.page_number, "chunk": chunk.chunk_index},
        )
        for chunk in chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ]
