"""Persistent Chroma vector-store helpers.

Manual check after installing requirements:

    from src.chunker import chunk_pages
    from src.pdf_loader import extract_pdf_pages
    from src.utils import generate_document_id
    from src.vector_store import add_chunks_to_vector_store
    from src.retriever import retrieve_relevant_chunks

    path = "data/uploads/example.pdf"
    document_id = generate_document_id(path)
    chunks = chunk_pages(extract_pdf_pages(path, document_id))
    add_chunks_to_vector_store(chunks, "pdf_docs")
    print(retrieve_relevant_chunks("What is this about?", "pdf_docs", document_id))
"""

import hashlib
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.embeddings import get_embedding_function
from src.logger import get_logger


logger = get_logger(__name__)

# A vector database stores embedding vectors plus the original chunk metadata so
# we can retrieve matching PDF text later and still know which PDF/page it came from.


def collection_name_for_hash(file_hash: str, embedding_key: str = "") -> str:
    """Create a stable Chroma collection name from a PDF hash."""
    name = f"pdf_{file_hash[:16]}"
    if not embedding_key:
        return name
    suffix = hashlib.sha1(embedding_key.encode("utf-8")).hexdigest()[:8]
    return f"{name}_{suffix}"


def get_vector_store(
    collection_name: str,
    embedding_function: Any | None = None,
    persist_directory: str | Path | None = None,
):
    """Open a local persistent Chroma collection."""
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError("Chroma needs langchain-chroma. Run: pip install -r requirements.txt") from exc

    settings = get_settings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function or get_embedding_function(settings),
        persist_directory=str(persist_directory or settings.chroma_persist_dir),
    )


def _chunk_text(chunk: Any) -> str:
    return getattr(chunk, "text", getattr(chunk, "page_content", ""))


def _chunk_metadata(chunk: Any) -> dict:
    return dict(getattr(chunk, "metadata", {}) or {})


def _chunk_id(chunk: Any, fallback: str) -> str:
    return getattr(chunk, "chunk_id", None) or _chunk_metadata(chunk).get("chunk_id") or fallback


def _existing_ids(vector_store: Any, ids: list[str]) -> set[str]:
    """Ask Chroma which IDs already exist; return empty if the backend cannot check."""
    try:
        existing = vector_store.get(ids=ids)
    except Exception:
        return set()
    return set(existing.get("ids", []) or [])


def _add_texts_in_batches(vector_store: Any, texts: list[str], metadatas: list[dict], ids: list[str], batch_size: int) -> None:
    """Add texts in small batches so local embedding servers are not overloaded."""
    batch_size = max(1, batch_size)
    total = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        vector_store.add_texts(
            texts=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        if hasattr(vector_store, "persist"):
            vector_store.persist()
        logger.info("Embedding batch completed: chunks=%s/%s", end, total)
def vector_store_has_documents(vector_store) -> bool:
    try:
        return vector_store._collection.count() > 0
    except Exception:
        return False


def add_chunks_to_vector_store(chunks, collection_name: str) -> list[str]:
    """Store chunk text, metadata, and IDs in Chroma, skipping existing chunk IDs."""
    if not chunks:
        raise ValueError("No chunks were provided to store.")

    vector_store = get_vector_store(collection_name)
    ids = [_chunk_id(chunk, f"{collection_name}_{index}") for index, chunk in enumerate(chunks)]
    existing = _existing_ids(vector_store, ids)

    texts: list[str] = []
    metadatas: list[dict] = []
    new_ids: list[str] = []
    for chunk, chunk_id in zip(chunks, ids):
        if chunk_id in existing:
            continue
        text = _chunk_text(chunk)
        if not text:
            continue
        metadata = _chunk_metadata(chunk)
        metadata.setdefault("chunk_id", chunk_id)
        texts.append(text)
        metadatas.append(metadata)
        new_ids.append(chunk_id)

    if new_ids:
        logger.info("Embedding started: collection=%s chunks=%s", collection_name, len(new_ids))
        try:
            _add_texts_in_batches(vector_store, texts, metadatas, new_ids, get_settings().embedding_batch_size)
            if hasattr(vector_store, "persist"):
                vector_store.persist()
        except Exception:
            logger.exception("Embedding/vector-store write failed")
            raise
        logger.info("Embedding completed: collection=%s chunks=%s", collection_name, len(new_ids))

    return new_ids


def index_documents(vector_store, documents: list, collection_name: str) -> None:
    """Compatibility wrapper for the existing Streamlit app path."""
    if not documents:
        raise ValueError("No chunks were generated from this PDF.")

    ids = [_chunk_id(doc, f"{collection_name}_{index}") for index, doc in enumerate(documents)]
    existing = _existing_ids(vector_store, ids)
    texts = [_chunk_text(doc) for doc, doc_id in zip(documents, ids) if doc_id not in existing]
    metadatas = [_chunk_metadata(doc) for doc, doc_id in zip(documents, ids) if doc_id not in existing]
    new_ids = [doc_id for doc_id in ids if doc_id not in existing]

    if new_ids:
        logger.info("Embedding started: collection=%s chunks=%s", collection_name, len(new_ids))
        try:
            _add_texts_in_batches(vector_store, texts, metadatas, new_ids, get_settings().embedding_batch_size)
            if hasattr(vector_store, "persist"):
                vector_store.persist()
        except Exception:
            logger.exception("Embedding/vector-store write failed")
            raise
        logger.info("Embedding completed: collection=%s chunks=%s", collection_name, len(new_ids))

