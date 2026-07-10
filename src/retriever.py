from typing import Any

from src.config import get_settings
from src.logger import get_logger
from src.vector_store import get_vector_store


logger = get_logger(__name__)

# Metadata filtering matters because one Chroma collection may contain many PDFs.
# Filtering by document_id prevents chunks from another uploaded PDF leaking into
# the current answer context.


def retrieve_relevant_chunks(
    question: str,
    collection_name: str,
    document_id: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve top matching chunks for one PDF document."""
    logger.info("Retrieval started: collection=%s document_id=%s", collection_name, document_id)
    vector_store = get_vector_store(collection_name)
    metadata_filter = {"document_id": document_id}

    try:
        if hasattr(vector_store, "similarity_search_with_relevance_scores"):
            pairs = vector_store.similarity_search_with_relevance_scores(
                question,
                k=top_k,
                filter=metadata_filter,
            )
        elif hasattr(vector_store, "similarity_search_with_score"):
            pairs = vector_store.similarity_search_with_score(
                question,
                k=top_k,
                filter=metadata_filter,
            )
        else:
            docs = vector_store.similarity_search(question, k=top_k, filter=metadata_filter)
            pairs = [(doc, None) for doc in docs]
    except Exception:
        logger.exception("Retrieval failed")
        raise

    results: list[dict] = []
    for doc, score in pairs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        results.append(
            {
                "text": getattr(doc, "page_content", ""),
                "page_number": metadata.get("page_number", metadata.get("page")),
                "similarity_score": score,
                "metadata": metadata,
            }
        )
    logger.info("Retrieval completed: collection=%s document_id=%s chunks=%s", collection_name, document_id, len(results))
    return results


def retrieve_documents(vector_store: Any, question: str, k: int | None = None) -> list[Any]:
    """Compatibility helper for the Streamlit RAG chain."""
    settings = get_settings()
    logger.info("Retrieval started")
    try:
        docs = vector_store.similarity_search(question, k=k or settings.top_k)
    except Exception:
        logger.exception("Retrieval failed")
        raise
    logger.info("Retrieval completed: chunks=%s", len(docs))
    return docs
