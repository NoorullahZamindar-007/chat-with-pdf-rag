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
    if not question or not question.strip():
        raise ValueError("question must not be blank.")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    # A blank document_id used to filter on an empty id, match nothing
    # and return [] with no error — the user got the generic "not
    # found" answer instead of a clear failure.
    if document_id is not None and not str(document_id).strip():
        raise ValueError("document_id must not be blank.")
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
        # A retrieved chunk can carry a non-dict metadata value (a
        # stray string from a JSON retriever, for example); dict() on
        # it raises ValueError/TypeError and loses the whole retrieval,
        # so treat it as no metadata instead — the same tolerance the
        # chunker, the prompt builder and the chat renderer already
        # apply to non-dict metadata.
        raw_metadata = getattr(doc, "metadata", None) or {}
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
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


def retrieve_documents(
    vector_store: Any,
    question: str,
    k: int | None = None,
    document_id: str | None = None,
) -> list[Any]:
    """Compatibility helper for the Streamlit RAG chain.

    Pass document_id to scope the search to one PDF: without it the
    helper searched the whole collection, so chunks from another
    uploaded PDF could leak into the answer context.
    """
    # The main retrieval path rejects blank questions and non-positive k,
    # but this helper passed them straight through: a blank question
    # wasted an embedding call and failed deep in the vector store, and
    # k=0 was silently swallowed by the `k or settings.top_k` fallback
    # instead of erroring like top_k=0 does above.
    if not question or not question.strip():
        raise ValueError("question must not be blank.")
    if k is not None and k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    # A blank document_id is falsy, so the filter below became None
    # and the helper silently searched the whole collection — chunks
    # from other PDFs leaking into the answer, defeating the scoping
    # this helper promises. Only an omitted (None) id means "search
    # everything".
    if document_id is not None and not str(document_id).strip():
        raise ValueError("document_id must not be blank.")
    settings = get_settings()
    logger.info("Retrieval started")
    try:
        docs = vector_store.similarity_search(
            question,
            k=k or settings.top_k,
            filter={"document_id": document_id} if document_id else None,
        )
    except Exception:
        logger.exception("Retrieval failed")
        raise
    logger.info("Retrieval completed: chunks=%s", len(docs))
    return docs
