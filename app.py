from pathlib import Path

import streamlit as st

from src.chunker import chunk_pages
from src.config import get_settings
from src.llm import MissingApiKeyError, get_llm
from src.logger import get_logger
from src.pdf_loader import (
    PDFValidationError,
    extract_pdf_pages,
    save_uploaded_pdf,
    validate_pdf,
    validate_pdf_upload,
)
from src.rag_chain import answer_question
from src.utils import ensure_directories, format_source_pages, generate_document_id
from src.vector_store import (
    add_chunks_to_vector_store,
    collection_name_for_hash,
    get_vector_store,
)


logger = get_logger(__name__)


def embedding_key(settings) -> str:
    """Keep Chroma collections separate when embedding settings change."""
    model = (
        settings.ollama_embed_model
        if settings.embedding_provider == "ollama"
        else settings.sentence_transformer_model
    )
    return f"{settings.embedding_provider}:{model}"


def document_collection_name(document_id: str, settings) -> str:
    return collection_name_for_hash(document_id, embedding_key(settings))


def init_session_state() -> None:
    """Streamlit reruns the script often, so long-lived UI state lives here."""
    defaults = {
        "messages": [],
        "document_id": None,
        "collection_name": None,
        "vector_store": None,
        "page_count": 0,
        "chunk_count": 0,
        "file_name": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def save_and_index_pdf(uploaded_file, settings) -> dict:
    """Run the full PDF indexing pipeline for one uploaded file."""
    logger.info("PDF uploaded: %s", getattr(uploaded_file, "name", "unknown"))
    validate_pdf_upload(uploaded_file, settings.max_pdf_mb)

    # Save first, then validate the actual file path used by the app.
    pdf_path, _ = save_uploaded_pdf(uploaded_file, settings.upload_dir)
    validate_pdf(str(pdf_path), settings.max_pdf_mb)
    logger.info("PDF validated: %s", pdf_path)

    document_id = generate_document_id(str(pdf_path))
    collection_name = document_collection_name(document_id, settings)

    # PDF text is extracted and cleaned in the loader, then split into RAG chunks.
    pages = extract_pdf_pages(str(pdf_path), document_id=document_id)
    logger.info("PDF pages extracted: document_id=%s pages=%s", document_id, len(pages))
    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    logger.info("Text chunks created: document_id=%s chunks=%s", document_id, len(chunks))

    # Chroma creates embeddings through the configured embedding function.
    logger.info("Embedding started: document_id=%s collection=%s", document_id, collection_name)
    added_ids = add_chunks_to_vector_store(chunks, collection_name)
    vector_store = get_vector_store(collection_name)
    logger.info(
        "Embedding completed: document_id=%s total_chunks=%s new_chunks=%s",
        document_id,
        len(chunks),
        len(added_ids),
    )

    return {
        "document_id": document_id,
        "collection_name": collection_name,
        "vector_store": vector_store,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "added_count": len(added_ids),
        "file_name": Path(pdf_path).name,
    }


def reset_current_document() -> None:
    """Delete only the current document chunks from the current Chroma collection."""
    document_id = st.session_state.get("document_id")
    collection_name = st.session_state.get("collection_name")
    if not document_id or not collection_name:
        st.warning("Upload and index a PDF before resetting its vector data.")
        return

    try:
        store = get_vector_store(collection_name)
        store._collection.delete(where={"document_id": document_id})
    except Exception:
        logger.exception("Vector database reset failed")
        st.error("Vector database reset failed. Check that Chroma is installed and available.")
        return

    st.session_state.vector_store = None
    st.session_state.collection_name = None
    st.session_state.document_id = None
    st.session_state.page_count = 0
    st.session_state.chunk_count = 0
    st.session_state.file_name = None
    st.session_state.messages = []
    st.success("Vector database entries for this document were reset.")


def render_sidebar(settings):
    with st.sidebar:
        st.header("Document")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="sidebar_pdf")

        st.header("Configuration")
        st.write(f"LLM provider: `{settings.llm_provider}`")
        st.write(f"Embedding provider: `{settings.embedding_provider}`")
        st.write(f"Chunk size: `{settings.chunk_size}`")
        st.write(f"Chunk overlap: `{settings.chunk_overlap}`")
        st.caption("Values come from .env. API keys are never shown here.")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("Reset vector database for this document", use_container_width=True):
            reset_current_document()
            st.rerun()

    return uploaded_file


def show_index_result(result: dict) -> None:
    st.success("PDF indexed successfully")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pages", result["page_count"])
    col2.metric("Chunks", result["chunk_count"])
    col3.metric("New chunks", result["added_count"])


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            chunks = message.get("chunks") or []
            if chunks:
                with st.expander("Retrieved chunks"):
                    for index, chunk in enumerate(chunks, start=1):
                        page = chunk.metadata.get("page_number", chunk.metadata.get("page", "unknown"))
                        st.markdown(f"**Chunk {index} - page {page}**")
                        st.write(chunk.page_content)


def answer_user_question(question: str, settings) -> dict:
    vector_store = st.session_state.vector_store
    if vector_store is None:
        raise RuntimeError("No PDF has been indexed yet. Upload a PDF first.")

    logger.info("Retrieval started: document_id=%s", st.session_state.document_id)
    with st.spinner("Searching PDF..."):
        docs = vector_store.similarity_search(
            question,
            k=settings.top_k,
            filter={"document_id": st.session_state.document_id},
        )
    logger.info("Retrieval completed: document_id=%s chunks=%s", st.session_state.document_id, len(docs))

    for doc in docs:
        if "page" not in doc.metadata and "page_number" in doc.metadata:
            doc.metadata["page"] = doc.metadata["page_number"]

    logger.info("LLM answer started: document_id=%s", st.session_state.document_id)
    with st.spinner("Generating answer..."):
        llm = get_llm(settings)
        result = answer_question(question, docs, llm)
    logger.info("LLM answer completed: document_id=%s", st.session_state.document_id)

    sources = format_source_pages(result["source_pages"])
    return {
        "content": f"{result['answer']}\n\n**Sources:** {sources}",
        "chunks": docs,
    }


def main() -> None:
    st.set_page_config(page_title="Chat with PDF using RAG", page_icon="PDF")
    st.title("Chat with PDF using RAG")

    settings = get_settings()
    ensure_directories(settings.upload_dir, settings.processed_dir, settings.chroma_persist_dir)
    init_session_state()

    sidebar_file = render_sidebar(settings)

    st.markdown(
        "Upload a text-based PDF, wait for indexing, then ask questions. "
        "Answers are generated only from retrieved PDF chunks and include source pages."
    )

    main_file = st.file_uploader("Upload PDF", type=["pdf"], key="main_pdf")
    uploaded_file = main_file or sidebar_file

    if uploaded_file is None:
        st.info("No PDF uploaded yet. Upload a PDF in the sidebar or above to start.")
    else:
        try:
            with st.spinner("Indexing PDF..."):
                result = save_and_index_pdf(uploaded_file, settings)
        except PDFValidationError as exc:
            logger.exception("PDF upload validation failed")
            st.error(str(exc))
        except ValueError as exc:
            logger.exception("PDF indexing validation failed")
            message = str(exc)
            if "OCR" in message or "extractable text" in message:
                st.error("This PDF has no readable text. It may be scanned and needs OCR first.")
            else:
                st.error(message)
        except MissingApiKeyError as exc:
            logger.exception("LLM configuration failed")
            st.error(str(exc))
        except RuntimeError as exc:
            logger.exception("PDF indexing runtime error")
            message = str(exc)
            if "Ollama server is not running" in message:
                st.error(message)
            elif "Chroma" in message or "vector" in message.lower():
                st.error("Vector DB error. Check Chroma installation and the vector_db folder.")
            else:
                st.error(message)
        except Exception:
            logger.exception("PDF indexing failed")
            st.error("Unknown error while indexing the PDF. Check the terminal logs.")
        else:
            st.session_state.document_id = result["document_id"]
            st.session_state.collection_name = result["collection_name"]
            st.session_state.vector_store = result["vector_store"]
            st.session_state.page_count = result["page_count"]
            st.session_state.chunk_count = result["chunk_count"]
            st.session_state.file_name = result["file_name"]
            show_index_result(result)

    render_chat_history()

    question = st.chat_input("Ask a question from the uploaded PDF")
    if not question:
        return

    if st.session_state.vector_store is None:
        st.error("No PDF uploaded or indexed yet. Upload a PDF before asking questions.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        assistant_message = answer_user_question(question, settings)
    except MissingApiKeyError as exc:
        logger.exception("LLM configuration failed")
        assistant_message = {"content": str(exc), "chunks": []}
    except RuntimeError as exc:
        logger.exception("Chat runtime error")
        message = str(exc)
        if "Ollama server is not running" in message:
            assistant_message = {"content": message, "chunks": []}
        elif "API request failed" in message or "Model not found" in message:
            assistant_message = {"content": message, "chunks": []}
        else:
            assistant_message = {"content": "Vector DB error or LLM error. Check the terminal logs.", "chunks": []}
    except Exception:
        logger.exception("Chat failed")
        assistant_message = {"content": "Unknown error while answering. Check the terminal logs.", "chunks": []}

    st.session_state.messages.append({"role": "assistant", **assistant_message})
    with st.chat_message("assistant"):
        st.markdown(assistant_message["content"])
        if assistant_message["chunks"]:
            with st.expander("Retrieved chunks"):
                for index, chunk in enumerate(assistant_message["chunks"], start=1):
                    page = chunk.metadata.get("page_number", chunk.metadata.get("page", "unknown"))
                    st.markdown(f"**Chunk {index} - page {page}**")
                    st.write(chunk.page_content)


if __name__ == "__main__":
    main()
