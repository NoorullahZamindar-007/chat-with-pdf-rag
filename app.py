from html import escape
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
from src.utils import ensure_directories, generate_document_id
from src.vector_store import (
    add_chunks_to_vector_store,
    collection_name_for_hash,
    get_vector_store,
)


logger = get_logger(__name__)


def apply_custom_styles() -> None:
    """Add a lightweight SaaS-style theme without extra frontend dependencies."""
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #f6f8fb;
                --surface: #ffffff;
                --surface-soft: #f8fafc;
                --border: #d9e2ec;
                --border-strong: #c6d3e1;
                --text: #172033;
                --muted: #61708a;
                --accent: #0f766e;
                --accent-soft: #e8f7f4;
                --warning-soft: #fff7ed;
            }

            .stApp {
                background: var(--app-bg);
                color: var(--text);
            }

            .block-container {
                max-width: 1180px;
                padding-top: 2rem;
                padding-bottom: 4rem;
            }

            [data-testid="stSidebar"] {
                background: var(--surface);
                border-right: 1px solid var(--border);
            }

            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: var(--text);
            }

            .app-hero {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 28px 30px;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
                margin-bottom: 20px;
            }

            .hero-topline {
                color: var(--accent);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0;
                text-transform: uppercase;
                margin-bottom: 8px;
            }

            .app-hero h1 {
                margin: 0 0 10px 0;
                font-size: clamp(2rem, 5vw, 3.2rem);
                line-height: 1.05;
                letter-spacing: 0;
                color: var(--text);
            }

            .app-hero p {
                margin: 0;
                max-width: 760px;
                color: var(--muted);
                font-size: 1.02rem;
                line-height: 1.65;
            }

            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 18px;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                border: 1px solid var(--border);
                background: var(--surface-soft);
                color: var(--text);
                font-size: 0.82rem;
                font-weight: 650;
                padding: 6px 10px;
                white-space: nowrap;
            }

            .badge.accent {
                border-color: #9bd8d0;
                background: var(--accent-soft);
                color: #0b5f59;
            }

            .doc-strip {
                display: grid;
                grid-template-columns: minmax(0, 1.6fr) repeat(3, minmax(120px, 0.5fr));
                gap: 10px;
                align-items: stretch;
                margin: 16px 0 18px 0;
            }

            .doc-cell {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 14px 16px;
                min-height: 72px;
            }

            .doc-label {
                display: block;
                color: var(--muted);
                font-size: 0.76rem;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: 4px;
            }

            .doc-value {
                color: var(--text);
                font-size: 1.05rem;
                font-weight: 750;
                word-break: break-word;
            }

            .doc-number {
                color: var(--text);
                font-size: 1.55rem;
                line-height: 1.1;
                font-weight: 800;
            }

            .section-title {
                color: var(--text);
                font-size: 1.05rem;
                font-weight: 780;
                margin: 8px 0 8px 0;
            }

            .empty-state {
                background: var(--surface);
                border: 1px dashed var(--border-strong);
                border-radius: 8px;
                padding: 20px;
                color: var(--muted);
                margin-top: 12px;
            }

            .source-wrap {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 10px;
                margin-bottom: 4px;
            }

            .source-pill {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                background: var(--accent-soft);
                border: 1px solid #a7ddd6;
                color: #0b5f59;
                font-size: 0.78rem;
                font-weight: 700;
                padding: 4px 9px;
            }

            div[data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 14px 16px;
                box-shadow: none;
            }

            div[data-testid="stFileUploader"] section {
                background: var(--surface);
                border: 1px dashed var(--border-strong);
                border-radius: 8px;
            }

            .stButton > button {
                border-radius: 8px;
                border: 1px solid var(--border-strong);
                font-weight: 700;
            }

            .stButton > button:hover {
                border-color: var(--accent);
                color: var(--accent);
            }

            @media (max-width: 800px) {
                .doc-strip {
                    grid-template-columns: 1fr;
                }
                .app-hero {
                    padding: 22px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(label: str, value: str, accent: bool = False) -> str:
    tone = " badge accent" if accent else " badge"
    return f'<span class="{tone.strip()}">{escape(label)}: {escape(value)}</span>'


def source_badges(pages: list[int]) -> str:
    if not pages:
        return '<div class="source-wrap"><span class="source-pill">No source pages</span></div>'
    pills = "".join(f'<span class="source-pill">Page {page}</span>' for page in pages)
    return f'<div class="source-wrap">{pills}</div>'


def render_header(settings) -> None:
    badges = "".join(
        [
            badge("LLM", settings.llm_provider, accent=True),
            badge("Embeddings", settings.embedding_provider),
            badge("Top K", str(settings.top_k)),
        ]
    )
    st.markdown(
        f"""
        <section class="app-hero">
            <div class="hero-topline">RAG document assistant</div>
            <h1>Chat with PDF using RAG</h1>
            <p>
                Upload a text-based PDF, index it once, and ask grounded questions.
                Answers are generated only from retrieved PDF context and include page citations.
            </p>
            <div class="badge-row">{badges}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_document_status() -> None:
    if not st.session_state.document_id:
        st.markdown(
            """
            <div class="empty-state">
                Upload a PDF to build a searchable local index. After indexing, the chat will stay ready for repeated questions.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    file_name = escape(st.session_state.file_name or "Indexed PDF")
    st.markdown(
        f"""
        <div class="doc-strip">
            <div class="doc-cell">
                <span class="doc-label">Active document</span>
                <span class="doc-value">{file_name}</span>
            </div>
            <div class="doc-cell">
                <span class="doc-label">Pages</span>
                <span class="doc-number">{st.session_state.page_count}</span>
            </div>
            <div class="doc-cell">
                <span class="doc-label">Chunks</span>
                <span class="doc-number">{st.session_state.chunk_count}</span>
            </div>
            <div class="doc-cell">
                <span class="doc-label">Chat turns</span>
                <span class="doc-number">{len(st.session_state.messages)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def embedding_key(settings) -> str:
    """Keep Chroma collections separate when embedding or chunk settings change."""
    model = (
        settings.ollama_embed_model
        if settings.embedding_provider == "ollama"
        else settings.sentence_transformer_model
    )
    return f"{settings.embedding_provider}:{model}:chunk-{settings.chunk_size}:overlap-{settings.chunk_overlap}"


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
    original_name = getattr(uploaded_file, "name", "uploaded.pdf")
    logger.info("PDF uploaded: %s", original_name)
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
        "file_name": original_name or Path(pdf_path).name,
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
        st.markdown("## Control Panel")
        st.caption("Upload, index, and manage the current PDF.")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="sidebar_pdf")

        st.markdown("## Runtime")
        st.markdown(
            f"""
            <div class="badge-row">
                {badge("LLM", settings.llm_provider, accent=True)}
                {badge("Embeddings", settings.embedding_provider)}
                {badge("Chunk", str(settings.chunk_size))}
                {badge("Overlap", str(settings.chunk_overlap))}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Values come from .env. API keys are never shown here.")

        if st.session_state.document_id:
            st.success("Document is indexed and ready.")
        else:
            st.info("No document indexed yet.")

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


def render_retrieved_chunks(chunks) -> None:
    if not chunks:
        return
    with st.expander("Retrieved chunks"):
        for index, chunk in enumerate(chunks, start=1):
            page = chunk.metadata.get("page_number", chunk.metadata.get("page", "unknown"))
            st.markdown(f"**Chunk {index} | Page {page}**")
            st.write(chunk.page_content)


def render_assistant_message(message: dict) -> None:
    answer = message.get("answer") or message.get("content", "")
    st.markdown(answer)
    if "source_pages" in message:
        st.markdown(source_badges(message["source_pages"]), unsafe_allow_html=True)
    render_retrieved_chunks(message.get("chunks") or [])


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])


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

    return {
        "content": result["answer"],
        "answer": result["answer"],
        "source_pages": result["source_pages"],
        "chunks": docs,
    }


def render_upload_area(sidebar_file, settings):
    st.markdown('<div class="section-title">Upload and index</div>', unsafe_allow_html=True)
    main_file = st.file_uploader("Upload PDF", type=["pdf"], key="main_pdf")
    uploaded_file = main_file or sidebar_file

    if uploaded_file is None:
        st.info("Upload a PDF in the sidebar or here to start.")
        return

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


def handle_chat(settings) -> None:
    st.markdown('<div class="section-title">Ask questions</div>', unsafe_allow_html=True)
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
        assistant_message = {"content": str(exc), "answer": str(exc), "source_pages": [], "chunks": []}
    except RuntimeError as exc:
        logger.exception("Chat runtime error")
        message = str(exc)
        if "Ollama server is not running" in message:
            answer = message
        elif "API request failed" in message or "Model not found" in message:
            answer = message
        else:
            answer = "Vector DB error or LLM error. Check the terminal logs."
        assistant_message = {"content": answer, "answer": answer, "source_pages": [], "chunks": []}
    except Exception:
        logger.exception("Chat failed")
        answer = "Unknown error while answering. Check the terminal logs."
        assistant_message = {"content": answer, "answer": answer, "source_pages": [], "chunks": []}

    st.session_state.messages.append({"role": "assistant", **assistant_message})
    with st.chat_message("assistant"):
        render_assistant_message(assistant_message)


def main() -> None:
    st.set_page_config(page_title="Chat with PDF using RAG", layout="wide")
    apply_custom_styles()

    settings = get_settings()
    ensure_directories(settings.upload_dir, settings.processed_dir, settings.chroma_persist_dir)
    init_session_state()

    render_header(settings)
    sidebar_file = render_sidebar(settings)
    render_document_status()

    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
        render_upload_area(sidebar_file, settings)
    with right:
        handle_chat(settings)


if __name__ == "__main__":
    main()
