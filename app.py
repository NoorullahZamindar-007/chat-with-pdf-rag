import hashlib
from html import escape
from pathlib import Path

import streamlit as st

from src.chunker import chunk_pages
from src.config import get_settings
from src.llm import MissingApiKeyError, get_llm
from src.logger import get_logger
from src.pdf_loader import (
    PDFValidationError,
    discard_saved_pdf,
    extract_pdf_pages,
    save_uploaded_pdf,
    validate_pdf,
    validate_pdf_upload,
)
from src.rag_chain import answer_question
from src.utils import (
    document_collection_name,
    embedding_key,
    ensure_directories,
    generate_document_id,
)
from src.vector_store import (
    add_chunks_to_vector_store,
    delete_collection,
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
        "indexed_upload": None,
        "failed_upload": None,
        # Rotated on every successful reset: the file uploaders below
        # are keyed with this value, so bumping it clears the widget
        # and the reset PDF is not silently re-indexed on the rerun.
        "upload_nonce": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def save_and_index_pdf(uploaded_file, settings) -> dict:
    """Run the full PDF indexing pipeline for one uploaded file."""
    original_name = getattr(uploaded_file, "name", "uploaded.pdf")
    logger.info("PDF uploaded: %s", original_name)
    validate_pdf_upload(uploaded_file, settings.max_pdf_mb)

    # Save first, then validate the actual file path used by the app.
    pdf_path, _, saved_by_this_upload = save_uploaded_pdf(uploaded_file, settings.upload_dir)
    try:
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
    except Exception:
        # The saved file is content-addressed, so a failed index is only
        # junk on disk when this upload created it. If the file already
        # existed (an identical PDF indexed earlier), deleting it would
        # remove the shared copy the earlier index still references.
        if saved_by_this_upload:
            discard_saved_pdf(pdf_path)
        raise

    return {
        "document_id": document_id,
        "collection_name": collection_name,
        "vector_store": vector_store,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "added_count": len(added_ids),
        "file_name": original_name or Path(pdf_path).name,
    }


def reset_current_document() -> bool:
    """Delete only the current document chunks from the current Chroma collection.

    Returns True when the document was reset, False when nothing was
    reset (no document indexed, or the delete failed) — the caller only
    reruns the app on success, because st.rerun() discards the current
    run's output: rerunning after a failure wiped the warning/error
    above before the user could ever read it.
    """
    document_id = st.session_state.get("document_id")
    collection_name = st.session_state.get("collection_name")
    if not document_id or not collection_name:
        st.warning("Upload and index a PDF before resetting its vector data.")
        return False

    try:
        # Use the public delete: it forwards where= to the collection,
        # so this does not depend on the wrapper's private _collection.
        # The store assignment was dropped in the refactor that made
        # this switch, so the button below always failed with a
        # NameError that the generic handler below swallowed.
        store = get_vector_store(collection_name)
        store.delete(where={"document_id": document_id})
    except Exception:
        logger.exception("Vector database reset failed")
        st.error("Vector database reset failed. Check that Chroma is installed and available.")
        return False

    st.session_state.vector_store = None
    st.session_state.collection_name = None
    st.session_state.document_id = None
    st.session_state.page_count = 0
    st.session_state.chunk_count = 0
    st.session_state.file_name = None
    st.session_state.indexed_upload = None
    st.session_state.messages = []
    # Clearing indexed_upload alone does not stop the re-index: both
    # file uploaders keep their values across the rerun (same widget
    # keys), so render_upload_area sees the same file with no indexed
    # signature and runs the whole pipeline again — the reset is undone
    # before the user ever sees the success message below. Rotating the
    # uploader keys clears both widgets, so the reset actually sticks
    # and the user re-uploads to index again. .get() because tests build
    # session state by hand without this key.
    st.session_state["upload_nonce"] = st.session_state.get("upload_nonce", 0) + 1
    st.success("Vector database entries for this document were reset.")
    return True


def render_sidebar(settings):
    with st.sidebar:
        st.markdown("## Control Panel")
        st.caption("Upload, index, and manage the current PDF.")
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            # Keyed with the reset nonce (see reset_current_document):
            # a fixed key would keep the old file after a reset.
            key=f"sidebar_pdf_{st.session_state.get('upload_nonce', 0)}",
        )

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
            # Only rerun on success: st.rerun() discards this run's
            # output, so rerunning after a failed reset wiped the
            # warning/error above before the user could read it.
            if reset_current_document():
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
            # Chunks can arrive as plain dicts from a JSON retriever
            # (build_context already tolerates those), but attribute
            # access on a dict crashed the whole chat render here.
            if isinstance(chunk, dict):
                metadata = chunk.get("metadata") or {}
                text = chunk.get("page_content", chunk.get("text", ""))
            else:
                metadata = getattr(chunk, "metadata", None) or {}
                text = getattr(chunk, "page_content", getattr(chunk, "text", ""))
            # A non-dict metadata value (a stray string from a JSON
            # retriever) has no .get; treat it as no metadata instead
            # of losing the whole render.
            if not isinstance(metadata, dict):
                metadata = {}
            page = metadata.get("page_number", metadata.get("page", "unknown"))
            st.markdown(f"**Chunk {index} | Page {page}**")
            st.write(text)


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
    # A whitespace-only question used to run a (wasted) embedding call
    # for retrieval first and only fail inside answer_question, surfacing
    # as a misleading "Unknown error". Fail fast, before any retrieval.
    if not question or not question.strip():
        raise ValueError("The question must not be empty.")
    vector_store = st.session_state.vector_store
    if vector_store is None:
        raise RuntimeError("No PDF has been indexed yet. Upload a PDF first.")

    logger.info("Retrieval started: document_id=%s", st.session_state.document_id)
    with st.spinner("Searching PDF..."):
        try:
            docs = vector_store.similarity_search(
                question,
                k=settings.top_k,
                filter={"document_id": st.session_state.document_id},
            )
        except Exception as exc:
            # Chroma errors are generally not RuntimeError, so without
            # this they fell through to the generic "Unknown error"
            # branch instead of the clear message below.
            raise RuntimeError(
                "Vector database search failed. Check Chroma and the vector_db folder."
            ) from exc
    logger.info("Retrieval completed: document_id=%s chunks=%s", st.session_state.document_id, len(docs))

    for doc in docs:
        # A retrieved doc can carry metadata=None; normalise it before
        # the page-aliasing below and before the docs reach the renderer.
        # A non-dict metadata value (a stray string from a JSON
        # retriever) has no item assignment, so without this guard it
        # crashed the whole chat turn here instead of rendering like
        # the chunker and the retrieved-chunks renderer already handle.
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        if "page" not in metadata and "page_number" in metadata:
            metadata["page"] = metadata["page_number"]
        doc.metadata = metadata

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


def _upload_signature(uploaded_file, settings):
    """Content hash + embedding settings for the uploaded PDF.

    Streamlit reruns the script on every interaction while the uploader
    value persists, so without this every chat turn would re-run the
    whole indexing pipeline. A settings change still re-indexes because
    the embedding key is part of the signature.
    """
    try:
        data = uploaded_file.getvalue()  # does not consume a Streamlit upload
    except Exception:
        return (getattr(uploaded_file, "name", None), getattr(uploaded_file, "size", None))
    return (hashlib.sha256(data).hexdigest(), embedding_key(settings))


def _drop_superseded_collection(old_collection_name, new_collection_name) -> None:
    """Delete the previous PDF's Chroma collection after a replacement upload.

    Each PDF gets its own collection, so without this every distinct
    PDF ever uploaded leaves a collection on disk forever. A cleanup
    failure must never block the new document: the old collection just
    stays on disk until the next upload tries again. Re-uploading the
    same PDF yields the same (content-addressed) collection name, so
    nothing is deleted in that case.
    """
    if not old_collection_name or old_collection_name == new_collection_name:
        return
    try:
        delete_collection(old_collection_name)
    except Exception:
        logger.exception("Could not delete superseded collection %s", old_collection_name)


def render_upload_area(sidebar_file, settings):
    st.markdown('<div class="section-title">Upload and index</div>', unsafe_allow_html=True)
    main_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        # Keyed with the reset nonce (see reset_current_document):
        # a fixed key would keep the old file after a reset.
        key=f"main_pdf_{st.session_state.get('upload_nonce', 0)}",
    )
    uploaded_file = main_file or sidebar_file

    if uploaded_file is None:
        st.info("Upload a PDF in the sidebar or here to start.")
        return

    signature = _upload_signature(uploaded_file, settings)
    if signature == st.session_state.get("indexed_upload"):
        return
    # A failed index used to be retried on every Streamlit rerun while
    # the uploader value persisted, so one bad PDF re-ran the whole
    # pipeline (and re-showed the error) on every chat interaction.
    # Remember the failed signature and skip it until the file or the
    # embedding settings change, which produces a new signature.
    if signature == st.session_state.get("failed_upload"):
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
        # A new PDF replaces the indexed one: drop the old collection
        # before pointing session state at the new one, or every PDF
        # ever uploaded leaves a collection on disk forever.
        old_collection_name = st.session_state.get("collection_name")
        _drop_superseded_collection(old_collection_name, result["collection_name"])
        # The old chat history asked about the previous PDF: keep it
        # only when the document is the same (a re-index after an
        # embedding-settings change reuses the collection); a different
        # document's Q&A would sit above answers grounded in the new
        # PDF, and the "Chat turns" count would count the old turns.
        if old_collection_name and old_collection_name != result["collection_name"]:
            st.session_state.messages = []
        st.session_state.document_id = result["document_id"]
        st.session_state.collection_name = result["collection_name"]
        st.session_state.vector_store = result["vector_store"]
        st.session_state.page_count = result["page_count"]
        st.session_state.chunk_count = result["chunk_count"]
        st.session_state.file_name = result["file_name"]
        st.session_state.indexed_upload = signature
        st.session_state.failed_upload = None
        show_index_result(result)
        return
    # Reached only when indexing raised: record the failure so the
    # next rerun does not retry this same file and settings.
    st.session_state.failed_upload = signature


def handle_chat(settings) -> None:
    st.markdown('<div class="section-title">Ask questions</div>', unsafe_allow_html=True)
    render_chat_history()

    question = st.chat_input("Ask a question from the uploaded PDF")
    if not question:
        return

    # Do not append a blank question to the chat history or send it to
    # retrieval: it can only fail, and as a history entry it would sit
    # there with an error reply attached.
    if not question.strip():
        st.warning("Please type a question before asking.")
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
