# Chat with PDF using RAG

## Overview

Chat with PDF using RAG is a beginner-friendly AI application that lets a user upload a PDF, ask questions, and receive answers grounded only in the uploaded document. The app extracts PDF text, chunks it, creates embeddings, stores those embeddings in ChromaDB, retrieves relevant chunks, and sends only that context to an LLM.
 
The project is designed as a portfolio-ready reference implementation for Retrieval-Augmented Generation with a clean Python module structure, Streamlit UI, local vector storage, evaluation utilities, Docker support, logging, and tests.

## Problem

Normal LLMs are powerful, but they do not automatically know the contents of a private PDF. If you ask a general model about a document it has never seen, it may guess, hallucinate, or use unrelated outside knowledge.

RAG solves this by retrieving relevant text from the uploaded PDF first, then giving that retrieved context to the LLM. This makes the answer more grounded, easier to verify, and safer for document-question-answering workflows.

## Features

- Upload PDF
- Extract text
- Chunk text
- Generate embeddings
- Store in ChromaDB
- Ask questions
- Answer only from PDF
- Source page citations
- Chat history
- Groq and Ollama support

## Architecture

```text
PDF Upload -> Text Extraction -> Cleaning -> Chunking -> Embeddings -> Vector DB -> Retriever -> LLM -> Answer with Sources
```

Main modules:

- `app.py`: Streamlit frontend and end-to-end workflow
- `src/pdf_loader.py`: PDF validation and PyMuPDF text extraction
- `src/text_cleaner.py`: text normalization
- `src/chunker.py`: LangChain text splitting with page metadata
- `src/embeddings.py`: SentenceTransformers or Ollama embeddings
- `src/vector_store.py`: persistent ChromaDB storage
- `src/retriever.py`: metadata-filtered retrieval
- `src/llm.py`: Groq and Ollama LLM calls
- `src/rag_chain.py`: prompt creation, answer formatting, source pages
- `src/evaluation.py`: basic retrieval and answer evaluation

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- LangChain text splitters
- ChromaDB
- SentenceTransformers or Ollama embeddings
- Groq or Ollama LLMs
- python-dotenv
- pytest
- Ruff
- Black
- Docker

## Setup

Use 64-bit Python 3.10 or newer. Python 3.11 is recommended.

```powershell
cd D:\Projcets\chat-with-pdf-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` before running the app.

For Groq:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_real_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

For local Ollama:

```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

Then set:

```env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBED_MODEL=nomic-embed-text
```

## Environment Variables

The app reads environment variables from `.env` using `python-dotenv`.

Important variables:

- `LLM_PROVIDER`: `groq` or `ollama`
- `GROQ_API_KEY`: Groq API key, required when `LLM_PROVIDER=groq`
- `GROQ_MODEL`: Groq chat model
- `OLLAMA_BASE_URL`: local Ollama server URL
- `OLLAMA_LLM_MODEL`: Ollama chat model
- `LLM_MAX_TOKENS`: output token limit
- `EMBEDDING_PROVIDER`: `sentence-transformers` or `ollama`
- `SENTENCE_TRANSFORMER_MODEL`: HuggingFace embedding model
- `OLLAMA_EMBED_MODEL`: Ollama embedding model
- `CHROMA_PERSIST_DIR`: ChromaDB persistence directory
- `TOP_K`: number of chunks to retrieve
- `MAX_PDF_MB`: upload size limit
- `CHUNK_SIZE`: text chunk size
- `CHUNK_OVERLAP`: overlap between chunks

Never commit `.env`. This repo includes `.gitignore` and `.dockerignore` entries for it.

## How to Run

```powershell
streamlit run app.py
```

Open the local Streamlit URL, upload a text-based PDF, wait for indexing, and ask questions.

Developer shortcuts:

```powershell
make run
make test
make lint
make format
```

Docker:

```powershell
docker build -t chat-with-pdf-rag .
docker run --env-file .env -p 8501:8501 -v ${PWD}/data:/app/data -v ${PWD}/vector_db:/app/vector_db chat-with-pdf-rag
docker compose up --build
```

## How RAG Works

1. The user uploads a PDF.
2. The app validates the file type and size.
3. PyMuPDF extracts selectable text page by page.
4. Text is cleaned so spacing and line breaks are easier to process.
5. The text is split into overlapping chunks.
6. Each chunk is converted into an embedding, which is a numeric representation of meaning.
7. ChromaDB stores embeddings with metadata such as document ID, page number, and chunk ID.
8. When the user asks a question, the app retrieves the most relevant chunks.
9. The LLM receives only the retrieved PDF context plus the question.
10. The answer is returned with source page citations.

If the answer is not present in the retrieved context, the app is instructed to say:

```text
I could not find this information in the uploaded PDF.
```

## Evaluation

Evaluation checks whether retrieval and answering are working before you trust the app.

Create examples in `data/eval/sample_qa.jsonl`:

```json
{"question":"What is the refund policy?","reference_answer":"Refunds are available within 30 days.","expected_pages":[2]}
```

Run:

```python
from src.evaluation import run_basic_evaluation

report = run_basic_evaluation(
    collection_name="pdf_docs",
    document_id="your_document_id_here",
    qa_file="data/eval/sample_qa.jsonl",
)
print(report["metrics"])
print(report["most_wrong_examples"][:3])
```

Results are saved to `data/eval/eval_results.csv`.

`retrieval_hit_rate` means the percentage of questions where at least one expected page appeared in the retrieved sources. A low value usually means the retriever did not send the right context to the LLM.

Inspecting `most_wrong_examples` helps you find practical failure patterns: missed expected pages, answers that say not found, or too many irrelevant source pages.

## Common Errors

- `ModuleNotFoundError`: activate `.venv` and run `pip install -r requirements.txt`.
- `GROQ_API_KEY missing`: set `GROQ_API_KEY` in `.env`, or use `LLM_PROVIDER=ollama`.
- `Ollama server not running`: start Ollama with `ollama serve` and pull the required models.
- `PDF has no extractable text`: the PDF is probably scanned; run OCR first.
- `Chroma database issue`: check `vector_db/` permissions, delete the affected collection, or use the reset button in the app.
- `wrong Python version`: use 64-bit Python 3.10+; Python 3.11 is recommended.
- Slow first run: embedding models may download on first use.
- Poor answers: try better chunking, a better embedding model, or evaluate retrieval hits.

## Future Improvements

- OCR for scanned PDFs
- Hybrid search
- Reranker
- Multi-PDF chat
- User login
- FastAPI backend
- React frontend
- Qdrant vector DB
- Docker production deployment
- Monitoring
- RAGAS evaluation

## Screenshots

Add screenshots here after running the app:

- PDF upload and indexing screen
- Chat answer with source pages
- Retrieved chunks expander
- Evaluation results CSV or notebook view

## Final Checklist

- App runs with `streamlit run app.py`
- PDF upload works
- Indexing works
- Question answering works
- Sources show
- Errors handled
- README complete
- `.env` ignored
- No secret key committed

## LinkedIn Project Description

I built a production-style "Chat with PDF using RAG" project in Python.

The app lets users upload a PDF, indexes the document with embeddings, stores chunks in ChromaDB, and answers questions only from the uploaded PDF with source page citations. It supports Groq for fast cloud inference and Ollama for local/private inference.

Key pieces I implemented:

- Streamlit frontend
- PyMuPDF PDF extraction
- LangChain text chunking
- SentenceTransformers/Ollama embeddings
- ChromaDB vector search
- Groq/Ollama LLM layer
- source page citations
- chat history
- evaluation with retrieval hit rate
- Docker and deployment docs
- tests, logging, and error handling

This project helped me practice the full RAG lifecycle: ingestion, chunking, embeddings, retrieval, generation, evaluation, and deployment readiness.
