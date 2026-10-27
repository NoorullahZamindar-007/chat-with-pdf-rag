from dataclasses import dataclass
from typing import Any


FALLBACK_ANSWER = "I could not find this information in the uploaded PDF."

SYSTEM_PROMPT = f"""
You answer questions using only the supplied PDF context.
Do not use outside knowledge.
Use all relevant context, ignore unrelated chunks, and combine facts when the context supports it.
If the answer is not clearly present in the context, reply exactly:
{FALLBACK_ANSWER}
Give a direct answer first, then cite useful page numbers in parentheses.
""".strip()


@dataclass(frozen=True)
class Message:
    """Small message object compatible with our local LLM wrapper and tests."""

    content: str


def _metadata(doc: Any) -> dict:
    """Return a document's metadata dict, tolerating a None metadata value.

    Some retrievers hand back chunks with metadata=None; without this guard
    a single such chunk crashed prompt building with AttributeError.
    """
    return getattr(doc, "metadata", None) or {}


def build_context(documents: list[Any]) -> str:
    """Format retrieved chunks with their page numbers for the LLM."""
    return "\n\n".join(
        f"[Page {_metadata(doc).get('page', _metadata(doc).get('page_number', 'unknown'))}]\n"
        f"{getattr(doc, 'page_content', getattr(doc, 'text', ''))}"
        for doc in documents
    )


def build_prompt(question: str, documents: list[Any]) -> str:
    """Create the exact RAG prompt sent to the LLM."""
    return f"{SYSTEM_PROMPT}\n\nPDF context:\n{build_context(documents)}\n\nQuestion: {question}"


def _response_text(response) -> str:
    return getattr(response, "content", str(response)).strip()


def _source_page(page: Any) -> int | None:
    """Parse a cited page number, or None when it is not a page.

    Chunks can carry page numbers as ints, digit strings, or integral
    floats (3.0 from a JSON retriever); the old str.isdigit() check
    dropped 3.0 entirely, so a cited chunk vanished from source_pages
    even though build_context cited it as [Page 3.0]. Non-integral
    floats (3.5) and negative values are still not pages.
    """
    if isinstance(page, bool):
        return None
    try:
        value = float(page)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    number = int(value)
    if number != value or number < 0:
        return None
    return number


def format_final_answer(answer: str, documents: list[Any]) -> dict:
    pages = sorted(
        {
            page_number
            for doc in documents
            for page_number in [
                _source_page(
                    _metadata(doc).get("page", _metadata(doc).get("page_number"))
                )
            ]
            if page_number is not None
        }
    )
    return {"answer": answer.strip() or FALLBACK_ANSWER, "source_pages": pages}


def answer_question(question: str, documents: list[Any], llm) -> dict:
    """Ask the LLM to answer only from retrieved PDF chunks."""
    if not question or not question.strip():
        raise ValueError("The question must not be empty.")
    if not documents:
        return {"answer": FALLBACK_ANSWER, "source_pages": []}

    response = llm.invoke([Message(content=build_prompt(question, documents))])
    return format_final_answer(_response_text(response), documents)

