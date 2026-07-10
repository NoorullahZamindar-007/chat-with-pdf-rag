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


def build_context(documents: list[Any]) -> str:
    """Format retrieved chunks with their page numbers for the LLM."""
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', doc.metadata.get('page_number', 'unknown'))}]\n"
        f"{getattr(doc, 'page_content', getattr(doc, 'text', ''))}"
        for doc in documents
    )


def build_prompt(question: str, documents: list[Any]) -> str:
    """Create the exact RAG prompt sent to the LLM."""
    return f"{SYSTEM_PROMPT}\n\nPDF context:\n{build_context(documents)}\n\nQuestion: {question}"


def _response_text(response) -> str:
    return getattr(response, "content", str(response)).strip()


def format_final_answer(answer: str, documents: list[Any]) -> dict:
    pages = sorted(
        {
            int(page)
            for doc in documents
            for page in [doc.metadata.get("page", doc.metadata.get("page_number"))]
            if str(page).isdigit()
        }
    )
    return {"answer": answer.strip() or FALLBACK_ANSWER, "source_pages": pages}


def answer_question(question: str, documents: list[Any], llm) -> dict:
    """Ask the LLM to answer only from retrieved PDF chunks."""
    if not documents:
        return {"answer": FALLBACK_ANSWER, "source_pages": []}

    response = llm.invoke([Message(content=build_prompt(question, documents))])
    return format_final_answer(_response_text(response), documents)

