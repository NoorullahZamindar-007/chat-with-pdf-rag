from types import SimpleNamespace

from src.rag_chain import (
    FALLBACK_ANSWER,
    answer_question,
    build_context,
    build_prompt,
    format_final_answer,
)


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.text)


def doc(text, page):
    return SimpleNamespace(page_content=text, metadata={"page": page})


def test_build_context_includes_page_numbers():
    context = build_context([doc("Refunds take 30 days.", 2)])

    assert "[Page 2]" in context
    assert "Refunds take 30 days." in context


def test_build_prompt_keeps_question_and_pdf_context():
    prompt = build_prompt("What is the refund period?", [doc("Refunds take 30 days.", 2)])

    assert "Do not use outside knowledge" in prompt
    assert "PDF context:" in prompt
    assert "Question: What is the refund period?" in prompt
    assert "Refunds take 30 days." in prompt


def test_answer_question_without_context_returns_fallback():
    result = answer_question("What is the policy?", [], FakeLLM("ignored"))

    assert result["answer"] == FALLBACK_ANSWER
    assert result["source_pages"] == []


def test_format_final_answer_adds_unique_sorted_source_pages():
    final = format_final_answer(
        "The refund period is 30 days.",
        [doc("A", 4), doc("B", 2), doc("C", 4)],
    )

    assert final["answer"] == "The refund period is 30 days."
    assert final["source_pages"] == [2, 4]


def test_answer_question_passes_prompt_to_llm_and_keeps_sources():
    llm = FakeLLM("The refund period is 30 days.")

    result = answer_question("What is the refund period?", [doc("Refunds take 30 days.", 2)], llm)

    assert result["answer"] == "The refund period is 30 days."
    assert result["source_pages"] == [2]
    assert "Question: What is the refund period?" in llm.messages[-1].content
