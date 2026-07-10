import csv
import json


def test_run_basic_evaluation_saves_csv_and_metrics(tmp_path, monkeypatch):
    from src import evaluation

    qa_file = tmp_path / "sample_qa.jsonl"
    qa_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question": "Where is the refund policy?",
                        "reference_answer": "It is on page two.",
                        "expected_pages": [2],
                    }
                ),
                json.dumps(
                    {
                        "question": "What is the warranty?",
                        "reference_answer": "Warranty details are on page nine.",
                        "expected_pages": [9],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    output_file = tmp_path / "eval_results.csv"
    monkeypatch.setattr(evaluation, "EVAL_RESULTS_FILE", output_file)

    def fake_retrieve_relevant_chunks(question, collection_name, document_id, top_k=5):
        if "refund" in question:
            return [
                {
                    "text": "Refunds are described here.",
                    "page_number": 2,
                    "similarity_score": 0.91,
                    "metadata": {"document_id": document_id, "page_number": 2},
                }
            ]
        return [
            {
                "text": "Unrelated content.",
                "page_number": 3,
                "similarity_score": 0.22,
                "metadata": {"document_id": document_id, "page_number": 3},
            }
        ]

    def fake_answer_question(question, docs, llm):
        if "refund" in question:
            return {"answer": "Refunds are on page two.", "source_pages": [2]}
        return {
            "answer": "I could not find this information in the uploaded PDF.",
            "source_pages": [3],
        }

    monkeypatch.setattr(evaluation, "retrieve_relevant_chunks", fake_retrieve_relevant_chunks)
    monkeypatch.setattr(evaluation, "answer_question", fake_answer_question)
    monkeypatch.setattr(evaluation, "get_llm", lambda settings: object())

    report = evaluation.run_basic_evaluation("pdf_docs", "doc-123", str(qa_file))

    assert report["metrics"]["retrieval_hit_rate"] == 0.5
    assert report["metrics"]["not_found_rate"] == 0.5
    assert report["rows"][0]["retrieval_hit"] is True
    assert report["rows"][1]["retrieval_hit"] is False
    assert report["most_wrong_examples"] == [report["rows"][1]]

    with output_file.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["question"] == "Where is the refund policy?"
    assert rows[0]["expected_pages"] == "[2]"
    assert rows[1]["retrieval_hit"] == "False"
