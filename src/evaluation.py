import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.config import BASE_DIR, get_settings
from src.llm import get_llm
from src.retriever import retrieve_relevant_chunks

try:
    from src.rag_chain import answer_question
except Exception:
    answer_question = None


EVAL_RESULTS_FILE = BASE_DIR / "data" / "eval" / "eval_results.csv"
NOT_FOUND_TEXT = "I could not find this information in the uploaded PDF."


def _load_qa_file(qa_file: str) -> list[dict[str, Any]]:
    path = Path(qa_file)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation QA file does not exist: {qa_file}")

    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "question" not in item or "reference_answer" not in item:
                raise ValueError(f"Missing question/reference_answer on line {line_number}.")
            item.setdefault("expected_pages", [])
            examples.append(item)
    return examples


def _docs_from_retrieved_chunks(chunks: list[dict]) -> list[Any]:
    docs = []
    for chunk in chunks:
        page = chunk.get("page_number")
        metadata = dict(chunk.get("metadata") or {})
        metadata.setdefault("page_number", page)
        metadata.setdefault("page", page)
        docs.append(SimpleNamespace(page_content=chunk.get("text", ""), metadata=metadata))
    return docs


def _unique_ints(values) -> list[int]:
    pages = []
    for value in values or []:
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(pages))


def _notes(retrieval_hit: bool, not_found: bool, expected_pages: list[int], retrieved_pages: list[int]) -> str:
    notes = []
    irrelevant_pages = sorted(set(retrieved_pages) - set(expected_pages))
    if not retrieval_hit:
        notes.append("expected page not retrieved")
    if not_found:
        notes.append("answer says not found")
    if len(irrelevant_pages) >= 3:
        notes.append("too many irrelevant source pages")
    return "; ".join(notes)


def _write_csv(rows: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "generated_answer",
        "reference_answer",
        "expected_pages",
        "retrieved_pages",
        "retrieval_hit",
        "notes",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "expected_pages": json.dumps(row["expected_pages"]),
                    "retrieved_pages": json.dumps(row["retrieved_pages"]),
                }
            )


def _metrics(rows: list[dict]) -> dict[str, float]:
    total = len(rows) or 1
    answer_lengths = [len(row["generated_answer"].split()) for row in rows]
    source_counts = [len(row["retrieved_pages"]) for row in rows]
    return {
        "retrieval_hit_rate": sum(row["retrieval_hit"] for row in rows) / total,
        "average_answer_length": sum(answer_lengths) / total,
        "not_found_rate": sum(NOT_FOUND_TEXT.lower() in row["generated_answer"].lower() for row in rows) / total,
        "average_number_of_source_pages": sum(source_counts) / total,
    }


def run_basic_evaluation(collection_name: str, document_id: str, qa_file: str) -> dict[str, Any]:
    """Run a simple RAG evaluation and save row-level error analysis to CSV."""
    global answer_question
    if answer_question is None:
        try:
            from src.rag_chain import answer_question as imported_answer_question
        except Exception as exc:
            raise RuntimeError("Evaluation needs rag_chain dependencies installed.") from exc
        answer_question = imported_answer_question

    settings = get_settings()
    llm = get_llm(settings)
    rows: list[dict] = []

    for item in _load_qa_file(qa_file):
        expected_pages = _unique_ints(item.get("expected_pages", []))
        retrieved = retrieve_relevant_chunks(
            item["question"],
            collection_name,
            document_id,
            top_k=settings.top_k,
        )
        docs = _docs_from_retrieved_chunks(retrieved)
        result = answer_question(item["question"], docs, llm)
        retrieved_pages = _unique_ints(result.get("source_pages")) or _unique_ints(
            doc.metadata.get("page") for doc in docs
        )
        retrieval_hit = bool(set(expected_pages) & set(retrieved_pages))
        generated_answer = result.get("answer", "")
        not_found = NOT_FOUND_TEXT.lower() in generated_answer.lower()
        notes = _notes(retrieval_hit, not_found, expected_pages, retrieved_pages)

        rows.append(
            {
                "question": item["question"],
                "generated_answer": generated_answer,
                "reference_answer": item["reference_answer"],
                "expected_pages": expected_pages,
                "retrieved_pages": retrieved_pages,
                "retrieval_hit": retrieval_hit,
                "notes": notes,
            }
        )

    _write_csv(rows, EVAL_RESULTS_FILE)
    most_wrong_examples = [row for row in rows if row["notes"]]
    return {
        "metrics": _metrics(rows),
        "rows": rows,
        "most_wrong_examples": most_wrong_examples,
        "output_file": str(EVAL_RESULTS_FILE),
    }
