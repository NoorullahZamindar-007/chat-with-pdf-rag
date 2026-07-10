import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_document_id(file_path: str) -> str:
    """Generate a stable ID from the file bytes, not from the file name."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_directories(*paths: Path | str) -> None:
    """Create project storage folders; custom paths keep app.py compatible."""
    if not paths:
        paths = (
            PROJECT_ROOT / "data" / "uploads",
            PROJECT_ROOT / "data" / "processed",
            PROJECT_ROOT / "vector_db",
        )

    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def format_source_pages(pages: list[int]) -> str:
    if not pages:
        return "No source pages"
    label = "page" if len(pages) == 1 else "pages"
    return f"{label} {', '.join(str(page) for page in pages)}"
