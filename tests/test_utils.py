from src.utils import ensure_directories, format_source_pages


def test_format_source_pages_single_page():
    assert format_source_pages([4]) == "page 4"


def test_format_source_pages_multiple_pages():
    assert format_source_pages([2, 4, 7]) == "pages 2, 4, 7"


def test_format_source_pages_empty():
    assert format_source_pages([]) == "No source pages"


def test_format_source_pages_sorts_and_dedupes():
    # Retrieval yields pages in relevance order with one entry per chunk.
    assert format_source_pages([7, 2, 7, 4]) == "pages 2, 4, 7"
    assert format_source_pages([3, 3]) == "page 3"


def test_ensure_directories_creates_nested_paths(tmp_path):
    target = tmp_path / "a" / "b" / "uploads"

    ensure_directories(target)

    assert target.is_dir()


def test_ensure_directories_accepts_multiple_paths(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two" / "nested"

    ensure_directories(first, second)

    assert first.is_dir()
    assert second.is_dir()
