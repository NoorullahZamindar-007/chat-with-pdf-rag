from src.utils import ensure_directories, format_source_pages


def test_format_source_pages_single_page():
    assert format_source_pages([4]) == "page 4"


def test_format_source_pages_multiple_pages():
    assert format_source_pages([2, 4, 7]) == "pages 2, 4, 7"


def test_format_source_pages_empty():
    assert format_source_pages([]) == "No source pages"


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
