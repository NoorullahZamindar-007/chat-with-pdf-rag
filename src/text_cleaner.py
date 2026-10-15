import re


def clean_text(text: str) -> str:
    """Clean PDF text without changing the meaning of the sentence."""
    if not text:
        return ""

    # Null bytes can appear in extracted PDF text and break downstream tools.
    text = text.replace("\x00", " ")

    # Rejoin words split by a hyphen at a line break ("exper-\nimental" ->
    # "experimental"). Justified PDF text commonly hyphenates this way, and
    # the fragments hurt embedding quality and retrieval. Only join when the
    # next line continues in lowercase, so deliberate compounds ("state-\nof
    # the art") and name splits ("Marie-\nCurie") keep their hyphen.
    text = re.sub(r"(\w)-[ \t]*\r?\n[ \t]*([a-z])", r"\1\2", text)

    # Normalize horizontal whitespace while preserving new lines between paragraphs.
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
