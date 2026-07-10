import re


def clean_text(text: str) -> str:
    """Clean PDF text without changing the meaning of the sentence."""
    if not text:
        return ""

    # Null bytes can appear in extracted PDF text and break downstream tools.
    text = text.replace("\x00", " ")

    # Normalize horizontal whitespace while preserving new lines between paragraphs.
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
