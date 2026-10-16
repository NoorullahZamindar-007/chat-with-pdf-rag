from src.text_cleaner import clean_text


def test_clean_text_joins_hyphenated_line_break():
    assert clean_text("exper-\nimental results") == "experimental results"


def test_clean_text_joins_hyphenated_line_break_with_crlf():
    assert clean_text("exper-\r\nimental results") == "experimental results"


def test_clean_text_keeps_deliberate_compound_hyphens():
    assert clean_text("state-of-the-art") == "state-of-the-art"


def test_clean_text_keeps_hyphen_before_capitalized_line_start():
    # A hyphen before a new sentence/paragraph is a real hyphen, not a wrap.
    assert clean_text("Marie-\nCurie") == "Marie-\nCurie"


def test_clean_text_keeps_hyphen_at_paragraph_break():
    assert clean_text("well-\n\nknown") == "well-\n\nknown"


def test_clean_text_folds_ligatures():
    # PyMuPDF often extracts typographic ligatures as single codepoints.
    assert clean_text("ﬁle ﬂow") == "file flow"


def test_clean_text_folds_fullwidth_characters():
    assert clean_text("ＡＩ") == "AI"
