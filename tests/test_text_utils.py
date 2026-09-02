from govdoc_explainer.text_utils import fs_safe_url, split_text_into_logical_sections


def test_fs_safe_url():
    assert fs_safe_url("https://example.com/path") == "https___example.com_path"
    assert fs_safe_url("simple") == "simple"


def test_split_text_empty():
    result = split_text_into_logical_sections("")
    assert result == [""]


def test_split_text_short():
    result = split_text_into_logical_sections("This is a simple sentence.")
    assert len(result) == 1
    assert "simple sentence" in result[0]


def test_split_text_multiple_sentences():
    text = "First sentence here. Second sentence follows. Third one arrives. Fourth is here. Fifth appears."
    result = split_text_into_logical_sections(text, max_sentences_per_section=2)
    assert len(result) >= 2
