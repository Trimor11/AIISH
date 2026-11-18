import pytest

from aiish.tokenizer import CharTokenizer


def test_round_trip(tmp_path):
    tok = CharTokenizer.from_text("hello world")
    assert tok.decode(tok.encode("hello world")) == "hello world"
    tok.save(tmp_path / "tok.json")
    assert CharTokenizer.load(tmp_path / "tok.json").chars == tok.chars


def test_unknown_character():
    with pytest.raises(ValueError):
        CharTokenizer.from_text("abc").encode("z")
