import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer. Swap for BPE once the model outgrows it."""

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as err:
            raise ValueError(f"Character {err.args[0]!r} is not in the vocabulary") from None

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.chars, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text()))
