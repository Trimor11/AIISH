from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 0
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


@dataclass
class TrainConfig:
    data_path: str = "data/sample.txt"
    out_dir: str = "checkpoints"
    batch_size: int = 32
    max_steps: int = 2000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_batches: int = 20
    val_fraction: float = 0.1
    seed: int = 1337


def _build(cls, values: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _build(ModelConfig, raw.get("model", {})), _build(TrainConfig, raw.get("train", {}))
