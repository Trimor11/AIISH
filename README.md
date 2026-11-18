# AIISH

A small GPT-style language model built from scratch in PyTorch. It trains on plain text and generates new text one character at a time.

## Structure

```
configs/default.yaml   model size and training settings
data/sample.txt        sample training text (Shakespeare excerpt, public domain)
src/aiish/
  config.py            ModelConfig / TrainConfig and YAML loading
  tokenizer.py         character-level tokenizer
  model.py             decoder-only transformer (attention, MLP, blocks, generate)
  data.py              train/val split and batch sampling
  train.py             training loop, evaluation, checkpointing
  generate.py          load a checkpoint and sample text
tests/                 unit tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Train

```bash
aiish-train                      # uses configs/default.yaml
aiish-train --max-steps 300      # quick run
```

The best checkpoint (lowest validation loss) is written to `checkpoints/`. To train on your own text, point `train.data_path` in the config at any UTF-8 text file.

## Generate

```bash
aiish-generate --prompt "First Citizen:" --max-new-tokens 300
```

## Develop

```bash
pytest
ruff check .
ruff format --check .
```
