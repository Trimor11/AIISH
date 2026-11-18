import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch

from aiish.config import ModelConfig, TrainConfig, load_config
from aiish.data import get_batch, split_data
from aiish.model import GPT
from aiish.tokenizer import CharTokenizer


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def estimate_loss(
    model: GPT, splits: dict[str, torch.Tensor], cfg: TrainConfig, device: str
) -> dict[str, float]:
    model.eval()
    out = {}
    for name, ids in splits.items():
        losses = [
            model(*get_batch(ids, model.cfg.block_size, cfg.batch_size, device))[1].item()
            for _ in range(cfg.eval_batches)
        ]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def save_checkpoint(model: GPT, tokenizer: CharTokenizer, out_dir: Path, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "model_config": asdict(model.cfg), "step": step},
        out_dir / "model.pt",
    )
    tokenizer.save(out_dir / "tokenizer.json")


def train(model_cfg: ModelConfig, cfg: TrainConfig, device: str) -> GPT:
    torch.manual_seed(cfg.seed)
    text = Path(cfg.data_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    model_cfg.vocab_size = tokenizer.vocab_size
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_ids, val_ids = split_data(ids, cfg.val_fraction)
    splits = {"train": train_ids, "val": val_ids}

    model = GPT(model_cfg).to(device)
    print(f"{model.num_parameters() / 1e6:.2f}M parameters, vocab {tokenizer.vocab_size}, {device}")
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
    )

    out_dir = Path(cfg.out_dir)
    best_val = float("inf")
    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            losses = estimate_loss(model, splits, cfg, device)
            print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(model, tokenizer, out_dir, step)
        if step == cfg.max_steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        _, loss = model(*get_batch(train_ids, model_cfg.block_size, cfg.batch_size, device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AIISH language model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_cfg, train_cfg = load_config(args.config)
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    train(model_cfg, train_cfg, args.device)


if __name__ == "__main__":
    main()
