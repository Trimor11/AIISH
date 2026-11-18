import argparse
from pathlib import Path

import torch

from aiish.config import ModelConfig
from aiish.model import GPT
from aiish.tokenizer import CharTokenizer


def load_model(checkpoint_dir: str | Path, device: str) -> tuple[GPT, CharTokenizer]:
    checkpoint_dir = Path(checkpoint_dir)
    ckpt = torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=True)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, CharTokenizer.load(checkpoint_dir / "tokenizer.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a trained AIISH model")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.checkpoint_dir, args.device)
    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
