import torch


def split_data(ids: torch.Tensor, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_val = int(len(ids) * val_fraction)
    return ids[: len(ids) - n_val], ids[len(ids) - n_val :]


def get_batch(
    ids: torch.Tensor, block_size: int, batch_size: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(ids) <= block_size:
        raise ValueError(f"Need more than {block_size} tokens, got {len(ids)}")
    starts = torch.randint(len(ids) - block_size, (batch_size,))
    x = torch.stack([ids[i : i + block_size] for i in starts])
    y = torch.stack([ids[i + 1 : i + 1 + block_size] for i in starts])
    return x.to(device), y.to(device)
