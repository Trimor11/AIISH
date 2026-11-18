import pytest
import torch

from aiish.config import ModelConfig
from aiish.model import GPT


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0))


def test_forward_shapes_and_initial_loss(model):
    x = torch.randint(20, (4, 16))
    logits, loss = model(x, x)
    assert logits.shape == (4, 16, 20)
    assert abs(loss.item() - torch.log(torch.tensor(20.0)).item()) < 0.5


def test_is_causal(model):
    model.eval()
    x = torch.randint(20, (1, 16))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 20
    assert torch.allclose(model(x)[0][:, :-1], model(y)[0][:, :-1], atol=1e-5)


def test_generate_extends_past_block_size(model):
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=30, top_k=5)
    assert out.shape == (1, 31)
    assert out.max() < 20


def test_rejects_too_long_sequence(model):
    with pytest.raises(ValueError):
        model(torch.zeros((1, 17), dtype=torch.long))


def test_overfits_single_batch(model):
    x = torch.randint(20, (2, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(150):
        _, loss = model(x, x.roll(-1, dims=1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5
