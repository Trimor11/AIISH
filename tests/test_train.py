from aiish.config import ModelConfig, TrainConfig, load_config
from aiish.generate import load_model
from aiish.train import train


def test_default_config_loads():
    model_cfg, train_cfg = load_config("configs/default.yaml")
    assert model_cfg.n_embd % model_cfg.n_head == 0
    assert train_cfg.max_steps > 0


def test_train_saves_loadable_checkpoint(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("the quick brown fox jumps over the lazy dog. " * 40)
    cfg = TrainConfig(
        data_path=str(data),
        out_dir=str(tmp_path / "ckpt"),
        batch_size=4,
        max_steps=5,
        warmup_steps=2,
        eval_interval=5,
        eval_batches=2,
    )
    train(ModelConfig(block_size=16, n_layer=1, n_head=2, n_embd=16), cfg, "cpu")
    model, tokenizer = load_model(tmp_path / "ckpt", "cpu")
    assert model.cfg.vocab_size == tokenizer.vocab_size
