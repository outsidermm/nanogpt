import torch

from config import GPTConfig
from model import GPTLanguageModel


def tiny_config(**overrides) -> GPTConfig:
    defaults = {"vocab_size": 13, "context_size": 8, "n_embd": 16, "n_layer": 2, "n_head": 2, "dropout": 0.0}
    defaults.update(overrides)
    return GPTConfig(**defaults)


def test_forward_logits_shape_without_targets():
    config = tiny_config()
    model = GPTLanguageModel(config)
    idx = torch.randint(0, config.vocab_size, (4, config.context_size))

    logits, loss = model(idx)

    assert logits.shape == (4, config.context_size, config.vocab_size)
    assert loss is None


def test_forward_loss_is_finite_scalar_with_targets():
    config = tiny_config()
    model = GPTLanguageModel(config)
    idx = torch.randint(0, config.vocab_size, (4, config.context_size))
    targets = torch.randint(0, config.vocab_size, (4, config.context_size))

    _, loss = model(idx, targets)

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_generate_appends_requested_number_of_tokens():
    config = tiny_config()
    model = GPTLanguageModel(config)
    idx = torch.zeros((1, 1), dtype=torch.long)

    out = model.generate(idx, max_new_tokens=10)

    assert out.shape == (1, 11)


def test_attention_is_causal():
    """Logits at position t must not change when future tokens (after t) change."""
    config = tiny_config(dropout=0.0)
    model = GPTLanguageModel(config)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (1, config.context_size))
    idx_modified = idx.clone()
    idx_modified[0, -1] = (idx_modified[0, -1] + 1) % config.vocab_size

    logits, _ = model(idx)
    logits_modified, _ = model(idx_modified)

    # All positions except the last (whose input token itself changed) must be identical.
    assert torch.allclose(logits[:, :-1, :], logits_modified[:, :-1, :], atol=1e-5)
