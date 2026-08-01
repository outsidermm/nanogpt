"""Model and training configuration."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Hyperparameters that define a GPT model's architecture."""

    vocab_size: int
    context_size: int = 256
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Hyperparameters that control the training run."""

    data_path: str = "input.txt"
    train_split: float = 0.9
    batch_size: int = 64
    max_steps: int = 5_000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4
    seed: int = 42
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 500
    resume: str | None = None
