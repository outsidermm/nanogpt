"""Character-level tokenization and batching."""

import torch


class CharTokenizer:
    """Maps characters to integer ids and back."""

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_data(path: str, train_split: float = 0.9) -> tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    """Load raw text and build a character-level tokenizer and train/val splits."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(sorted(set(text)))
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_split * len(data))
    return data[:n], data[n:], tokenizer


def get_batch(
    data: torch.Tensor, context_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences from a dataset split."""
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i : i + context_size] for i in ix])
    y = torch.stack([data[i + 1 : i + context_size + 1] for i in ix])
    return x.to(device), y.to(device)
