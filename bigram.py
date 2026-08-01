"""
Bigram character-level language model, trained on a text corpus.
Based on the nanoGPT / "Let's build GPT" walkthrough by Andrej Karpathy.
"""

import torch
from torch import nn
from torch.nn import functional as F

# --- Hyperparameters ---
CONTEXT_SIZE = 8
BATCH_SIZE = 32
MAX_STEPS = 20_000
EVAL_INTERVAL = 200
EVAL_ITERS = 200
LEARNING_RATE = 1e-2
MAX_NEW_TOKENS = 500
SEED = 42
DATA_PATH = "input.txt"
TRAIN_SPLIT = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)


def load_data(path: str):
    """Load raw text and build character-level vocab and codecs."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(TRAIN_SPLIT * len(data))
    train_data, val_data = data[:n], data[n:]

    return train_data, val_data, vocab_size, decode


def get_batch(data: torch.Tensor):
    """Sample a random batch of (input, target) sequences from a dataset split."""
    ix = torch.randint(len(data) - CONTEXT_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + CONTEXT_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor):
    """Estimate average train/val loss over EVAL_ITERS batches."""
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(data)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    """Predicts the next token using only the current token (no context beyond it)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        idx, targets: (B, T) tensors of integers.
        Returns logits (B, T, C) and, if targets given, the mean cross-entropy loss.
        """
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """Autoregressively sample max_new_tokens new tokens, appended to idx."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def train(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor):
    """Run the training loop, periodically reporting train/val loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_STEPS):
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def main():
    train_data, val_data, vocab_size, decode = load_data(DATA_PATH)

    model = BigramLanguageModel(vocab_size).to(DEVICE)
    train(model, train_data, val_data)

    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    result = model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()
    print(decode(result))


if __name__ == "__main__":
    main()
