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
MAX_STEPS = 5_000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LEARNING_RATE = 1e-3
MAX_NEW_TOKENS = 500
SEED = 42
DATA_PATH = "input.txt"
TRAIN_SPLIT = 0.9
N_EMBEDDING_DIM = 32  # Dimensionality of the token embeddings
HEAD_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)


def load_data(path: str):
    """Load raw text and build character-level vocab and codecs."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
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


class SingleHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(N_EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(N_EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(N_EMBEDDING_DIM, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1) # (B, T, T)
        output = weight @ v  # (B, T, head_size)
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBEDDING_DIM, N_EMBEDDING_DIM)  # Project concatenated heads back to embedding dimension]
    
    def forward(self, x):
        output =  torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate along the embedding dimension
        output = self.proj(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd)
        )
    
    def forward(self, x):
        return self.net(x)

class AttentionBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.mha = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = x + self.mha(x)
        x = x + self.ffwd(x)
        return x

class BigramLanguageModel(nn.Module):
    """Predicts the next token using only the current token (no context beyond it)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBEDDING_DIM)  # Embedding layer to map token indices to embeddings
        self.positional_embedding_table = nn.Embedding(CONTEXT_SIZE, N_EMBEDDING_DIM)  # Embedding layer for positional encodings
        self.blocks = nn.Sequential(
            AttentionBlock(N_EMBEDDING_DIM, 4),  # Add attention blocks for context
            AttentionBlock(N_EMBEDDING_DIM, 4),
            AttentionBlock(N_EMBEDDING_DIM, 4),
        )
        self.lm_head = nn.Linear(N_EMBEDDING_DIM, vocab_size)  # Linear layer to project embeddings to vocab size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        idx, targets: (B, T) tensors of integers.
        Returns logits (B, T, C) and, if targets given, the mean cross-entropy loss.
        """
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B, T, N_EMBEDDING_DIM)
        positional_embeddings = self.positional_embedding_table(torch.arange(T, device=idx.device))  # (B, T, N_EMBEDDING_DIM)
        x = token_embeddings + positional_embeddings  # (B, T, N_EMBEDDING_DIM)
        x = self.blocks(x)  # (B, T, N_EMBEDDING_DIM)
        logits = self.lm_head(x)

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
            idx_cond = idx[:, -CONTEXT_SIZE:]  # crop context to the last CONTEXT_SIZE tokens
            logits, _ = self(idx_cond)
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

    
    # Attention is a communication mechanism that allows each token to attend to all other tokens in the sequence. It computes a weighted sum of the values (V) based on the similarity between queries (Q) and keys (K). The weights are determined by the dot product of Q and K, followed by a softmax operation to ensure they sum to 1. This allows the model to focus on relevant tokens when making predictions.

if __name__ == "__main__":
    main()
