"""A small decoder-only transformer (GPT), following Karpathy's nanoGPT walkthrough."""

import torch
from torch import nn
from torch.nn import functional as F

from config import GPTConfig


class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.context_size, config.context_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        _, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = q @ k.transpose(-2, -1) * (C**-0.5)  # (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weight = F.softmax(weight, dim=-1)  # (B, T, T)
        weight = self.dropout(weight)
        output = weight @ v  # (B, T, head_size)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(config, head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.ReLU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.mha = MultiHeadAttention(config, config.n_head, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_embedding_table = nn.Embedding(config.context_size, config.n_embd)
        self.blocks = nn.Sequential(*[AttentionBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        idx, targets: (B, T) tensors of integers.
        Returns logits (B, T, C) and, if targets given, the mean cross-entropy loss.
        """
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        positional_embeddings = self.positional_embedding_table(
            torch.arange(T, device=idx.device)
        )
        x = token_embeddings + positional_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
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
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """Autoregressively sample max_new_tokens new tokens, appended to idx."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (B, C)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
