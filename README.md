# nanoGPT

[![CI](https://github.com/outsidermm/nanogpt/actions/workflows/ci.yml/badge.svg)](https://github.com/outsidermm/nanogpt/actions/workflows/ci.yml)

A small, from-scratch, character-level GPT implementation in PyTorch, built while following
Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) walkthrough
and then extended into a runnable, testable project: CLI-configurable training, checkpointing,
resumable runs, and controllable sampling (temperature / top-k).

Two models are included:

- **`bigram.py`** — a minimal bigram model (predicts the next character from only the current
  one). Useful as a baseline to see how much attention actually buys you.
- **`model.py`** — a decoder-only transformer (multi-head self-attention, feed-forward blocks,
  residual connections, LayerNorm, dropout) trained on the same data.

## Architecture

- Character-level tokenizer (`data.py`) — vocabulary is just the unique characters in the corpus.
- Causal (masked) self-attention, computed per-head then concatenated and projected
  (`SingleHeadAttention` / `MultiHeadAttention` in `model.py`).
- Pre-norm transformer blocks: `x = x + attn(ln(x))`, `x = x + ffwd(ln(x))`.
- Final `LayerNorm` applied after all blocks, before the language-modeling head.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Training

```bash
uv run train.py --data input.txt --max-steps 5000
```

Key flags (see `uv run train.py --help` for the full list):

| Flag | Description | Default |
| --- | --- | --- |
| `--context-size` | tokens of context per example | 256 |
| `--n-embd` / `--n-layer` / `--n-head` | model size | 384 / 6 / 6 |
| `--lr` | learning rate | 3e-4 |
| `--checkpoint-dir`, `--checkpoint-every` | where/how often to save | `checkpoints/`, 500 |
| `--resume` | path to a checkpoint to resume training from | — |

Checkpoints (model weights, optimizer state, config, and vocabulary) are written to
`--checkpoint-dir` periodically and at the end of training, so a run can be resumed with
`--resume checkpoints/ckpt_step500.pt`.

## Sampling

```bash
uv run sample.py --checkpoint checkpoints/ckpt_final.pt --prompt "ROMEO:" \
    --max-new-tokens 500 --temperature 0.8 --top-k 40
```

`--temperature` controls randomness (lower = more deterministic) and `--top-k` restricts
sampling to the k most likely next characters at each step.

## Tests

```bash
uv run pytest
```

Covers tokenizer round-tripping, batch shapes, output shapes/loss sanity, and — importantly —
that attention is actually causal (a token's logits are unaffected by tokens that come after it).

## Project layout

```
config.py   # GPTConfig / TrainConfig dataclasses
data.py     # tokenizer + batching
model.py    # transformer model
train.py    # CLI training loop + checkpointing
sample.py   # CLI text generation from a checkpoint
bigram.py   # baseline bigram model (self-contained)
tests/      # pytest suite
```

## License

MIT — see [LICENSE](LICENSE).
