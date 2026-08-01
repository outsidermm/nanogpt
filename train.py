"""Train a character-level GPT on a text corpus.

Usage:
    uv run train.py --data input.txt --max-steps 5000
    uv run train.py --resume checkpoints/ckpt_step500.pt
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from config import GPTConfig, TrainConfig
from data import get_batch, load_data
from model import GPTLanguageModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=TrainConfig.data_path, help="path to training text file")
    p.add_argument("--train-split", type=float, default=TrainConfig.train_split)
    p.add_argument("--context-size", type=int, default=GPTConfig.context_size)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--n-embd", type=int, default=GPTConfig.n_embd)
    p.add_argument("--n-layer", type=int, default=GPTConfig.n_layer)
    p.add_argument("--n-head", type=int, default=GPTConfig.n_head)
    p.add_argument("--dropout", type=float, default=GPTConfig.dropout)
    p.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    p.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    p.add_argument("--eval-iters", type=int, default=TrainConfig.eval_iters)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-dir", default=TrainConfig.checkpoint_dir)
    p.add_argument("--checkpoint-every", type=int, default=TrainConfig.checkpoint_every)
    p.add_argument("--resume", default=None, help="path to a checkpoint .pt file to resume from")
    return p.parse_args()


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    context_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Estimate average train/val loss over eval_iters batches."""
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, context_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def save_checkpoint(
    path: Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    chars: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "chars": chars,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_data, val_data, tokenizer = load_data(args.data, args.train_split)

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_size=args.context_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
    )
    model = GPTLanguageModel(model_config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params / 1e6:.2f}M parameters, vocab size {tokenizer.vocab_size}")

    checkpoint_dir = Path(args.checkpoint_dir)
    for step in range(start_step, args.max_steps):
        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(
                model, train_data, val_data, args.context_size, args.batch_size, args.eval_iters, args.device
            )
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if step > start_step and step % args.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"ckpt_step{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, tokenizer.chars)
            print(f"Saved checkpoint to {ckpt_path}")

        xb, yb = get_batch(train_data, args.context_size, args.batch_size, args.device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_path = checkpoint_dir / "ckpt_final.pt"
    save_checkpoint(final_path, model, optimizer, args.max_steps - 1, tokenizer.chars)
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
