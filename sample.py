"""Generate text from a trained checkpoint.

Usage:
    uv run sample.py --checkpoint checkpoints/ckpt_final.pt --prompt "ROMEO:" --temperature 0.8 --top-k 40
"""

import argparse

import torch

from config import GPTConfig
from data import CharTokenizer
from model import GPTLanguageModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="path to a checkpoint .pt file")
    p.add_argument("--prompt", default="", help="text to seed generation with")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0, help="higher = more random")
    p.add_argument("--top-k", type=int, default=None, help="restrict sampling to top-k logits")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    tokenizer = CharTokenizer(checkpoint["chars"])
    model = GPTLanguageModel(GPTConfig(**checkpoint["model_config"])).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.prompt:
        context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=args.device)

    result = model.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    print(tokenizer.decode(result))


if __name__ == "__main__":
    main()
