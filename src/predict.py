from __future__ import annotations

import argparse

import torch

from .data import LABEL_NAMES
from .text import Vocab, tokenize
from .train_classifier import build_classifier_model
from .utils import get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    vocab = Vocab(stoi={w: i for i, w in enumerate(ckpt["vocab_itos"])}, itos=ckpt["vocab_itos"])

    model = build_classifier_model(
        cfg=cfg,
        vocab_size=len(vocab.itos),
        embedding_matrix=None,
        pad_idx=vocab.pad_idx,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokens = tokenize(args.text, lower=bool(cfg.get("lower", True)))
    ids, length = vocab.encode(tokens, int(cfg["max_len"]))
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids, lengths)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    for label, prob in sorted(zip(LABEL_NAMES, probs), key=lambda x: x[1], reverse=True):
        print(f"{label}: {prob:.4f}")


if __name__ == "__main__":
    main()
