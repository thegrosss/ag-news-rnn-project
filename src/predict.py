from __future__ import annotations

import argparse

import torch

from .data import LABEL_NAMES
from .model import RNNTextClassifier
from .text import Vocab, tokenize
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

    model = RNNTextClassifier(
        vocab_size=len(vocab.itos),
        embedding_dim=int(cfg["embedding_dim"]),
        num_classes=4,
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        model_type=str(cfg["model_type"]).lower(),
        bidirectional=bool(cfg["bidirectional"]),
        dropout=float(cfg["dropout"]),
        pad_idx=vocab.pad_idx,
        pooling=str(cfg.get("pooling", "last")),
        embedding_weights=None,
        freeze_embeddings=bool(cfg.get("freeze_embeddings", False)),
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
