from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = Path(args.history)
    df = pd.read_csv(history)
    out = Path(args.output) if args.output else history.with_name("training_curves.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved {out}")

    out_acc = out.with_name("accuracy_curves.png")
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.plot(df["epoch"], df["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Accuracy and macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_acc, dpi=160)
    plt.close()
    print(f"Saved {out_acc}")


if __name__ == "__main__":
    main()
