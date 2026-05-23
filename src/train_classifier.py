from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import LABEL_NAMES, prepare_data
from .embeddings import build_embedding_matrix
from .model import RNNTextClassifier
from .utils import ensure_dir, get_device, load_yaml, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lstm_baseline.yaml")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--ag_news_path", default=None)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, grad_clip: float | None = None):
    model.train()
    total_loss = 0.0
    preds_all, labels_all = [], []
    for batch in tqdm(loader, desc="training...", leave=False):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds_all.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels_all, preds_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []
    for batch in tqdm(loader, desc="checking...", leave=False):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds_all.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels_all, preds_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    return avg_loss, acc, macro_f1, np.array(labels_all), np.array(preds_all)


def train_from_config(config: dict[str, Any], run_name: str | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()
    print(f"Device: {device}")

    if cfg.get("ag_news_path") is None and cfg.get("ag_news_path_override") is not None:
        cfg["ag_news_path"] = cfg["ag_news_path_override"]

    prepared = prepare_data(
        ag_news_path=cfg.get("ag_news_path"),
        val_size=float(cfg["val_size"]),
        max_vocab_size=int(cfg["max_vocab_size"]),
        min_freq=int(cfg["min_freq"]),
        max_len=int(cfg["max_len"]),
        lower=bool(cfg.get("lower", True)),
        seed=int(cfg.get("seed", 42)),
    )
    print(f"Train: {len(prepared.train_dataset)}, valid: {len(prepared.val_dataset)}, vocab: {len(prepared.vocab.itos)}")

    embedding_matrix = build_embedding_matrix(
        prepared.vocab,
        cfg["embedding_path"],
        embedding_dim=int(cfg["embedding_dim"]),
        seed=int(cfg.get("seed", 42)),
    )

    train_loader = DataLoader(
        prepared.train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        prepared.val_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    model = RNNTextClassifier(
        vocab_size=len(prepared.vocab.itos),
        embedding_dim=int(cfg["embedding_dim"]),
        num_classes=4,
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        model_type=str(cfg["model_type"]).lower(),
        bidirectional=bool(cfg["bidirectional"]),
        dropout=float(cfg["dropout"]),
        pad_idx=prepared.vocab.pad_idx,
        pooling=str(cfg.get("pooling", "last")),
        embedding_weights=embedding_matrix,
        freeze_embeddings=bool(cfg.get("freeze_embeddings", False)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    run_name = run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path(cfg.get("outputs_dir", "outputs")) / run_name)
    ensure_dir(out_dir)
    save_json(cfg, out_dir / "config.json")

    history = []
    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip=float(cfg.get("grad_clip", 0))
        )
        val_loss, val_acc, val_f1, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "vocab_itos": prepared.vocab.itos,
                    "config": cfg,
                    "best_val_macro_f1": best_f1,
                    "label_names": LABEL_NAMES,
                },
                out_dir / "best_model.pt",
            )
            np.savetxt(out_dir / "confusion_matrix.csv", confusion_matrix(y_true, y_pred), fmt="%d", delimiter=",")
            report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4)
            (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(cfg.get("patience", 3)):
            print("Early stopping triggered.")
            break

    final = {
        "run_name": run_name,
        "output_dir": str(out_dir),
        "best_val_macro_f1": float(best_f1),
        "best_epoch": int(pd.DataFrame(history)["val_macro_f1"].idxmax() + 1),
        "history_path": str(out_dir / "history.csv"),
        "checkpoint_path": str(out_dir / "best_model.pt"),
    }
    save_json(final, out_dir / "summary.json")
    print(final)
    return final


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.ag_news_path is not None:
        cfg["ag_news_path"] = args.ag_news_path
    train_from_config(cfg, run_name=args.run_name)


if __name__ == "__main__":
    main()
