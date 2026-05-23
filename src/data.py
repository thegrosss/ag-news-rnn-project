from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .text import Vocab, tokenize

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass
class PreparedData:
    train_dataset: Dataset
    val_dataset: Dataset
    vocab: Vocab
    train_df: pd.DataFrame
    val_df: pd.DataFrame


def _try_download_ag_news() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed"
        ) from exc

    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
    return Path(path)


def resolve_train_csv(ag_news_path: Optional[str | Path]) -> Path:
    if ag_news_path is None:
        train_csv = Path("data/train.csv")
    else:
        train_csv = Path(ag_news_path)

    if train_csv.is_dir():
        train_csv = train_csv / "train.csv"

    if not train_csv.exists():
        raise FileNotFoundError("train file not found")

    return train_csv


def load_ag_news_train(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = {"Class Index", "Title", "Description"}
    if not required_columns.issubset(df.columns):
        raise ValueError("train.csv must contain Class Index, Title and Description columns")

    labels = df["Class Index"].astype(int) - 1
    text = df["Title"].fillna("").astype(str) + " " + df["Description"].fillna("").astype(str)

    return pd.DataFrame({"label": labels, "text": text})


class AGNewsTensorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocab, max_len: int, lower: bool = True):
        self.labels = df["label"].astype(int).to_numpy()
        self.texts = df["text"].astype(str).tolist()
        self.vocab = vocab
        self.max_len = max_len
        self.lower = lower

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        tokens = tokenize(self.texts[idx], lower=self.lower)
        ids, length = self.vocab.encode(tokens, self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def prepare_data(
    ag_news_path: str | Path | None,
    val_size: float,
    max_vocab_size: int,
    min_freq: int,
    max_len: int,
    lower: bool,
    seed: int,
) -> PreparedData:
    train_csv = resolve_train_csv(ag_news_path)
    df = load_ag_news_train(train_csv)

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    tokenized_train = [tokenize(t, lower=lower) for t in train_df["text"].tolist()]
    vocab = Vocab.build(tokenized_train, max_size=max_vocab_size, min_freq=min_freq)

    train_dataset = AGNewsTensorDataset(train_df, vocab=vocab, max_len=max_len, lower=lower)
    val_dataset = AGNewsTensorDataset(val_df, vocab=vocab, max_len=max_len, lower=lower)

    return PreparedData(train_dataset, val_dataset, vocab, train_df, val_df)
