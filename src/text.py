from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text: str, lower: bool = True) -> list[str]:
    if lower:
        text = text.lower()
    return TOKEN_RE.findall(text)


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    pad_idx: int = 0
    unk_idx: int = 1

    @classmethod
    def build(
        cls,
        tokenized_texts: Iterable[Sequence[str]],
        max_size: int = 50_000,
        min_freq: int = 2,
    ) -> "Vocab":
        counter: Counter[str] = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        itos = [PAD_TOKEN, UNK_TOKEN]
        for word, freq in counter.most_common():
            if freq < min_freq:
                break
            if word in (PAD_TOKEN, UNK_TOKEN):
                continue
            itos.append(word)
            if len(itos) >= max_size:
                break
        stoi = {word: i for i, word in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, tokens: Sequence[str], max_len: int) -> tuple[list[int], int]:
        ids = [self.stoi.get(tok, self.unk_idx) for tok in tokens[:max_len]]
        length = max(1, len(ids))
        if len(ids) < max_len:
            ids.extend([self.pad_idx] * (max_len - len(ids)))
        return ids, length
