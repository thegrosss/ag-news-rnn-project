from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from .text import Vocab


def load_word2vec(path: str | Path) -> Word2Vec:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Word2Vec model not found"
        )
    return Word2Vec.load(str(path))


def build_embedding_matrix(vocab: Vocab, w2v_path: str | Path, embedding_dim: int, seed: int = 42) -> np.ndarray:
    model = load_word2vec(w2v_path)
    if model.vector_size != embedding_dim:
        raise ValueError("embedding_dim != Word2Vec vector_size")

    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.05, size=(len(vocab.itos), embedding_dim)).astype("float32")
    matrix[vocab.pad_idx] = 0.0

    found = 0
    for word, idx in vocab.stoi.items():
        if word in model.wv:
            matrix[idx] = model.wv[word]
            found += 1
    coverage = found / max(1, len(vocab.itos))
    print(f"Embedding coverage: {found}/{len(vocab.itos)} = {coverage:.2%}")
    return matrix
