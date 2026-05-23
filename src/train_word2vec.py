from __future__ import annotations

import argparse
from pathlib import Path

import gensim.downloader as api
from gensim.models import Word2Vec

from .utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/text8_word2vec.model")
    parser.add_argument("--vector_size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sg", type=int, default=1)
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("Loading text8 from gensim-data...")
    corpus = api.load("text8")

    print("Training Word2Vec...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,
        negative=args.negative,
        epochs=args.epochs,
        seed=args.seed,
    )

    output = Path(args.output)
    ensure_dir(output.parent)
    model.save(str(output))
    print(f"Saved Word2Vec model to {output}")
    print(f"Vocabulary size: {len(model.wv.index_to_key)}")
    print("Nearest words to 'car':", model.wv.most_similar("car", topn=5))


if __name__ == "__main__":
    main()
