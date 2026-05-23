from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from .train_classifier import train_from_config
from .utils import ensure_dir, load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--ag_news_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_yaml(args.experiments)
    base = load_yaml(spec["base_config"])
    if args.ag_news_path is not None:
        base["ag_news_path"] = args.ag_news_path

    summaries = []
    for exp in spec["experiments"]:
        cfg = copy.deepcopy(base)
        cfg.update(exp.get("overrides", {}))
        name = exp["name"]
        print(f"\nExperiment: {name}")
        summary = train_from_config(cfg, run_name=name)
        summary.update(exp.get("overrides", {}))
        summaries.append(summary)

    out_dir = ensure_dir(base.get("outputs_dir", "outputs"))
    table = pd.DataFrame(summaries).sort_values("best_val_macro_f1", ascending=False)
    table.to_csv(Path(out_dir) / "experiments_summary.csv", index=False)
    print(table)
    print(f"Saved results to {Path(out_dir) / 'experiments_summary.csv'}")


if __name__ == "__main__":
    main()
