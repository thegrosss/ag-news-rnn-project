from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .train_classifier import train_from_config
from .utils import ensure_dir, load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--ag_news_path", default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--study_name", default=None)
    return parser.parse_args()


def _add_history_metrics(summary: dict) -> dict:
    history_path = Path(summary["history_path"])
    if not history_path.exists():
        return summary

    history = pd.read_csv(history_path)
    best_idx = history["val_macro_f1"].idxmax()
    best_row = history.loc[best_idx]
    summary["best_train_macro_f1"] = float(best_row["train_macro_f1"])
    summary["best_val_loss"] = float(best_row["val_loss"])
    summary["train_val_macro_f1_gap"] = float(best_row["train_macro_f1"] - best_row["val_macro_f1"])
    return summary


def _save_summary_plots(table: pd.DataFrame, out_dir: Path) -> None:
    if table.empty:
        return

    ordered = table.sort_values("best_val_macro_f1", ascending=True)
    labels = ordered["run_name"].astype(str)

    plt.figure(figsize=(9, max(4, 0.35 * len(ordered))))
    plt.barh(labels, ordered["best_val_macro_f1"])
    plt.xlabel("Best validation macro-F1")
    plt.tight_layout()
    plt.savefig(out_dir / "experiments_val_macro_f1.png", dpi=160)
    plt.close()

    if "train_val_macro_f1_gap" in ordered.columns:
        plt.figure(figsize=(9, max(4, 0.35 * len(ordered))))
        plt.barh(labels, ordered["train_val_macro_f1_gap"])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Train macro-F1 minus validation macro-F1")
        plt.tight_layout()
        plt.savefig(out_dir / "experiments_train_val_gap.png", dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    spec = load_yaml(args.experiments)
    base = load_yaml(spec["base_config"])
    if args.ag_news_path is not None:
        base["ag_news_path"] = args.ag_news_path

    if args.tune:
        from .optuna_tuning import run_study

        tuning_spec = spec.get("tuning", {})
        tuning_cfg = copy.deepcopy(base)
        tuning_cfg.update(tuning_spec.get("overrides", {}))
        n_trials = args.n_trials or int(tuning_spec.get("n_trials", spec.get("n_trials", 30)))
        study_name = args.study_name or tuning_spec.get("study_name", "ag_news_optuna")
        run_study(tuning_cfg, n_trials=n_trials, study_name=study_name)
        return

    summaries = []
    for exp in spec["experiments"]:
        cfg = copy.deepcopy(base)
        cfg.update(exp.get("overrides", {}))
        name = exp["name"]
        print(f"\nExperiment: {name}")
        summary = train_from_config(cfg, run_name=name)
        summary = _add_history_metrics(summary)
        summary.update(exp.get("overrides", {}))
        summaries.append(summary)

    out_dir = ensure_dir(base.get("outputs_dir", "outputs"))
    table = pd.DataFrame(summaries).sort_values("best_val_macro_f1", ascending=False)
    table.to_csv(Path(out_dir) / "experiments_summary.csv", index=False)
    _save_summary_plots(table, Path(out_dir))
    print(table)
    print(f"Saved results to {Path(out_dir) / 'experiments_summary.csv'}")
    print(f"Saved plots to {Path(out_dir) / 'experiments_val_macro_f1.png'}")


if __name__ == "__main__":
    main()
