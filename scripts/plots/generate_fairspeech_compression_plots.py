#!/usr/bin/env python3
"""Generate plots for the FairSpeech compression experiment.

Inputs are CSVs produced by scripts/metrics/compute_fairspeech_compression_metrics.py.
The script creates publication-ready scaffolding plots: group WER bars,
variant degradation curves, fairness-gap heatmaps, and insertion subtype stacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_pandas():
    try:
        import pandas as pd  # type: ignore
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore

        return pd, plt, sns
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("pandas, matplotlib, and seaborn are required for plotting.") from exc


def save_group_wer_bars(group_df, output_dir: Path, plt, sns) -> None:
    for model, model_df in group_df.groupby("model"):
        plt.figure(figsize=(12, 6))
        sns.barplot(data=model_df, x="group", y="wer", hue="audio_variant")
        plt.title(f"FairSpeech ethnicity WER by audio variant — {model}")
        plt.ylabel("WER")
        plt.xlabel("Ethnicity")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"group_wer_bars_{model}.png", dpi=200)
        plt.close()


def save_degradation_curves(delta_df, output_dir: Path, plt, sns) -> None:
    if delta_df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=delta_df, x="audio_variant", y="mean_delta_wer_vs_baseline", hue="group", style="model", markers=True)
    plt.title("Paired ΔWER versus baseline by ethnicity")
    plt.ylabel("Mean ΔWER vs baseline")
    plt.xlabel("Audio variant")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "degradation_curves_delta_wer.png", dpi=200)
    plt.close()


def save_gap_heatmap(fairness_df, output_dir: Path, plt, sns) -> None:
    if fairness_df.empty:
        return
    pivot = fairness_df.pivot_table(index="model", columns="audio_variant", values="relative_gap", aggfunc="mean")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="magma")
    plt.title("FairSpeech relative WER gap by model and audio variant")
    plt.tight_layout()
    plt.savefig(output_dir / "fairness_gap_heatmap.png", dpi=200)
    plt.close()


def save_insertion_subtype_stacks(group_df, output_dir: Path, pd, plt) -> None:
    cols = ["repetition_insertion_rate", "syntactic_insertion_rate", "content_insertion_rate"]
    if any(col not in group_df.columns for col in cols):
        return
    collapsed = group_df.groupby(["audio_variant"])[cols].mean().reset_index()
    collapsed = collapsed.set_index("audio_variant")
    ax = collapsed.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title("Insertion subtype rates by audio variant")
    ax.set_ylabel("Insertion subtype rate per reference word")
    ax.set_xlabel("Audio variant")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "insertion_subtype_stacks.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FairSpeech compression plots")
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pd, plt, sns = load_pandas()
    group_path = args.metrics_dir / "fairspeech_compression_group_metrics.csv"
    fairness_path = args.metrics_dir / "fairspeech_compression_fairness_metrics.csv"
    delta_path = args.metrics_dir / "fairspeech_compression_paired_delta.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    group_df = pd.read_csv(group_path) if group_path.exists() and group_path.stat().st_size else pd.DataFrame()
    fairness_df = pd.read_csv(fairness_path) if fairness_path.exists() and fairness_path.stat().st_size else pd.DataFrame()
    delta_df = pd.read_csv(delta_path) if delta_path.exists() and delta_path.stat().st_size else pd.DataFrame()

    if not group_df.empty:
        save_group_wer_bars(group_df, args.output_dir, plt, sns)
        save_insertion_subtype_stacks(group_df, args.output_dir, pd, plt)
    if not fairness_df.empty:
        save_gap_heatmap(fairness_df, args.output_dir, plt, sns)
    if not delta_df.empty:
        save_degradation_curves(delta_df, args.output_dir, plt, sns)

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
