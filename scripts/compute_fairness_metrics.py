#!/usr/bin/env python3
"""
Compute fairness metrics from ASR inference predictions.

This script:
1. Reads prediction CSVs from run_inference.py
2. Computes aggregate WER per demographic subgroup
3. Calculates fairness metrics (gaps, ratios, std)
4. Runs statistical significance tests (bootstrap CI, Mann-Whitney U)
5. Generates publication-ready tables and figures

Usage:
    python scripts/compute_fairness_metrics.py \
        --results_dir results/ \
        --output_dir results/analysis/ \
        --models whisper_medium whisper_large_v3 wav2vec2_large
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import jiwer
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

MIN_GROUP_SIZE = 50  # Minimum samples per subgroup to report

# ── Model ordering: generation-first, then size ────────────────────────────
MODEL_ORDER = [
    "wav2vec2-large",         # Gen 1
    "whisper-small",          # Gen 2
    "whisper-medium",         # Gen 2
    "whisper-large-v3",       # Gen 2
    "qwen3-asr-0.6b",        # Gen 3
    "qwen3-asr-1.7b",        # Gen 3
    "granite-speech-3.3-2b",  # Gen 3
    "granite-speech-3.3-8b",  # Gen 3
    "canary-qwen-2.5b",      # Gen 3
]

MODEL_SHORT = {
    "wav2vec2-large": "W2V2-L",
    "whisper-small": "Wh-S",
    "whisper-medium": "Wh-M",
    "whisper-large-v3": "Wh-L",
    "qwen3-asr-0.6b": "Q3-0.6B",
    "qwen3-asr-1.7b": "Q3-1.7B",
    "granite-speech-3.3-2b": "Gr-2B",
    "granite-speech-3.3-8b": "Gr-8B",
    "canary-qwen-2.5b": "Can-2.5B",
}

MODEL_COLORS = {
    "wav2vec2-large":        "#2E86AB",  # Steel blue
    "whisper-small":         "#F18F01",  # Amber
    "whisper-medium":        "#3B7A57",  # Forest green
    "whisper-large-v3":      "#A23B72",  # Plum
    "qwen3-asr-0.6b":       "#FFFF00",  # Yellow
    "qwen3-asr-1.7b":       "#44AF69",  # Green
    "granite-speech-3.3-2b": "#6B4C9A",  # Purple
    "granite-speech-3.3-8b": "#E84855",  # Coral red
    "canary-qwen-2.5b":     "#D4A574",  # Tan
}

def get_ordered_models(models):
    """Return models sorted by generation-first, size-second order."""
    return [m for m in MODEL_ORDER if m in models]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ASR fairness metrics")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/users/PAS2030/srishti/asr_fairness/results/commonvoice",
        help="Directory with prediction CSVs from run_inference.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/users/PAS2030/srishti/asr_fairness/results/analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to analyze (auto-detected if not specified)",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for confidence intervals",
    )
    return parser.parse_args()


# ── WER computation ─────────────────────────────────────────────────────────
def compute_group_wer(df: pd.DataFrame) -> dict:
    """Compute aggregate WER for a group of utterances."""
    refs = df["reference"].fillna("").astype(str).tolist()
    hyps = df["hypothesis"].fillna("").astype(str).tolist()

    # Filter out empty references/hypotheses
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if not valid:
        return {"wer": None, "n_utterances": 0}

    refs_clean, hyps_clean = zip(*valid)
    refs_clean, hyps_clean = list(refs_clean), list(hyps_clean)

    # Aggregate WER (corpus-level, not average of per-utterance)
    wer_val = jiwer.wer(refs_clean, hyps_clean)

    # Detailed error breakdown using process_words
    output = jiwer.process_words(refs_clean, hyps_clean)
    total_words = output.substitutions + output.insertions + output.deletions + output.hits

    return {
        "wer": wer_val,
        "n_utterances": len(refs_clean),
        "substitution_rate": output.substitutions / output.hits if output.hits > 0 else 0,
        "insertion_rate": output.insertions / output.hits if output.hits > 0 else 0,
        "deletion_rate": output.deletions / output.hits if output.hits > 0 else 0,
        "substitutions": output.substitutions,
        "insertions": output.insertions,
        "deletions": output.deletions,
        "hits": output.hits,
    }


def bootstrap_wer(df: pd.DataFrame, n_bootstrap: int = 1000) -> tuple:
    """Compute bootstrap 95% confidence interval for WER."""
    refs = df["reference"].fillna("").astype(str).tolist()
    hyps = df["hypothesis"].fillna("").astype(str).tolist()
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if len(valid) < 10:
        return None, None, None

    refs_clean, hyps_clean = zip(*valid)
    refs_clean, hyps_clean = list(refs_clean), list(hyps_clean)
    n = len(refs_clean)

    wers = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(n, n, replace=True)
        ref_sample = [refs_clean[i] for i in indices]
        hyp_sample = [hyps_clean[i] for i in indices]
        try:
            wers.append(jiwer.wer(ref_sample, hyp_sample))
        except Exception:
            continue

    if not wers:
        return None, None, None

    return np.mean(wers), np.percentile(wers, 2.5), np.percentile(wers, 97.5)


# ── Fairness metrics ───────────────────────────────────────────────────────
def compute_fairness_metrics(group_wers: dict) -> dict:
    """Compute fairness metrics from per-group WER values."""
    valid_groups = {k: v for k, v in group_wers.items() if v["wer"] is not None and v["n_utterances"] >= MIN_GROUP_SIZE}

    if len(valid_groups) < 2:
        return {"error": "Too few valid groups"}

    wer_values = {k: v["wer"] for k, v in valid_groups.items()}
    wer_list = list(wer_values.values())

    best_group = min(wer_values, key=wer_values.get)
    worst_group = max(wer_values, key=wer_values.get)

    return {
        "group_wers": {k: {"wer": v["wer"], "n": v["n_utterances"]} for k, v in valid_groups.items()},
        "best_group": best_group,
        "best_wer": wer_values[best_group],
        "worst_group": worst_group,
        "worst_wer": wer_values[worst_group],
        "max_min_ratio": max(wer_list) / min(wer_list) if min(wer_list) > 0 else float("inf"),
        "relative_gap_pct": (wer_values[worst_group] - wer_values[best_group]) / wer_values[best_group] * 100 if wer_values[best_group] > 0 else float("inf"),
        "wer_std": float(np.std(wer_list)),
        "wer_range": max(wer_list) - min(wer_list),
    }


# ── Statistical tests ──────────────────────────────────────────────────────
def pairwise_significance(df: pd.DataFrame, group_col: str, min_size: int = MIN_GROUP_SIZE) -> dict:
    """Run Mann-Whitney U tests between all pairs of demographic groups."""
    results = {}
    groups = df[df[group_col] != ""].groupby(group_col)
    valid_groups = {name: group for name, group in groups if len(group) >= min_size}

    for (g1, df1), (g2, df2) in combinations(valid_groups.items(), 2):
        wers1 = df1["wer"].dropna().values
        wers2 = df2["wer"].dropna().values

        if len(wers1) < 10 or len(wers2) < 10:
            continue

        stat, p_value = stats.mannwhitneyu(wers1, wers2, alternative="two-sided")
        results[f"{g1}_vs_{g2}"] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01,
            "n1": len(wers1),
            "n2": len(wers2),
            "median_wer_1": float(np.median(wers1)),
            "median_wer_2": float(np.median(wers2)),
        }

    return results


# ── Visualization ──────────────────────────────────────────────────────────
def plot_wer_by_group(all_results: dict, demographic: str, output_dir: str):
    """Bar chart of WER by demographic group across models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use generation-first, size-second ordering
    models = get_ordered_models(list(all_results.keys()))

    # Collect all groups across models
    all_groups = set()
    for model, results in all_results.items():
        if demographic in results and "group_wers" in results[demographic]:
            all_groups.update(results[demographic]["group_wers"].keys())

    groups = sorted(all_groups)
    x = np.arange(len(groups))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        wers = []
        for group in groups:
            if (
                demographic in all_results[model]
                and "group_wers" in all_results[model][demographic]
                and group in all_results[model][demographic]["group_wers"]
            ):
                wers.append(all_results[model][demographic]["group_wers"][group]["wer"] * 100)
            else:
                wers.append(0)

        color = MODEL_COLORS.get(model, "#999999")
        label = MODEL_SHORT.get(model, model)
        ax.bar(x + i * width, wers, width, label=label, color=color,
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel(demographic.capitalize(), fontsize=12)
    ax.set_ylabel("WER (%)", fontsize=12)
    ax.set_title(f"WER by {demographic.capitalize()} Across ASR Models", fontsize=14)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"wer_by_{demographic}.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, f"wer_by_{demographic}.pdf"))
    plt.close()


def plot_fairness_heatmap(all_results: dict, output_dir: str):
    """Heatmap of fairness metrics across models and demographics."""
    demographics = ["gender", "accent", "age"]
    models = get_ordered_models(list(all_results.keys()))

    data = []
    for model in models:
        row = {}
        # Include overall WER as the first metric
        row["Overall\nWER (%)"] = all_results[model].get("overall_wer", 0) * 100
        
        for demo in demographics:
            if demo in all_results[model] and "max_min_ratio" in all_results[model][demo]:
                row[f"{demo}\nMax/Min"] = all_results[model][demo]["max_min_ratio"]
                row[f"{demo}\nGap (%)"] = all_results[model][demo]["relative_gap_pct"]
            else:
                row[f"{demo}\nMax/Min"] = np.nan
                row[f"{demo}\nGap (%)"] = np.nan
        data.append(row)

    df_heat = pd.DataFrame(data, index=models)

    # Reorder columns to ensure Overall WER is first
    cols = ["Overall\nWER (%)"] + [c for c in df_heat.columns if c != "Overall\nWER (%)"]
    df_heat = df_heat[cols]

    fig, ax = plt.subplots(figsize=(12, max(3, len(models))))
    sns.heatmap(df_heat, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Value"})
    ax.set_title("Fairness & Performance Metrics Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fairness_heatmap.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "fairness_heatmap.pdf"))
    plt.close()


# ── Table generation ───────────────────────────────────────────────────────
def generate_latex_table(all_results: dict, demographic: str, output_dir: str):
    """Generate LaTeX table for the paper."""
    models = get_ordered_models(list(all_results.keys()))

    # Collect all groups
    all_groups = set()
    for model in models:
        if demographic in all_results[model] and "group_wers" in all_results[model][demographic]:
            all_groups.update(all_results[model][demographic]["group_wers"].keys())
    groups = sorted(all_groups)

    # Header
    header = "Model & Architecture"
    for g in groups:
        header += f" & {g.capitalize()}"
    header += r" & Gap (\%) & Max/Min \\"

    # Rows
    rows = []
    for model in models:
        arch = all_results[model].get("model_info", {}).get("architecture", "")
        row = f"{model} & {arch}"
        for g in groups:
            if (
                demographic in all_results[model]
                and "group_wers" in all_results[model][demographic]
                and g in all_results[model][demographic]["group_wers"]
            ):
                wer = all_results[model][demographic]["group_wers"][g]["wer"] * 100
                row += f" & {wer:.1f}"
            else:
                row += " & --"

        if demographic in all_results[model]:
            gap = all_results[model][demographic].get("relative_gap_pct", 0)
            ratio = all_results[model][demographic].get("max_min_ratio", 0)
            row += f" & {gap:.1f} & {ratio:.2f}"
        else:
            row += " & -- & --"

        row += r" \\"
        rows.append(row)

    # Build table
    n_cols = 2 + len(groups) + 2
    latex = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        r"\caption{WER (\%) by " + demographic + r" across ASR architectures on Common Voice English test set.}" + "\n"
        r"\label{tab:wer_" + demographic + "}" + "\n"
        r"\begin{tabular}{" + "l" * 2 + "r" * (len(groups) + 2) + "}" + "\n"
        r"\toprule" + "\n"
        + header + "\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )

    table_path = os.path.join(output_dir, f"table_{demographic}.tex")
    with open(table_path, "w") as f:
        f.write(latex)
    print(f"  Saved LaTeX table: {table_path}")
    return latex


# ── Main analysis ──────────────────────────────────────────────────────────
def analyze_model(df: pd.DataFrame, model_name: str, n_bootstrap: int = 1000) -> dict:
    """Run complete fairness analysis for one model."""
    results = {}

    # Overall WER
    overall = compute_group_wer(df)
    results["overall_wer"] = overall["wer"]
    results["n_utterances"] = overall["n_utterances"]

    # Analyze each demographic dimension
    for demographic in ["gender", "accent", "age"]:
        print(f"\n  Analyzing {demographic}...")
        df_demo = df[df[demographic] != ""]

        if len(df_demo) == 0:
            print(f"    No {demographic} labels found, skipping.")
            continue

        # Compute per-group WER
        group_wers = {}
        groups = df_demo.groupby(demographic)
        for name, group in groups:
            if len(group) < MIN_GROUP_SIZE:
                print(f"    {name}: {len(group)} samples (< {MIN_GROUP_SIZE}, skipping)")
                continue

            wer_info = compute_group_wer(group)

            # Bootstrap CI
            mean_wer, ci_low, ci_high = bootstrap_wer(group, n_bootstrap)
            wer_info["bootstrap_ci_low"] = ci_low
            wer_info["bootstrap_ci_high"] = ci_high

            group_wers[name] = wer_info
            print(f"    {name:15s}: WER={wer_info['wer']*100:.2f}% [{ci_low*100:.2f}%, {ci_high*100:.2f}%] (n={len(group):,})")

        # Fairness metrics
        fairness = compute_fairness_metrics(group_wers)
        results[demographic] = fairness

        # Pairwise significance
        sig_results = pairwise_significance(df, demographic)
        results[f"{demographic}_significance"] = sig_results

        if sig_results:
            print(f"    Pairwise tests:")
            for pair, result in sig_results.items():
                sig_str = "***" if result["significant_001"] else ("*" if result["significant_005"] else "n.s.")
                print(f"      {pair}: p={result['p_value']:.4f} {sig_str}")

    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ASR Fairness Analysis")
    print(f"{'='*60}\n")

    # Auto-detect prediction files
    pred_files = sorted([
        f for f in os.listdir(args.results_dir)
        if f.startswith("predictions_") and f.endswith(".csv")
    ])

    if not pred_files:
        print(f"ERROR: No prediction files found in {args.results_dir}")
        print(f"Run run_inference.py first to generate predictions.")
        sys.exit(1)

    print(f"Found prediction files: {pred_files}")

    all_results = {}

    for pred_file in pred_files:
        model_name = pred_file.replace("predictions_", "").replace(".csv", "")
        display_name = model_name.replace("_", "-")

        print(f"\n{'─'*40}")
        print(f"Analyzing: {display_name}")
        print(f"{'─'*40}")

        df = pd.read_csv(os.path.join(args.results_dir, pred_file))
        # Fill NaN demographic columns
        for col in ["gender", "accent", "age"]:
            if col in df.columns:
                df[col] = df[col].fillna("")

        print(f"  Total predictions: {len(df):,}")

        # Load model metadata if available
        meta_path = os.path.join(args.results_dir, f"meta_{model_name}.json")
        model_info = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                model_info = json.load(f).get("model_info", {})

        results = analyze_model(df, model_name, args.n_bootstrap)
        results["model_info"] = model_info
        all_results[display_name] = results

    # ── Generate outputs ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Generating outputs...")
    print(f"{'='*60}")

    # Save complete analysis JSON
    analysis_path = os.path.join(args.output_dir, "full_analysis.json")

    # Deep-convert all values to JSON-safe types
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    serializable_results = make_serializable(all_results)

    with open(analysis_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"  Full analysis: {analysis_path}")

    # Generate plots
    for demographic in ["gender", "accent", "age"]:
        try:
            plot_wer_by_group(all_results, demographic, args.output_dir)
            print(f"  Plot: wer_by_{demographic}.png")
        except Exception as e:
            print(f"  WARNING: Could not plot {demographic}: {e}")

    try:
        plot_fairness_heatmap(all_results, args.output_dir)
        print(f"  Plot: fairness_heatmap.png")
    except Exception as e:
        print(f"  WARNING: Could not plot heatmap: {e}")

    # Generate LaTeX tables
    for demographic in ["gender", "accent", "age"]:
        try:
            generate_latex_table(all_results, demographic, args.output_dir)
        except Exception as e:
            print(f"  WARNING: Could not generate {demographic} table: {e}")

    # ── Print summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    summary_rows = []
    ordered = get_ordered_models(list(all_results.keys()))
    for model in ordered:
        results = all_results[model]
        row = {"Model": model, "Overall WER (%)": f"{results.get('overall_wer', 0)*100:.2f}"}
        for demo in ["gender", "accent", "age"]:
            if demo in results and "max_min_ratio" in results[demo]:
                row[f"{demo.capitalize()} Max/Min"] = f"{results[demo]['max_min_ratio']:.2f}"
                row[f"{demo.capitalize()} Gap (%)"] = f"{results[demo]['relative_gap_pct']:.1f}"
            else:
                row[f"{demo.capitalize()} Max/Min"] = "--"
                row[f"{demo.capitalize()} Gap (%)"] = "--"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_csv_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary: {summary_csv_path}")

    print(f"\nAll outputs in: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
