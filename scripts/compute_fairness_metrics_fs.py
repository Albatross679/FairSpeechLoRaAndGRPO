#!/usr/bin/env python3
"""
Compute fairness metrics from ASR inference predictions on Fair-Speech.

Handles 5 demographic axes: ethnicity, gender, age, first_language (L1), socioeconomic.
Computes WER, MMR, gap%, error decomposition, and bootstrap CIs per group.

Usage:
    python scripts/compute_fairness_metrics_fs.py \
        --results_dir results/fairspeech/ \
        --output_dir results/fairspeech/analysis/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import jiwer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

MIN_GROUP_SIZE = 50
DEMOGRAPHIC_AXES = ["ethnicity", "gender", "age", "first_language", "l1_group", "socioeconomic"]

# ── Publication color palette ───────────────────────────────────────────────
# Distinct, readable on white, color-blind-friendly
MODEL_ORDER = [
    "wav2vec2-large",        # Gen 1
    "whisper-small",         # Gen 2
    "whisper-medium",        # Gen 2
    "whisper-large-v3",      # Gen 2
    "qwen3-asr-0.6b",       # Gen 3
    "qwen3-asr-1.7b",       # Gen 3
    "granite-speech-3.3-2b", # Gen 3
    "granite-speech-3.3-8b", # Gen 3
    "canary-qwen-2.5b",     # Gen 3
]

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

GEN_COLORS = {1: "#2E86AB", 2: "#C73E1D", 3: "#44AF69"}
GEN_LABELS = {1: "Gen 1 (CTC)", 2: "Gen 2 (Enc-Dec)", 3: "Gen 3 (LLM)"}

# Logical ordering for x-axis groups
ETHNICITY_ORDER = ["White", "Black/AA", "Hispanic", "Asian", "Native American", "Pacific Islander", "Middle Eastern"]
AGE_ORDER = ["18-22", "23-30", "31-45", "46-65"]
GENDER_ORDER = ["female", "male"]
SES_ORDER = ["Low", "Medium", "Affluent"]
L1_GROUP_ORDER = ["English", "Spanish", "Mandarin", "Hindi", "Other"]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ASR fairness metrics for Fair-Speech")
    parser.add_argument("--results_dir", type=str,
                        default="/users/PAS2030/srishti/asr_fairness/results/fairspeech")
    parser.add_argument("--output_dir", type=str,
                        default="/users/PAS2030/srishti/asr_fairness/results/fairspeech/analysis")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    return parser.parse_args()


# ── WER + error decomposition ──────────────────────────────────────────────
def compute_group_wer(df):
    """Compute aggregate WER and error decomposition for a group."""
    refs = df["reference"].fillna("").astype(str).tolist()
    hyps = df["hypothesis"].fillna("").astype(str).tolist()
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if not valid:
        return {"wer": None, "n_utterances": 0}

    refs_clean, hyps_clean = zip(*valid)
    refs_clean, hyps_clean = list(refs_clean), list(hyps_clean)

    wer_val = jiwer.wer(refs_clean, hyps_clean)
    output = jiwer.process_words(refs_clean, hyps_clean)
    total_ref = output.substitutions + output.deletions + output.hits
    total_errors = output.substitutions + output.insertions + output.deletions

    return {
        "wer": wer_val,
        "n_utterances": len(refs_clean),
        "substitution_rate": output.substitutions / total_ref if total_ref > 0 else 0,
        "insertion_rate": output.insertions / total_ref if total_ref > 0 else 0,
        "deletion_rate": output.deletions / total_ref if total_ref > 0 else 0,
        "substitutions": output.substitutions,
        "insertions": output.insertions,
        "deletions": output.deletions,
        "hits": output.hits,
        "sub_pct_of_errors": output.substitutions / max(1, total_errors) * 100,
        "ins_pct_of_errors": output.insertions / max(1, total_errors) * 100,
        "del_pct_of_errors": output.deletions / max(1, total_errors) * 100,
    }


def bootstrap_wer(df, n_bootstrap=1000):
    """Compute bootstrap 95% CI for WER."""
    refs = df["reference"].fillna("").astype(str).tolist()
    hyps = df["hypothesis"].fillna("").astype(str).tolist()
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if len(valid) < 10:
        return None, None, None

    refs_clean, hyps_clean = zip(*valid)
    refs_clean, hyps_clean = list(refs_clean), list(hyps_clean)
    n = len(refs_clean)
    rng = np.random.RandomState(42)
    wers = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            wers.append(jiwer.wer([refs_clean[i] for i in idx], [hyps_clean[i] for i in idx]))
        except Exception:
            continue
    if not wers:
        return None, None, None
    return np.mean(wers), np.percentile(wers, 2.5), np.percentile(wers, 97.5)


def compute_fairness_metrics(group_wers):
    """MMR, gap%, std from per-group WER dict."""
    valid = {k: v for k, v in group_wers.items()
             if v.get("wer") is not None and v.get("n_utterances", 0) >= MIN_GROUP_SIZE}
    if len(valid) < 2:
        return {"error": "Too few valid groups"}

    wer_values = {k: v["wer"] for k, v in valid.items()}
    wer_list = list(wer_values.values())
    best = min(wer_values, key=wer_values.get)
    worst = max(wer_values, key=wer_values.get)

    return {
        "group_wers": {k: {"wer": v["wer"], "n": v["n_utterances"]} for k, v in valid.items()},
        "best_group": best,
        "best_wer": wer_values[best],
        "worst_group": worst,
        "worst_wer": wer_values[worst],
        "mmr": max(wer_list) / min(wer_list) if min(wer_list) > 0 else float("inf"),
        "relative_gap_pct": (wer_values[worst] - wer_values[best]) / wer_values[best] * 100
            if wer_values[best] > 0 else float("inf"),
        "wer_std": float(np.std(wer_list)),
        "wer_range": max(wer_list) - min(wer_list),
    }


def get_group_order(axis):
    """Return the logical ordering for a demographic axis."""
    return {
        "ethnicity": ETHNICITY_ORDER,
        "age": AGE_ORDER,
        "gender": GENDER_ORDER,
        "socioeconomic": SES_ORDER,
        "l1_group": L1_GROUP_ORDER,
    }.get(axis, None)


def analyze_model(df, model_name, n_bootstrap=1000):
    """Run complete fairness analysis for one model on Fair-Speech."""
    results = {}
    overall = compute_group_wer(df)
    results["overall_wer"] = overall["wer"]
    results["n_utterances"] = overall["n_utterances"]

    for axis in DEMOGRAPHIC_AXES:
        if axis not in df.columns:
            continue
        df_axis = df[df[axis].notna() & (df[axis] != "")]
        if len(df_axis) == 0:
            continue

        print(f"\n  Analyzing {axis}...")
        group_wers = {}
        for name, group in df_axis.groupby(axis):
            if len(group) < MIN_GROUP_SIZE:
                print(f"    {name}: {len(group)} samples (< {MIN_GROUP_SIZE}, skipping)")
                continue
            wer_info = compute_group_wer(group)
            mean_wer, ci_lo, ci_hi = bootstrap_wer(group, n_bootstrap)
            wer_info["bootstrap_ci_low"] = ci_lo
            wer_info["bootstrap_ci_high"] = ci_hi
            group_wers[name] = wer_info
            ci_str = f"[{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]" if ci_lo is not None else "[N/A]"
            print(f"    {name:25s}: WER={wer_info['wer']*100:.2f}% {ci_str} (n={len(group):,})")

        fairness = compute_fairness_metrics(group_wers)
        fairness["error_decomposition"] = {k: {
            "sub_rate": v["substitution_rate"],
            "ins_rate": v["insertion_rate"],
            "del_rate": v["deletion_rate"],
            "sub_pct": v["sub_pct_of_errors"],
            "ins_pct": v["ins_pct_of_errors"],
            "del_pct": v["del_pct_of_errors"],
        } for k, v in group_wers.items() if v.get("wer") is not None}
        results[axis] = fairness

    return results


# ── H1-specific: Black/AA vs White gap ─────────────────────────────────────
def compute_h1_gap(all_results):
    """Compute Black/AA vs White relative gap per model."""
    print(f"\n{'='*60}")
    print("H1-c: Black/AA vs White Gap Analysis")
    print(f"{'='*60}")

    gaps = {}
    for model, results in all_results.items():
        eth = results.get("ethnicity", {})
        gw = eth.get("group_wers", {})
        if "Black/AA" in gw and "White" in gw:
            black_wer = gw["Black/AA"]["wer"]
            white_wer = gw["White"]["wer"]
            gap = (black_wer - white_wer) / white_wer * 100 if white_wer > 0 else float("inf")
            gaps[model] = {"black_wer": black_wer, "white_wer": white_wer, "gap_pct": gap}
            print(f"  {model:28s}: Black/AA={black_wer*100:.2f}%, White={white_wer*100:.2f}%, Gap={gap:+.1f}%")

    return gaps


# ── JSON serialization helper ──────────────────────────────────────────────
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


# ── LaTeX table generation ─────────────────────────────────────────────────
def generate_latex_table(all_results, axis, output_dir, dataset_label="Fair-Speech"):
    """Generate LaTeX table for a demographic axis."""
    models = [m for m in MODEL_ORDER if m in all_results]
    order = get_group_order(axis)

    all_groups = set()
    for m in models:
        if axis in all_results[m] and "group_wers" in all_results[m][axis]:
            all_groups.update(all_results[m][axis]["group_wers"].keys())
    groups = [g for g in (order or sorted(all_groups)) if g in all_groups]

    header = "Model"
    for g in groups:
        header += f" & {g}"
    header += r" & MMR & Gap (\%) \\"

    rows = []
    for model in models:
        row = model.replace("_", "-")
        gw = all_results[model].get(axis, {}).get("group_wers", {})
        for g in groups:
            if g in gw:
                row += f" & {gw[g]['wer']*100:.1f}"
            else:
                row += " & --"
        mmr = all_results[model].get(axis, {}).get("mmr", 0)
        gap = all_results[model].get(axis, {}).get("relative_gap_pct", 0)
        row += f" & {mmr:.2f} & {gap:.1f}"
        row += r" \\"
        rows.append(row)

    n_cols = 1 + len(groups) + 2
    latex = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        r"\caption{WER (\%) by " + axis.replace("_", " ") + f" across ASR models on {dataset_label}." + "}\n"
        r"\label{tab:fs_wer_" + axis + "}\n"
        r"\resizebox{\textwidth}{!}{" + "\n"
        r"\begin{tabular}{l" + "r" * (len(groups) + 2) + "}\n"
        r"\toprule" + "\n"
        + header + "\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"}" + "\n"
        r"\end{table}"
    )

    path = os.path.join(output_dir, f"table_fs_{axis}.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"  Saved LaTeX: {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fair-Speech Fairness Analysis")
    print(f"{'='*60}\n")

    # Auto-detect prediction files
    pred_files = sorted([
        f for f in os.listdir(args.results_dir)
        if f.startswith("predictions_") and f.endswith(".csv") and "checkpoint" not in f
    ])
    if not pred_files:
        print(f"ERROR: No prediction files found in {args.results_dir}")
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
        for col in DEMOGRAPHIC_AXES:
            if col in df.columns:
                df[col] = df[col].fillna("")

        print(f"  Total predictions: {len(df):,}")

        results = analyze_model(df, model_name, args.n_bootstrap)

        # Get gen info from model name
        meta_path = os.path.join(args.results_dir, f"meta_{model_name}.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                results["model_info"] = json.load(f).get("model_info", {})
        all_results[display_name] = results

    # ── H1 gap analysis ────────────────────────────────────────────
    h1_gaps = compute_h1_gap(all_results)

    # ── Save complete analysis ─────────────────────────────────────
    analysis_path = os.path.join(args.output_dir, "full_analysis_fs.json")
    with open(analysis_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n  Full analysis: {analysis_path}")

    # ── H1 gaps JSON ──────────────────────────────────────────────
    gaps_path = os.path.join(args.output_dir, "h1_black_white_gap.json")
    with open(gaps_path, "w") as f:
        json.dump(make_serializable(h1_gaps), f, indent=2)

    # ── LaTeX tables ──────────────────────────────────────────────
    for axis in DEMOGRAPHIC_AXES:
        try:
            generate_latex_table(all_results, axis, args.output_dir)
        except Exception as e:
            print(f"  WARNING: Could not generate {axis} table: {e}")

    # ── Summary CSV ───────────────────────────────────────────────
    summary_rows = []
    for model, results in all_results.items():
        row = {"Model": model, "Overall WER (%)": f"{results.get('overall_wer', 0)*100:.2f}"}
        for axis in DEMOGRAPHIC_AXES:
            if axis in results and "mmr" in results[axis]:
                row[f"{axis} MMR"] = f"{results[axis]['mmr']:.2f}"
                row[f"{axis} Gap (%)"] = f"{results[axis]['relative_gap_pct']:.1f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, "summary_fs.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
