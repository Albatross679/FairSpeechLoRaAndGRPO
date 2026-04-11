#!/usr/bin/env python3
"""
Whisper hallucination analysis — extraction, categorization, cross-model comparison.

Tests H3: Whisper scaling pathology on accented and minority speech.

Usage:
    # For Common Voice (accent-based):
    python scripts/analysis/whisper_hallucination_analysis.py --dataset cv

    # For Fair-Speech (ethnicity-based):
    python scripts/analysis/whisper_hallucination_analysis.py --dataset fs
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import jiwer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
CV_RESULTS_DIR = "/users/PAS2030/srishti/asr_fairness/results/commonvoice"
FS_RESULTS_DIR = "/users/PAS2030/srishti/asr_fairness/results/fairspeech"
OUTPUT_DIR = "/users/PAS2030/srishti/asr_fairness/results/hallucination_analysis"

FUNCTION_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in',
    'for', 'on', 'at', 'by', 'with', 'and', 'or', 'but', 'that',
    'this', 'it', 'he', 'she', 'they', 'we', 'i', 'you', 'my',
    'your', 'his', 'her', 'its', 'our', 'their', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'can', 'may', 'might', 'shall',
    'not', 'no', 'so', 'if', 'than', 'then', 'just', 'also',
}

# Publication colors — distinct per model, consistent with other scripts
MODEL_COLORS = {
    "wav2vec2-large":        "#2E86AB",
    "whisper-small":         "#F18F01",
    "whisper-medium":        "#3B7A57",
    "whisper-large-v3":      "#A23B72",
    "qwen3-asr-0.6b":       "#FFFF00",
    "qwen3-asr-1.7b":       "#44AF69",
    "granite-speech-3.3-2b": "#6B4C9A",
    "granite-speech-3.3-8b": "#E84855",
    "canary-qwen-2.5b":     "#D4A574",
}

MODEL_ORDER = [
    "wav2vec2-large",
    "whisper-small",
    "whisper-medium",
    "whisper-large-v3",
    "qwen3-asr-0.6b",
    "qwen3-asr-1.7b",
    "canary-qwen-2.5b",
    "granite-speech-3.3-2b",
    "granite-speech-3.3-8b",
]

WHISPER_MODELS = ["whisper_small", "whisper_medium", "whisper_large_v3"]
GEN3_MODELS = ["qwen3_asr_0.6b", "qwen3_asr_1.7b", "granite_speech_3.3_2b",
               "canary_qwen_2.5b", "granite_speech_3.3_8b"]


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper hallucination analysis")
    parser.add_argument("--dataset", type=str, choices=["cv", "fs", "both"], default="both")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()


def extract_and_categorize_insertions(ref, hyp):
    """Extract inserted words with context and auto-categorize."""
    if not ref.strip() or not hyp.strip():
        return []

    try:
        output = jiwer.process_words(ref, hyp)
    except Exception:
        return []

    insertions = []
    hyp_words = hyp.split()
    ref_words = ref.split()

    for chunk in output.alignments[0]:
        if chunk.type == 'insert':
            for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                if i >= len(hyp_words):
                    continue
                word = hyp_words[i]
                ctx_before = ' '.join(hyp_words[max(0, i-3):i])
                ctx_after = ' '.join(hyp_words[i+1:i+4])

                # Auto-categorize
                if word.lower() in FUNCTION_WORDS:
                    category = 'syntactic_completion'
                elif hyp_words[:i].count(word) > 0:
                    category = 'repetition'
                else:
                    category = 'content_hallucination'

                insertions.append({
                    'word': word,
                    'word_lower': word.lower(),
                    'category': category,
                    'context': f"...{ctx_before} **[{word}]** {ctx_after}...",
                    'ref': ref,
                    'hyp': hyp,
                })
    return insertions


def analyze_model_insertions(df, model_name, group_col, groups_of_interest=None):
    """Analyze insertions for a model across demographic groups."""
    results = {}

    df_labeled = df[df[group_col].notna() & (df[group_col] != "")]
    if groups_of_interest:
        df_labeled = df_labeled[df_labeled[group_col].isin(groups_of_interest)]

    for group_name, group_df in df_labeled.groupby(group_col):
        if len(group_df) < 50:
            continue

        all_insertions = []
        total_ref_words = 0
        total_errors = 0

        for _, row in group_df.iterrows():
            ref = str(row.get("reference", "")).strip()
            hyp = str(row.get("hypothesis", "")).strip()
            if not ref or not hyp:
                continue

            total_ref_words += len(ref.split())
            ins_list = extract_and_categorize_insertions(ref, hyp)
            all_insertions.extend(ins_list)

        if total_ref_words == 0:
            continue

        cat_counts = Counter(ins['category'] for ins in all_insertions)
        word_freq = Counter(ins['word_lower'] for ins in all_insertions)

        results[group_name] = {
            "n_utterances": len(group_df),
            "total_ref_words": total_ref_words,
            "total_insertions": len(all_insertions),
            "insertion_rate": len(all_insertions) / total_ref_words,
            "category_counts": dict(cat_counts),
            "category_pcts": {
                k: v / max(1, len(all_insertions)) * 100
                for k, v in cat_counts.items()
            },
            "top_20_words": word_freq.most_common(20),
            # Store a few examples for manual review
            "examples": [
                {"word": ins["word"], "category": ins["category"],
                 "context": ins["context"], "ref": ins["ref"][:100], "hyp": ins["hyp"][:100]}
                for ins in all_insertions[:20]
            ],
        }

    return results


def cross_model_comparison(results_dir, group_col, groups_of_interest, dataset_name):
    """Compare insertion profiles across models on the same utterances."""
    print(f"\n{'='*60}")
    print(f"Cross-Model Insertion Comparison ({dataset_name})")
    print(f"{'='*60}")

    all_model_results = {}

    pred_files = sorted([
        f for f in os.listdir(results_dir)
        if f.startswith("predictions_") and f.endswith(".csv") and "checkpoint" not in f
    ])

    for pred_file in pred_files:
        model_name = pred_file.replace("predictions_", "").replace(".csv", "")
        display = model_name.replace("_", "-")

        try:
            df = pd.read_csv(os.path.join(results_dir, pred_file))
        except Exception as e:
            print(f"  WARNING: Could not read {pred_file}: {e}")
            continue

        if group_col not in df.columns:
            continue

        df[group_col] = df[group_col].fillna("")
        res = analyze_model_insertions(df, model_name, group_col, groups_of_interest)
        if res:
            all_model_results[display] = res

    # Print comparison table
    for group in groups_of_interest:
        print(f"\n  {group_col}: {group}")
        print(f"  {'Model':<28s} | {'Ins Rate':>8s} | {'Syntactic':>9s} | {'Content':>9s} | {'Repeat':>8s} | {'Top 3 words'}")
        print(f"  {'-'*28}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*30}")

        for model in all_model_results:
            if group in all_model_results[model]:
                r = all_model_results[model][group]
                cats = r["category_pcts"]
                top3 = ", ".join(w for w, _ in r["top_20_words"][:3])
                print(f"  {model:<28s} | {r['insertion_rate']*100:>7.3f}% | "
                      f"{cats.get('syntactic_completion', 0):>8.1f}% | "
                      f"{cats.get('content_hallucination', 0):>8.1f}% | "
                      f"{cats.get('repetition', 0):>7.1f}% | {top3}")

    return all_model_results


def plot_insertion_rate_comparison(all_results, group_col, groups, output_dir, dataset_name):
    """Plot insertion rates across models for given groups."""
    # Ensure PDF text is crisp
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(14, 6))

    # Order models by generation/size, consistent with other scripts
    models_ordered = [m for m in MODEL_ORDER if m in all_results]

    x = np.arange(len(groups))
    width = 0.8 / max(1, len(models_ordered))

    for i, model in enumerate(models_ordered):
        rates = []
        for g in groups:
            if g in all_results[model]:
                rates.append(all_results[model][g]["insertion_rate"] * 100)
            else:
                rates.append(0)

        color = MODEL_COLORS.get(model, "#999999")
        ax.bar(x + i * width, rates, width, label=model, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=16)
    ax.set_ylabel("Insertion Rate (%)", fontsize=16)
    ax.set_title(f"Insertion Rate by {group_col.replace('_', ' ').title()} ({dataset_name})", fontsize=18)
    ax.set_xticks(x + width * (len(models_ordered) - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        path = os.path.join(output_dir, f"insertion_rate_{dataset_name}_{group_col}.{fmt}")
        plt.savefig(path, dpi=300 if fmt == "png" else None)
    plt.close()
    print(f"  Plot saved: insertion_rate_{dataset_name}_{group_col}")


def plot_hallucination_categories(all_results, output_dir, dataset_name):
    """Stacked bar chart of hallucination categories per model."""
    # Ensure PDF text is crisp
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Use consistent model ordering (Gen 1 → Gen 2 → Gen 3)
    models = [m for m in MODEL_ORDER if m in all_results]
    categories = ["syntactic_completion", "content_hallucination", "repetition"]
    cat_colors = {"syntactic_completion": "#2E86AB", "content_hallucination": "#C73E1D", "repetition": "#F18F01"}
    cat_labels = {"syntactic_completion": "Syntactic", "content_hallucination": "Content", "repetition": "Repetition"}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bar_width = 0.6

    # Aggregate across all groups for each model
    bottoms = np.zeros(len(models))
    for cat in categories:
        vals = []
        for model in models:
            total_ins = sum(r.get("total_insertions", 0) for r in all_results[model].values())
            cat_count = sum(r.get("category_counts", {}).get(cat, 0) for r in all_results[model].values())
            vals.append(cat_count / max(1, total_ins) * 100)
        ax.bar(x, vals, bar_width, bottom=bottoms, label=cat_labels[cat],
               color=cat_colors[cat], edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_ylabel("% of Insertions", fontsize=16)
    ax.set_title(f"Hallucination Category Distribution ({dataset_name})", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        plt.savefig(os.path.join(output_dir, f"hallucination_categories_{dataset_name}.{fmt}"),
                    dpi=300 if fmt == "png" else None)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_analysis = {}

    if args.dataset in ("cv", "both"):
        print("\n" + "="*60)
        print("Analyzing Common Voice (accent-based)")
        print("="*60)
        accents_of_interest = ["us", "indian", "england", "canada", "australia", "african"]
        cv_results = cross_model_comparison(CV_RESULTS_DIR, "accent", accents_of_interest, "CommonVoice")
        all_analysis["cv"] = cv_results

        if cv_results:
            plot_insertion_rate_comparison(cv_results, "accent", accents_of_interest, args.output_dir, "CV")
            plot_hallucination_categories(cv_results, args.output_dir, "CV")

    if args.dataset in ("fs", "both"):
        print("\n" + "="*60)
        print("Analyzing Fair-Speech (ethnicity-based)")
        print("="*60)
        ethnicities = ["White", "Black/AA", "Hispanic", "Asian", "Native American", "Pacific Islander", "Middle Eastern"]
        fs_results = cross_model_comparison(FS_RESULTS_DIR, "ethnicity", ethnicities, "FairSpeech")
        all_analysis["fs"] = fs_results

        if fs_results:
            plot_insertion_rate_comparison(fs_results, "ethnicity", ethnicities, args.output_dir, "FS")
            plot_hallucination_categories(fs_results, args.output_dir, "FS")

    # ── Save complete analysis ─────────────────────────────────────
    # Convert Counter objects and other non-serializable types
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

    output_path = os.path.join(args.output_dir, "hallucination_analysis.json")
    with open(output_path, "w") as f:
        json.dump(make_serializable(all_analysis), f, indent=2)
    print(f"\nSaved: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
