#!/usr/bin/env python3
"""
Compute perturbation-specific fairness metrics for ASR experiments.

Computes:
  - WERD (WER Degradation): (WER_perturbed - WER_clean) / WER_clean
  - HER / HERD (Hallucination Error Rate / Degradation)
  - Fairness Gap Amplification: MMR_perturbed / MMR_clean
  - Hallucination classification (silence/masking conditions)
  - Bootstrap 95% CIs (200 resamples)
  - All tables and figure data from README

Usage:
    python scripts/compute_perturbation_metrics.py --dataset fs
    python scripts/compute_perturbation_metrics.py --dataset cv
    python scripts/compute_perturbation_metrics.py --dataset fs --dry-run
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_DIR = Path("/users/PAS2030/srishti/asr_fairness")
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Constants ──────────────────────────────────────────────────────────────
MIN_GROUP_SIZE = 50

MODEL_ORDER = [
    "wav2vec2-large",
    "whisper-small", "whisper-medium", "whisper-large-v3",
    "qwen3-asr-0.6b", "qwen3-asr-1.7b",
    "canary-qwen-2.5b",
    "granite-speech-3.3-2b", "granite-speech-3.3-8b",
]

MODEL_GENERATIONS = {
    "wav2vec2-large": 1,
    "whisper-small": 2, "whisper-medium": 2, "whisper-large-v3": 2,
    "qwen3-asr-0.6b": 3, "qwen3-asr-1.7b": 3,
    "canary-qwen-2.5b": 3,
    "granite-speech-3.3-2b": 3, "granite-speech-3.3-8b": 3,
}

PERTURBATION_ORDER = [
    "clean",
    "snr_20db", "snr_10db", "snr_0db",
    "reverb_0.3s", "reverb_0.6s", "reverb_1.0s",
    "silence_25pct", "silence_50pct", "silence_75pct",
    "mask_10pct", "mask_20pct", "mask_30pct",
]

DEMOGRAPHIC_AXES = {
    "fs": ["ethnicity", "gender", "age", "first_language", "socioeconomic"],
    "cv": ["accent", "gender", "age"],
}

# ── Function words for hallucination classification ────────────────────────
FUNCTION_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "of", "for", "is", "was",
    "are", "were", "and", "or", "but", "that", "this", "it", "he", "she",
    "they", "we", "i", "you", "my", "your", "his", "her", "its", "our",
    "their", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "may", "might", "can",
    "could", "not", "no", "so", "if", "then", "than", "as", "with",
    "by", "from", "up", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "under", "again", "further",
    "there", "here", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "too", "very", "just"
}


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_predictions(dataset: str):
    """Load all prediction CSVs for a dataset, organized by (model, perturbation)."""
    if dataset == "fs":
        results_path = RESULTS_DIR / "fairspeech"
    else:
        results_path = RESULTS_DIR / "commonvoice"

    data = {}
    for csv_path in sorted(results_path.glob("predictions_*.csv")):
        fname = csv_path.stem  # predictions_whisper_large_v3_snr_20db
        # Parse model and perturbation from filename
        parts = fname.replace("predictions_", "")

        # Try to match perturbation suffix
        perturbation = "clean"
        model_part = parts
        for pert in PERTURBATION_ORDER[1:]:  # Skip 'clean'
            pert_suffix = pert.replace(".", "_")
            if parts.endswith(f"_{pert_suffix}"):
                perturbation = pert
                model_part = parts[:-(len(pert_suffix) + 1)]
                break

        # Convert model_part back to model name
        model_name = model_part.replace("_", "-")
        # Fix known patterns
        model_name = model_name.replace("speech-3-3", "speech-3.3")
        model_name = model_name.replace("asr-0-6b", "asr-0.6b")
        model_name = model_name.replace("asr-1-7b", "asr-1.7b")
        model_name = model_name.replace("qwen-2-5b", "qwen-2.5b")

        if model_name not in MODEL_ORDER:
            print(f"  WARNING: Unrecognized model '{model_name}' from {csv_path.name}, skipping")
            continue

        df = pd.read_csv(csv_path)
        df["reference"] = df["reference"].fillna("").astype(str)
        df["hypothesis"] = df["hypothesis"].fillna("").astype(str)
        data[(model_name, perturbation)] = df
        print(f"  Loaded: {model_name} / {perturbation} ({len(df):,} rows)")

    return data


# ═════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_group_wer(df, axis):
    """Compute WER and error decomposition per demographic group."""
    groups = df[df[axis].notna() & (df[axis] != "")].groupby(axis)
    results = {}
    for name, group in groups:
        if len(group) < MIN_GROUP_SIZE:
            continue
        refs = group["reference"].tolist()
        hyps = group["hypothesis"].tolist()

        # Filter empty refs
        valid = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
        if not valid:
            continue
        refs, hyps = zip(*valid)
        refs, hyps = list(refs), list(hyps)

        wer_val = jiwer.wer(refs, hyps)

        # Error decomposition (jiwer 4.x API)
        measures = jiwer.process_words(refs, hyps)
        total_words = measures.hits + measures.substitutions + measures.deletions
        if total_words == 0:
            total_words = 1

        results[name] = {
            "n": len(group),
            "wer": wer_val,
            "substitution_rate": measures.substitutions / total_words,
            "insertion_rate": measures.insertions / total_words,
            "deletion_rate": measures.deletions / total_words,
        }
    return results


def compute_mmr(group_wers: dict) -> float:
    """Max-min ratio across groups."""
    wers = [v["wer"] for v in group_wers.values() if v["wer"] > 0]
    if len(wers) < 2:
        return float("nan")
    return max(wers) / min(wers)


def compute_absolute_gap(group_wers: dict) -> float:
    """WER_worst - WER_best."""
    wers = [v["wer"] for v in group_wers.values()]
    if len(wers) < 2:
        return float("nan")
    return max(wers) - min(wers)


def bootstrap_ci(refs, hyps, n_resamples=200, ci=0.95, seed=42):
    """Bootstrap CI for corpus-level WER."""
    rng = np.random.RandomState(seed)
    n = len(refs)
    boot_wers = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        r = [refs[i] for i in idx]
        h = [hyps[i] for i in idx]
        try:
            boot_wers.append(jiwer.wer(r, h))
        except Exception:
            pass
    if not boot_wers:
        return (float("nan"), float("nan"))
    lower = np.percentile(boot_wers, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_wers, (1 + ci) / 2 * 100)
    return (lower, upper)


# ═════════════════════════════════════════════════════════════════════════════
# HALLUCINATION CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def classify_insertions(ref: str, hyp: str):
    """Classify inserted words using jiwer alignment.

    Categories:
    - repetition: repeats a preceding N-gram
    - syntactic: function word insertion
    - content: semantic fabrication
    """
    if not ref or not hyp:
        return []

    try:
        out = jiwer.process_words(ref, hyp)
    except Exception:
        return []

    insertions = []
    hyp_words = hyp.split()

    for chunk in out.alignments[0]:
        if chunk.type == "insert":
            for pos in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                if pos >= len(hyp_words):
                    break
                word = hyp_words[pos].lower()

                # Check repetition (compare with preceding 20 words)
                preceding = hyp_words[max(0, pos - 20):pos]
                is_repetition = False
                for n in [2, 3, 4]:
                    if pos >= n:
                        ngram = tuple(hyp_words[pos - n + 1:pos + 1])
                        for start in range(max(0, pos - 20), pos - n + 1):
                            candidate = tuple(hyp_words[start:start + n])
                            if ngram == candidate:
                                is_repetition = True
                                break
                    if is_repetition:
                        break

                if is_repetition:
                    category = "repetition"
                elif word in FUNCTION_WORDS:
                    category = "syntactic"
                else:
                    category = "content"

                insertions.append({
                    "word": word,
                    "position": pos,
                    "category": category,
                })

    return insertions


# ═════════════════════════════════════════════════════════════════════════════
# TABLE GENERATORS
# ═════════════════════════════════════════════════════════════════════════════

def generate_wer_by_perturbation_and_group(data, dataset, axis):
    """Table 1/2: WER by (model, perturbation) × demographic groups."""
    rows = []
    for model in MODEL_ORDER:
        for pert in PERTURBATION_ORDER:
            key = (model, pert)
            if key not in data:
                continue
            df = data[key]
            group_wers = compute_group_wer(df, axis)
            if not group_wers:
                continue

            row = {
                "model": model,
                "generation": MODEL_GENERATIONS[model],
                "perturbation": pert,
                "overall_wer": jiwer.wer(
                    df[df["reference"] != ""]["reference"].tolist(),
                    df[df["reference"] != ""]["hypothesis"].tolist()
                ),
                "mmr": compute_mmr(group_wers),
                "absolute_gap": compute_absolute_gap(group_wers),
            }
            for group, metrics in group_wers.items():
                row[f"wer_{group}"] = metrics["wer"]
                row[f"n_{group}"] = metrics["n"]
            rows.append(row)

    return pd.DataFrame(rows)


def generate_insertion_rate_table(data, dataset, axis):
    """Table 3/4: Insertion rate by (model, perturbation) × demographic groups."""
    rows = []
    for model in MODEL_ORDER:
        for pert in PERTURBATION_ORDER:
            key = (model, pert)
            if key not in data:
                continue
            df = data[key]
            group_wers = compute_group_wer(df, axis)
            if not group_wers:
                continue

            row = {
                "model": model,
                "generation": MODEL_GENERATIONS[model],
                "perturbation": pert,
            }
            ins_rates = {}
            for group, metrics in group_wers.items():
                row[f"ins_rate_{group}"] = metrics["insertion_rate"]
                ins_rates[group] = metrics["insertion_rate"]

            nonzero = [v for v in ins_rates.values() if v > 0]
            row["ins_max_min_ratio"] = max(nonzero) / min(nonzero) if len(nonzero) >= 2 else float("nan")
            rows.append(row)

    return pd.DataFrame(rows)


def generate_fairness_gap_amplification(data, dataset, axis):
    """Table 5: Fairness gap amplification."""
    rows = []
    for model in MODEL_ORDER:
        # Get clean baseline
        clean_key = (model, "clean")
        if clean_key not in data:
            continue
        clean_wers = compute_group_wer(data[clean_key], axis)
        if not clean_wers:
            continue
        mmr_clean = compute_mmr(clean_wers)

        for pert in PERTURBATION_ORDER[1:]:
            pert_key = (model, pert)
            if pert_key not in data:
                continue
            pert_wers = compute_group_wer(data[pert_key], axis)
            if not pert_wers:
                continue
            mmr_pert = compute_mmr(pert_wers)

            # WERD for worst and best groups
            common_groups = set(clean_wers.keys()) & set(pert_wers.keys())
            if not common_groups:
                continue

            werd_by_group = {}
            for g in common_groups:
                wer_c = clean_wers[g]["wer"]
                wer_p = pert_wers[g]["wer"]
                werd_by_group[g] = (wer_p - wer_c) / wer_c if wer_c > 0 else float("nan")

            valid_werds = {g: w for g, w in werd_by_group.items() if not np.isnan(w)}
            if not valid_werds:
                continue

            worst_group = max(valid_werds, key=valid_werds.get)
            best_group = min(valid_werds, key=valid_werds.get)

            rows.append({
                "model": model,
                "generation": MODEL_GENERATIONS[model],
                "perturbation_type": pert.split("_")[0],
                "perturbation_level": pert,
                "mmr_clean": mmr_clean,
                "mmr_perturbed": mmr_pert,
                "amplification_ratio": mmr_pert / mmr_clean if mmr_clean > 0 else float("nan"),
                "werd_worst_group": valid_werds[worst_group],
                "werd_worst_group_name": worst_group,
                "werd_best_group": valid_werds[best_group],
                "werd_best_group_name": best_group,
                "werd_ratio": valid_werds[worst_group] / valid_werds[best_group]
                    if valid_werds[best_group] != 0 else float("nan"),
            })

    return pd.DataFrame(rows)


def generate_hallucination_classification_table(data, dataset, pert_type_prefix):
    """Table 6/7: Hallucination classification for silence/masking conditions."""
    rows = []
    for model in MODEL_ORDER:
        for pert in PERTURBATION_ORDER:
            if not pert.startswith(pert_type_prefix):
                continue
            key = (model, pert)
            if key not in data:
                continue
            df = data[key]

            total_insertions = 0
            categories = {"repetition": 0, "syntactic": 0, "content": 0}

            for _, row in df.iterrows():
                ref = str(row.get("reference", ""))
                hyp = str(row.get("hypothesis", ""))
                if not ref or not hyp:
                    continue
                ins = classify_insertions(ref, hyp)
                total_insertions += len(ins)
                for i in ins:
                    categories[i["category"]] += 1

            total = sum(categories.values())
            rows.append({
                "model": model,
                "generation": MODEL_GENERATIONS[model],
                "perturbation": pert,
                "total_insertions": total_insertions,
                "pct_repetition": categories["repetition"] / total * 100 if total > 0 else 0,
                "pct_syntactic": categories["syntactic"] / total * 100 if total > 0 else 0,
                "pct_content": categories["content"] / total * 100 if total > 0 else 0,
            })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE DATA GENERATORS
# ═════════════════════════════════════════════════════════════════════════════

def generate_werd_by_generation_and_snr(data, dataset, axis):
    """Figure 1: WERD at each SNR level by generation."""
    snr_perts = ["snr_20db", "snr_10db", "snr_0db"]
    rows = []

    for model in MODEL_ORDER:
        clean_key = (model, "clean")
        if clean_key not in data:
            continue
        clean_wers = compute_group_wer(data[clean_key], axis)
        if not clean_wers:
            continue

        for pert in snr_perts:
            pert_key = (model, pert)
            if pert_key not in data:
                continue
            pert_wers = compute_group_wer(data[pert_key], axis)

            common_groups = set(clean_wers.keys()) & set(pert_wers.keys())
            for g in common_groups:
                wer_c = clean_wers[g]["wer"]
                wer_p = pert_wers[g]["wer"]
                werd = (wer_p - wer_c) / wer_c if wer_c > 0 else float("nan")
                rows.append({
                    "model": model,
                    "generation": MODEL_GENERATIONS[model],
                    "perturbation": pert,
                    "group": g,
                    "wer_clean": wer_c,
                    "wer_perturbed": wer_p,
                    "werd": werd,
                })

    return pd.DataFrame(rows)


def generate_fairness_amplification_heatmap(data, dataset, axis):
    """Figure 2: Heatmap of MMR_perturbed / MMR_clean."""
    rows = []
    for model in MODEL_ORDER:
        clean_key = (model, "clean")
        if clean_key not in data:
            continue
        clean_wers = compute_group_wer(data[clean_key], axis)
        if not clean_wers:
            continue
        mmr_clean = compute_mmr(clean_wers)

        for pert in PERTURBATION_ORDER[1:]:
            pert_key = (model, pert)
            if pert_key not in data:
                continue
            pert_wers = compute_group_wer(data[pert_key], axis)
            if not pert_wers:
                continue
            mmr_pert = compute_mmr(pert_wers)

            rows.append({
                "model": model,
                "perturbation": pert,
                "mmr_clean": mmr_clean,
                "mmr_perturbed": mmr_pert,
                "amplification": mmr_pert / mmr_clean if mmr_clean > 0 else float("nan"),
            })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compute perturbation fairness metrics")
    parser.add_argument("--dataset", required=True, choices=["cv", "fs"],
                        help="Dataset: cv (Common Voice) or fs (Fair-Speech)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only load data and report counts, don't compute metrics")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Perturbation Fairness Metrics — {args.dataset.upper()}")
    print(f"{'='*60}\n")

    # Load all predictions
    data = load_predictions(args.dataset)
    print(f"\nLoaded {len(data)} (model, perturbation) pairs\n")

    if args.dry_run:
        print("Dry run — exiting without computing metrics.")
        return

    # Ensure output dirs
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    axes = DEMOGRAPHIC_AXES[args.dataset]
    ds_suffix = args.dataset

    # ── Tables ──────────────────────────────────────────────────────────
    for axis in axes:
        print(f"\n{'─'*40}")
        print(f"Computing metrics for axis: {axis}")
        print(f"{'─'*40}")

        # Table 1/2: WER by perturbation × group
        df_wer = generate_wer_by_perturbation_and_group(data, args.dataset, axis)
        if len(df_wer) > 0:
            path = TABLES_DIR / f"table_wer_by_perturbation_and_{axis}_{ds_suffix}.csv"
            df_wer.to_csv(path, index=False)
            print(f"  Saved: {path.name} ({len(df_wer)} rows)")

        # Table 3/4: Insertion rate
        df_ins = generate_insertion_rate_table(data, args.dataset, axis)
        if len(df_ins) > 0:
            path = TABLES_DIR / f"table_insertion_rate_by_perturbation_and_{axis}_{ds_suffix}.csv"
            df_ins.to_csv(path, index=False)
            print(f"  Saved: {path.name} ({len(df_ins)} rows)")

        # Table 5: Fairness gap amplification
        df_amp = generate_fairness_gap_amplification(data, args.dataset, axis)
        if len(df_amp) > 0:
            path = TABLES_DIR / f"table_fairness_gap_amplification_{axis}_{ds_suffix}.csv"
            df_amp.to_csv(path, index=False)
            print(f"  Saved: {path.name} ({len(df_amp)} rows)")

        # Figure 1: WERD by generation and SNR
        df_werd = generate_werd_by_generation_and_snr(data, args.dataset, axis)
        if len(df_werd) > 0:
            path = FIGURES_DIR / f"fig_werd_by_generation_and_snr_{axis}_{ds_suffix}.csv"
            df_werd.to_csv(path, index=False)
            print(f"  Saved: {path.name} ({len(df_werd)} rows)")

        # Figure 2: Amplification heatmap
        df_heat = generate_fairness_amplification_heatmap(data, args.dataset, axis)
        if len(df_heat) > 0:
            path = FIGURES_DIR / f"fig_fairness_amplification_heatmap_{axis}_{ds_suffix}.csv"
            df_heat.to_csv(path, index=False)
            print(f"  Saved: {path.name} ({len(df_heat)} rows)")

    # ── Hallucination classification (silence + masking only) ───────────
    print(f"\n{'─'*40}")
    print("Hallucination classification (silence conditions)...")
    df_hall_silence = generate_hallucination_classification_table(data, args.dataset, "silence")
    if len(df_hall_silence) > 0:
        path = TABLES_DIR / f"table_hallucination_classification_silence_{ds_suffix}.csv"
        df_hall_silence.to_csv(path, index=False)
        print(f"  Saved: {path.name}")

    print("Hallucination classification (masking conditions)...")
    df_hall_mask = generate_hallucination_classification_table(data, args.dataset, "mask")
    if len(df_hall_mask) > 0:
        path = TABLES_DIR / f"table_hallucination_classification_masking_{ds_suffix}.csv"
        df_hall_mask.to_csv(path, index=False)
        print(f"  Saved: {path.name}")

    print(f"\n{'='*60}")
    print("Done! Tables in: results/tables/")
    print("Figure data in: results/figures/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
