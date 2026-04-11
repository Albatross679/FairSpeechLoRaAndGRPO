#!/usr/bin/env python3
"""
Error-type decomposition by accent — tests Hypothesis 3.

Computes substitution/insertion/deletion rates per accent group per model
from existing prediction CSVs. No GPU needed.

Tests H3: "Gen-3 models will show higher insertion and substitution rates
for accents that create systematic local ambiguities."
"""

import json
import os
import sys

import jiwer
import numpy as np
import pandas as pd

RESULTS_DIR = "/users/PAS2030/srishti/asr_fairness/results/commonvoice"
OUTPUT_DIR = "/users/PAS2030/srishti/asr_fairness/results/commonvoice/analysis"
MIN_GROUP_SIZE = 50


def compute_error_decomposition(refs, hyps):
    """Compute substitution, insertion, deletion rates from ref/hyp lists."""
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if len(valid) < 10:
        return None

    refs_clean, hyps_clean = zip(*valid)
    refs_clean, hyps_clean = list(refs_clean), list(hyps_clean)

    output = jiwer.process_words(refs_clean, hyps_clean)
    total_ref_words = output.substitutions + output.deletions + output.hits

    if total_ref_words == 0:
        return None

    return {
        "n_utterances": len(refs_clean),
        "total_ref_words": total_ref_words,
        "wer": (output.substitutions + output.insertions + output.deletions) / total_ref_words,
        "substitution_rate": output.substitutions / total_ref_words,
        "insertion_rate": output.insertions / total_ref_words,
        "deletion_rate": output.deletions / total_ref_words,
        "substitutions": output.substitutions,
        "insertions": output.insertions,
        "deletions": output.deletions,
        "hits": output.hits,
        "sub_pct_of_errors": output.substitutions / max(1, output.substitutions + output.insertions + output.deletions) * 100,
        "ins_pct_of_errors": output.insertions / max(1, output.substitutions + output.insertions + output.deletions) * 100,
        "del_pct_of_errors": output.deletions / max(1, output.substitutions + output.insertions + output.deletions) * 100,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find prediction files
    pred_files = sorted([
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("predictions_") and f.endswith(".csv")
    ])

    if not pred_files:
        print("No prediction files found!")
        sys.exit(1)

    print(f"Found {len(pred_files)} prediction files")
    print("=" * 80)

    all_results = {}

    for pred_file in pred_files:
        model_name = pred_file.replace("predictions_", "").replace(".csv", "").replace("_", "-")
        df = pd.read_csv(os.path.join(RESULTS_DIR, pred_file))

        # Fill NaN
        for col in ["reference", "hypothesis", "accent", "age", "gender"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        print(f"\n{'─' * 60}")
        print(f"Model: {model_name} ({len(df):,} utterances)")
        print(f"{'─' * 60}")

        model_results = {}

        for axis in ["accent", "age", "gender"]:
            df_labeled = df[df[axis] != ""]
            if len(df_labeled) == 0:
                continue

            print(f"\n  {axis.upper()} error decomposition:")
            print(f"  {'Group':<15s} | {'N':>5s} | {'WER':>6s} | {'Sub%':>6s} | {'Ins%':>6s} | {'Del%':>6s} | {'Sub/Err':>7s} | {'Ins/Err':>7s} | {'Del/Err':>7s}")
            print(f"  {'-'*15}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

            axis_results = {}
            groups = df_labeled.groupby(axis)

            for name, group in sorted(groups, key=lambda x: -len(x[1])):
                if len(group) < MIN_GROUP_SIZE:
                    continue

                refs = group["reference"].tolist()
                hyps = group["hypothesis"].tolist()
                decomp = compute_error_decomposition(refs, hyps)

                if decomp is None:
                    continue

                axis_results[name] = decomp
                print(f"  {name:<15s} | {decomp['n_utterances']:>5d} | {decomp['wer']*100:>5.1f}% | {decomp['substitution_rate']*100:>5.2f}% | {decomp['insertion_rate']*100:>5.2f}% | {decomp['deletion_rate']*100:>5.2f}% | {decomp['sub_pct_of_errors']:>6.1f}% | {decomp['ins_pct_of_errors']:>6.1f}% | {decomp['del_pct_of_errors']:>6.1f}%")

            model_results[axis] = axis_results

        all_results[model_name] = model_results

    # ── H3 analysis: Compare error profiles across generations ──────
    print(f"\n\n{'=' * 80}")
    print("HYPOTHESIS 3 ANALYSIS: Error-type profiles by accent × generation")
    print(f"{'=' * 80}")

    accents_of_interest = ["us", "indian", "england", "canada", "australia", "african"]
    models = list(all_results.keys())

    print(f"\nKey question: Do Gen-3 models show higher insertion/substitution rates")
    print(f"for accented speech compared to Gen-2 models with similar overall WER?\n")

    for accent in accents_of_interest:
        print(f"\n  Accent: {accent}")
        print(f"  {'Model':<20s} | {'WER':>6s} | {'Sub rate':>8s} | {'Ins rate':>8s} | {'Del rate':>8s} | {'Sub%err':>7s} | {'Ins%err':>7s} | {'Del%err':>7s}")
        print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        for model in models:
            if "accent" in all_results[model] and accent in all_results[model]["accent"]:
                d = all_results[model]["accent"][accent]
                print(f"  {model:<20s} | {d['wer']*100:>5.1f}% | {d['substitution_rate']*100:>7.2f}% | {d['insertion_rate']*100:>7.2f}% | {d['deletion_rate']*100:>7.2f}% | {d['sub_pct_of_errors']:>6.1f}% | {d['ins_pct_of_errors']:>6.1f}% | {d['del_pct_of_errors']:>6.1f}%")

    # ── Save results ────────────────────────────────────────────────
    # Convert for JSON
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

    output_path = os.path.join(OUTPUT_DIR, "error_decomposition.json")
    with open(output_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n\nSaved: {output_path}")


if __name__ == "__main__":
    main()
