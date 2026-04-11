#!/usr/bin/env python3
"""
Error decomposition for Fair-Speech: substitution/insertion/deletion rates
by ethnicity and L1 group across all models. Extends the CV error_decomposition.py.

Usage:
    python scripts/metrics/error_decomposition_fs.py \
        --results_dir results/fairspeech/
        [--output_dir results/fairspeech/analysis/]
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import jiwer
import numpy as np
import pandas as pd

ETHNICITY_ORDER = ["White", "Black/AA", "Hispanic", "Asian", "Native American", "Pacific Islander", "Middle Eastern"]
L1_GROUP_ORDER = ["English", "Spanish", "Mandarin", "Hindi", "Other"]
AGE_ORDER = ["18-22", "23-30", "31-45", "46-65"]

MODEL_ORDER = [
    "wav2vec2-large", "whisper-small", "whisper-medium", "whisper-large-v3",
    "qwen3-asr-0.6b", "qwen3-asr-1.7b", "granite-speech-3.3-2b",
    "canary-qwen-2.5b", "granite-speech-3.3-8b",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Error decomposition for Fair-Speech")
    parser.add_argument("--results_dir", type=str,
                        default="/users/PAS2030/srishti/asr_fairness/results/fairspeech")
    parser.add_argument("--output_dir", type=str,
                        default="/users/PAS2030/srishti/asr_fairness/results/fairspeech/analysis")
    parser.add_argument("--axes", nargs="+", default=["ethnicity", "l1_group"])
    return parser.parse_args()


def compute_error_decomposition(refs, hyps):
    """Compute substitution, insertion, deletion rates and counts."""
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if not valid:
        return None

    refs_clean, hyps_clean = zip(*valid)
    refs_clean, hyps_clean = list(refs_clean), list(hyps_clean)

    try:
        output = jiwer.process_words(refs_clean, hyps_clean)
    except Exception:
        return None

    total_ref = output.substitutions + output.deletions + output.hits
    total_errors = output.substitutions + output.insertions + output.deletions
    wer = total_errors / total_ref if total_ref > 0 else 0

    return {
        "n_utterances": len(refs_clean),
        "total_ref_words": total_ref,
        "wer": wer,
        "substitutions": output.substitutions,
        "insertions": output.insertions,
        "deletions": output.deletions,
        "hits": output.hits,
        "sub_rate": output.substitutions / total_ref if total_ref > 0 else 0,
        "ins_rate": output.insertions / total_ref if total_ref > 0 else 0,
        "del_rate": output.deletions / total_ref if total_ref > 0 else 0,
        "sub_pct": output.substitutions / max(1, total_errors) * 100,
        "ins_pct": output.insertions / max(1, total_errors) * 100,
        "del_pct": output.deletions / max(1, total_errors) * 100,
    }


def get_group_order(axis):
    return {
        "ethnicity": ETHNICITY_ORDER,
        "l1_group": L1_GROUP_ORDER,
        "age": AGE_ORDER,
    }.get(axis, None)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pred_files = sorted([
        f for f in os.listdir(args.results_dir)
        if f.startswith("predictions_") and f.endswith(".csv") and "checkpoint" not in f
    ])
    if not pred_files:
        print(f"No prediction files in {args.results_dir}")
        sys.exit(1)

    all_results = {}

    for pred_file in pred_files:
        model_name = pred_file.replace("predictions_", "").replace(".csv", "")
        display = model_name.replace("_", "-")
        df = pd.read_csv(os.path.join(args.results_dir, pred_file))

        print(f"\n{'='*60}")
        print(f"Model: {display}")
        print(f"{'='*60}")

        model_results = {}

        for axis in args.axes:
            if axis not in df.columns:
                print(f"  {axis}: not found in predictions")
                continue

            df_ax = df[df[axis].notna() & (df[axis] != "")]
            order = get_group_order(axis)
            groups_in_data = df_ax[axis].unique()
            if order:
                groups = [g for g in order if g in groups_in_data]
            else:
                groups = sorted(groups_in_data)

            print(f"\n  {axis.upper()}")
            print(f"  {'Group':<25s} | {'N':>6s} | {'WER':>6s} | {'Sub%':>5s} | {'Ins%':>5s} | {'Del%':>5s} | {'SubR':>5s} | {'InsR':>5s} | {'DelR':>5s}")
            print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")

            axis_results = {}
            for group in groups:
                grp = df_ax[df_ax[axis] == group]
                if len(grp) < 50:
                    continue

                refs = grp["reference"].fillna("").astype(str).tolist()
                hyps = grp["hypothesis"].fillna("").astype(str).tolist()
                decomp = compute_error_decomposition(refs, hyps)
                if decomp is None:
                    continue

                axis_results[group] = decomp
                print(f"  {group:<25s} | {decomp['n_utterances']:>6,} | {decomp['wer']*100:>5.2f}% | "
                      f"{decomp['sub_pct']:>4.1f}% | {decomp['ins_pct']:>4.1f}% | {decomp['del_pct']:>4.1f}% | "
                      f"{decomp['sub_rate']*100:>4.2f}% | {decomp['ins_rate']*100:>4.2f}% | {decomp['del_rate']*100:>4.2f}%")

            model_results[axis] = axis_results

        all_results[display] = model_results

    # ── Save JSON ──────────────────────────────────────────────────
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [make_serializable(v) for v in obj]
        return obj

    output_path = os.path.join(args.output_dir, "error_decomposition_fs.json")
    with open(output_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nSaved: {output_path}")

    # ── Generate LaTeX table (ethnicity) ───────────────────────────
    if "ethnicity" in args.axes:
        models = [m.replace("_", "-") for m in MODEL_ORDER if m.replace("_", "-") in all_results]
        groups = ETHNICITY_ORDER

        print(f"\n{'='*60}")
        print("LaTeX: Insertion Rate by Ethnicity")
        print(f"{'='*60}")

        header = "Model"
        for g in groups:
            header += f" & {g}"
        header += r" \\"

        rows = []
        for model in models:
            ed = all_results.get(model, {}).get("ethnicity", {})
            row = model
            for g in groups:
                if g in ed:
                    row += f" & {ed[g]['ins_rate']*100:.2f}"
                else:
                    row += " & --"
            row += r" \\"
            rows.append(row)

        latex = (
            r"\begin{table}[t]" + "\n"
            r"\centering" + "\n"
            r"\caption{Insertion Rate (\%) by Ethnicity}" + "\n"
            r"\begin{tabular}{l" + "r" * len(groups) + "}\n"
            r"\toprule" + "\n"
            + header + "\n"
            r"\midrule" + "\n"
            + "\n".join(rows) + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\end{table}"
        )
        latex_path = os.path.join(args.output_dir, "table_insertion_by_ethnicity.tex")
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"Saved: {latex_path}")


if __name__ == "__main__":
    main()
