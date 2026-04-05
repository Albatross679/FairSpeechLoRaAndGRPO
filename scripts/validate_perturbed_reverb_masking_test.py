#!/usr/bin/env python3
"""
Validate perturbed reverb and masking test runs on Fair-Speech (5 samples each).

Usage:
    python scripts/validate_perturbed_reverb_masking_test.py

Checks:
  1. All 27 reverb + 27 masking prediction files exist (9 models x 3 levels each)
  2. Each file has 5 rows
  3. WER increases with degradation level (weak monotonicity, 5% tolerance)
  4. No empty hypotheses or chatbot wrapper text
  5. Perturbation column is correctly set
  6. Compares against clean baselines if available

Exit code 0 = all pass, 1 = issues found.
"""

import argparse
import os
import re
import sys

import pandas as pd

MODELS = [
    "wav2vec2_large",
    "whisper_small",
    "whisper_medium",
    "whisper_large_v3",
    "qwen3_asr_0.6b",
    "qwen3_asr_1.7b",
    "canary_qwen_2.5b",
    "granite_speech_3.3_2b",
    "granite_speech_3.3_8b",
]

# run_inference.py does perturbation.replace(".", "_") for filenames
# so reverb_0.3s -> reverb_0_3s in filenames, but column keeps reverb_0.3s
REVERB_LEVELS = [
    ("reverb_0.3s", "reverb_0_3s"),   # (column_value, filename_suffix)
    ("reverb_0.6s", "reverb_0_6s"),
    ("reverb_1.0s", "reverb_1_0s"),
]

MASK_LEVELS = [
    ("mask_10pct", "mask_10pct"),
    ("mask_20pct", "mask_20pct"),
    ("mask_30pct", "mask_30pct"),
]

CHATTY_PATTERNS = [
    r"^Sure", r"^Of course", r"^Here(?:'s| is)",
    r"^The user", r"transcri(?:be|ption|bed)", r"written format",
]

BASE_DIR = "/users/PAS2030/srishti/asr_fairness"


def check_chatty(raw_text):
    for pat in CHATTY_PATTERNS:
        if re.search(pat, str(raw_text), re.IGNORECASE):
            return True
    return False


def validate_condition(output_dir, clean_dir, levels, condition_name):
    """Validate one perturbation type (reverb or masking)."""
    issues = []
    all_results = {}  # (model, level) -> mean_wer

    for model in MODELS:
        for col_val, file_suffix in levels:
            fname = f"predictions_{model}_{file_suffix}.csv"
            fpath = os.path.join(output_dir, fname)

            if not os.path.exists(fpath):
                issues.append(f"MISSING: {fname}")
                continue

            df = pd.read_csv(fpath)

            if len(df) != 5:
                issues.append(f"{fname}: expected 5 rows, got {len(df)}")

            # Empty hypotheses
            empty = (df["hypothesis_raw"].fillna("") == "").sum()
            if empty > 0:
                issues.append(f"{fname}: {empty}/{len(df)} empty hypotheses")

            # Perturbation column
            if "perturbation" in df.columns:
                pert_vals = df["perturbation"].unique()
                if len(pert_vals) != 1 or pert_vals[0] != col_val:
                    issues.append(f"{fname}: perturbation column = {pert_vals}, expected [{col_val}]")

            # Chatty text
            chatty = df["hypothesis_raw"].fillna("").apply(check_chatty).sum()
            if chatty > 0:
                issues.append(f"{fname}: {chatty}/{len(df)} chatbot-style outputs")

            mean_wer = df["wer"].mean() * 100
            all_results[(model, col_val)] = mean_wer

    # Monotonicity check
    level_names = [l[0] for l in levels]
    short_names = [l[0].split("_", 1)[1] for l in levels]
    header = f"Model                       {short_names[0]:>9s} {short_names[1]:>9s} {short_names[2]:>9s}   Monotonic?"
    print(f"\n  {condition_name.upper()}")
    print(f"  {header}")
    print(f"  {'─' * 75}")

    for model in MODELS:
        wers = [all_results.get((model, l)) for l in level_names]
        if all(w is not None for w in wers):
            monotonic = wers[0] <= wers[1] + 5 and wers[1] <= wers[2] + 5
            flag = "✓" if monotonic else "⚠ non-monotonic"
            model_display = model.replace("_", "-")[:27]
            print(f"    {model_display:<27s} {wers[0]:7.1f}%  {wers[1]:7.1f}%  {wers[2]:7.1f}%  {flag}")
            if not monotonic:
                issues.append(f"{model}: WER not monotonically increasing with {condition_name} "
                              f"({wers[0]:.1f}→{wers[1]:.1f}→{wers[2]:.1f})")
        else:
            missing = [l for l, w in zip(level_names, wers) if w is None]
            print(f"    {model:<27s}  (missing: {', '.join(missing)})")

    # Clean baseline comparison
    print(f"\n  Clean baseline comparison:")
    mildest = level_names[0]
    for model in MODELS:
        clean_path = os.path.join(clean_dir, f"predictions_{model}.csv")
        if os.path.exists(clean_path):
            clean_df = pd.read_csv(clean_path)
            clean_wer = clean_df["wer"].mean() * 100
            mild_wer = all_results.get((model, mildest))
            if mild_wer is not None:
                model_display = model.replace("_", "-")[:27]
                delta = mild_wer - clean_wer
                print(f"    {model_display:<27s}  clean={clean_wer:.1f}%  {mildest}={mild_wer:.1f}%  Δ={delta:+.1f}%")
                if mild_wer < clean_wer - 5:
                    issues.append(f"{model}: {mildest} WER ({mild_wer:.1f}%) is LOWER than "
                                  f"clean ({clean_wer:.1f}%) — suspicious")

    return issues, all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverb_dir", default="results/perturbed_reverb_test")
    parser.add_argument("--masking_dir", default="results/perturbed_masking_test")
    parser.add_argument("--clean_dir", default="results/fairspeech")
    args = parser.parse_args()

    def resolve(p):
        return os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p

    reverb_dir = resolve(args.reverb_dir)
    masking_dir = resolve(args.masking_dir)
    clean_dir = resolve(args.clean_dir)

    print(f"Validating FS reverb + masking test results\n")
    print(f"  Reverb dir:  {reverb_dir}")
    print(f"  Masking dir: {masking_dir}")
    print(f"  Clean dir:   {clean_dir}")

    all_issues = []

    # Validate reverb
    reverb_issues, reverb_results = validate_condition(
        reverb_dir, clean_dir, REVERB_LEVELS, "reverb"
    )
    all_issues.extend(reverb_issues)

    # Validate masking
    masking_issues, masking_results = validate_condition(
        masking_dir, clean_dir, MASK_LEVELS, "masking"
    )
    all_issues.extend(masking_issues)

    # Summary
    reverb_expected = len(MODELS) * len(REVERB_LEVELS)
    masking_expected = len(MODELS) * len(MASK_LEVELS)
    reverb_found = len(reverb_results)
    masking_found = len(masking_results)

    print(f"\n{'═' * 75}")
    print(f"  Reverb files:  {reverb_found}/{reverb_expected}")
    print(f"  Masking files: {masking_found}/{masking_expected}")
    print(f"  Total issues:  {len(all_issues)}")

    if all_issues:
        print(f"\n  ISSUES:")
        for issue in all_issues:
            print(f"    - {issue}")
        print(f"\n  ⚠ Fix issues before submitting full batch!")
        sys.exit(1)
    else:
        print(f"\n  ✓ All checks passed — safe to run full reverb + masking batch on Fair-Speech")
        sys.exit(0)


if __name__ == "__main__":
    main()
