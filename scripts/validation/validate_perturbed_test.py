#!/usr/bin/env python3
"""
Validate perturbed SNR test run: checks all 9 models × 3 SNR levels completed
and results are sane.

Usage:
    python scripts/validation/validate_perturbed_test.py
    python scripts/validation/validate_perturbed_test.py --output_dir results/perturbed_test

Checks:
  1. All 27 prediction files exist (9 models × 3 SNR levels)
  2. Each file has 5 rows
  3. WER increases with noise (snr_20db < snr_10db < snr_0db) — at least weakly
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
    "qwen3_asr_0_6b",
    "qwen3_asr_1_7b",
    "canary_qwen_2_5b",
    "granite_speech_3_3_2b",
    "granite_speech_3_3_8b",
]

SNR_LEVELS = ["snr_20db", "snr_10db", "snr_0db"]

# Expected filename pattern: predictions_{model}_{snr}.csv
# e.g., predictions_whisper_large_v3_snr_20db.csv

CHATTY_PATTERNS = [
    r"^Sure", r"^Of course", r"^Here(?:'s| is)",
    r"^The user", r"transcri(?:be|ption|bed)", r"written format",
]


def check_chatty(raw_text):
    for pat in CHATTY_PATTERNS:
        if re.search(pat, str(raw_text), re.IGNORECASE):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/perturbed_test")
    parser.add_argument("--clean_dir", default="results/fairspeech",
                        help="Dir with clean predictions for comparison")
    args = parser.parse_args()

    output_dir = os.path.join(
        "/users/PAS2030/srishti/asr_fairness", args.output_dir
    ) if not os.path.isabs(args.output_dir) else args.output_dir

    clean_dir = os.path.join(
        "/users/PAS2030/srishti/asr_fairness", args.clean_dir
    ) if not os.path.isabs(args.clean_dir) else args.clean_dir

    print(f"Validating perturbed SNR test results in: {output_dir}\n")

    issues = []
    all_results = {}  # (model, snr) -> mean_wer

    # ── Check all expected files exist ──────────────────────────────────
    for model in MODELS:
        for snr in SNR_LEVELS:
            fname = f"predictions_{model}_{snr}.csv"
            fpath = os.path.join(output_dir, fname)

            if not os.path.exists(fpath):
                issues.append(f"MISSING: {fname}")
                continue

            df = pd.read_csv(fpath)

            # Check row count
            if len(df) != 5:
                issues.append(f"{fname}: expected 5 rows, got {len(df)}")

            # Check for empty hypotheses
            empty = (df["hypothesis_raw"].fillna("") == "").sum()
            if empty > 0:
                issues.append(f"{fname}: {empty}/{len(df)} empty hypotheses")

            # Check perturbation column
            if "perturbation" in df.columns:
                pert_vals = df["perturbation"].unique()
                if len(pert_vals) != 1 or pert_vals[0] != snr:
                    issues.append(f"{fname}: perturbation column = {pert_vals}, expected [{snr}]")

            # Check for chatty text
            chatty = df["hypothesis_raw"].fillna("").apply(check_chatty).sum()
            if chatty > 0:
                issues.append(f"{fname}: {chatty}/{len(df)} chatbot-style outputs")

            # Store mean WER
            mean_wer = df["wer"].mean() * 100
            all_results[(model, snr)] = mean_wer

    # ── Check WER monotonicity: should generally increase with more noise ──
    print("Model                       SNR_20dB  SNR_10dB  SNR_0dB   Monotonic?")
    print("─" * 75)
    for model in MODELS:
        wers = [all_results.get((model, snr)) for snr in SNR_LEVELS]
        if all(w is not None for w in wers):
            # Check weak monotonicity (20dB <= 10dB <= 0dB)
            monotonic = wers[0] <= wers[1] + 5 and wers[1] <= wers[2] + 5
            flag = "✓" if monotonic else "⚠ non-monotonic"
            model_display = model.replace("_", "-")[:27]
            print(f"  {model_display:<27s} {wers[0]:7.1f}%  {wers[1]:7.1f}%  {wers[2]:7.1f}%  {flag}")
            if not monotonic:
                issues.append(f"{model}: WER not monotonically increasing with noise "
                              f"({wers[0]:.1f}→{wers[1]:.1f}→{wers[2]:.1f})")
        else:
            missing = [snr for snr, w in zip(SNR_LEVELS, wers) if w is None]
            print(f"  {model:<27s}  (missing: {', '.join(missing)})")

    # ── Compare with clean baselines if available ───────────────────────
    print(f"\n{'─' * 75}")
    print("Clean baseline comparison (if available):")
    for model in MODELS:
        clean_path = os.path.join(clean_dir, f"predictions_{model}.csv")
        if os.path.exists(clean_path):
            clean_df = pd.read_csv(clean_path)
            clean_wer = clean_df["wer"].mean() * 100
            snr20_wer = all_results.get((model, "snr_20db"))
            if snr20_wer is not None:
                model_display = model.replace("_", "-")[:27]
                degradation = snr20_wer - clean_wer
                print(f"  {model_display:<27s}  clean={clean_wer:.1f}%  snr_20db={snr20_wer:.1f}%  "
                      f"Δ={degradation:+.1f}%")
                if snr20_wer < clean_wer - 5:
                    issues.append(f"{model}: SNR_20dB WER ({snr20_wer:.1f}%) is LOWER than "
                                  f"clean ({clean_wer:.1f}%) — suspicious")

    # ── Summary ────────────────────────────────────────────────────────
    expected = len(MODELS) * len(SNR_LEVELS)
    found = len(all_results)

    print(f"\n{'═' * 75}")
    print(f"  Files: {found}/{expected}")
    print(f"  Issues: {len(issues)}")

    if issues:
        print(f"\n  ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
        print(f"\n  ⚠ FIX issues before submitting full batch!")
        sys.exit(1)
    else:
        print(f"\n  ✓ All checks passed — safe to run full SNR batch on Fair-Speech")
        print(f"    Command: bash scripts/launch_priority_batch.sh")
        sys.exit(0)


if __name__ == "__main__":
    main()
