#!/usr/bin/env python3
"""
Validate CV perturbed test run: checks all 9 models × 12 perturbation conditions.

Usage:
    python scripts/validation/validate_cv_perturbed_test.py
    python scripts/validation/validate_cv_perturbed_test.py --output_dir results/perturbed_cv_test

Checks:
  1. All 108 prediction files exist (9 models × 12 conditions)
  2. Each file has 5 rows
  3. WER increases with degradation severity (within each perturbation type)
  4. No empty hypotheses or chatbot wrapper text
  5. Perturbation column is correctly set
  6. Compares against clean CV baselines if available

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

# Perturbation families: each is (label, [levels in increasing severity])
PERTURBATION_FAMILIES = [
    ("SNR",     ["snr_20db", "snr_10db", "snr_0db"]),
    ("Reverb",  ["reverb_0.3s", "reverb_0.6s", "reverb_1.0s"]),
    ("Silence", ["silence_25pct", "silence_50pct", "silence_75pct"]),
    ("Masking", ["mask_10pct", "mask_20pct", "mask_30pct"]),
]

ALL_PERTURBATIONS = [p for _, levels in PERTURBATION_FAMILIES for p in levels]

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
    parser.add_argument("--output_dir", default="results/perturbed_cv_test")
    parser.add_argument("--clean_dir", default="results/commonvoice",
                        help="Dir with clean CV predictions for comparison")
    args = parser.parse_args()

    project = "/users/PAS2030/srishti/asr_fairness"
    output_dir = os.path.join(project, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    clean_dir = os.path.join(project, args.clean_dir) if not os.path.isabs(args.clean_dir) else args.clean_dir

    print(f"Validating CV perturbed test results in: {output_dir}\n")

    issues = []
    all_results = {}  # (model, perturbation) -> mean_wer

    # ── Check all expected files exist and are valid ──────────────────
    for model in MODELS:
        for pert in ALL_PERTURBATIONS:
            # run_inference.py replaces dots in perturbation suffix: reverb_0.3s -> reverb_0_3s
            pert_suffix = pert.replace(".", "_")
            fname = f"predictions_{model}_{pert_suffix}.csv"
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
                if len(pert_vals) != 1 or pert_vals[0] != pert:
                    issues.append(f"{fname}: perturbation column = {pert_vals}, expected [{pert}]")

            # Check for chatty text
            chatty = df["hypothesis_raw"].fillna("").apply(check_chatty).sum()
            if chatty > 0:
                issues.append(f"{fname}: {chatty}/{len(df)} chatbot-style outputs")

            # Store mean WER
            mean_wer = df["wer"].mean() * 100
            all_results[(model, pert)] = mean_wer

    # ── Check WER monotonicity per perturbation family ────────────────
    for family_name, levels in PERTURBATION_FAMILIES:
        print(f"\n{family_name}: {levels[0]:>15s}  {levels[1]:>15s}  {levels[2]:>15s}  Monotonic?")
        print("─" * 80)
        for model in MODELS:
            wers = [all_results.get((model, lv)) for lv in levels]
            if all(w is not None for w in wers):
                # Weak monotonicity with 5% tolerance (5 samples = high variance)
                monotonic = wers[0] <= wers[1] + 5 and wers[1] <= wers[2] + 5
                flag = "✓" if monotonic else "⚠ non-monotonic"
                model_display = model.replace("_", "-")[:27]
                print(f"  {model_display:<27s} {wers[0]:7.1f}%  {wers[1]:7.1f}%  {wers[2]:7.1f}%  {flag}")
                if not monotonic:
                    issues.append(f"{model}/{family_name}: WER not monotonic "
                                  f"({wers[0]:.1f}→{wers[1]:.1f}→{wers[2]:.1f})")
            else:
                missing = [lv for lv, w in zip(levels, wers) if w is None]
                model_display = model.replace("_", "-")[:27]
                print(f"  {model_display:<27s}  (missing: {', '.join(missing)})")

    # ── Compare mildest condition with clean baselines ────────────────
    print(f"\n{'─' * 80}")
    print("Clean baseline comparison (SNR 20dB — mildest noise condition):")
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

    # ── Summary ──────────────────────────────────────────────────────
    expected = len(MODELS) * len(ALL_PERTURBATIONS)
    found = len(all_results)

    print(f"\n{'═' * 80}")
    print(f"  Files: {found}/{expected}")
    print(f"  Issues: {len(issues)}")

    if issues:
        print(f"\n  ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
        print(f"\n  ⚠ FIX issues before submitting full CV batch!")
        sys.exit(1)
    else:
        print(f"\n  ✓ All 108 checks passed — safe to run full CV perturbation batch")
        print(f"    Submit with: bash slurm/launch_cv_full_batch.sh")
        sys.exit(0)


if __name__ == "__main__":
    main()
