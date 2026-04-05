"""
Validate test run predictions: check WER is reasonable and output is not chatty.

Usage:
    python scripts/validate_test_run.py --output_dir results/fairspeech_test

Exit code 0 = all checks pass, 1 = one or more models failed validation.
"""

import argparse
import os
import re
import sys
import pandas as pd

WER_THRESHOLD = 80.0  # any model above this is likely broken
CHATTY_PATTERNS = [
    r"^Sure",
    r"^Of course",
    r"^Here(?:'s| is)",
    r"^The user(?:'s)?",
    r"transcri(?:be|ption|bed)",
    r"written format",
]


def validate_model(csv_path):
    """Return (passed: bool, issues: list[str])."""
    model_name = os.path.basename(csv_path).replace("predictions_", "").replace(".csv", "")
    df = pd.read_csv(csv_path)
    issues = []

    if len(df) == 0:
        return False, ["no predictions found"]

    # Check 1: Overall WER
    avg_wer = df["wer"].mean() * 100
    if avg_wer > WER_THRESHOLD:
        issues.append(f"WER {avg_wer:.1f}% exceeds {WER_THRESHOLD}% threshold")

    # Check 2: All-empty hypotheses
    empty_count = (df["hypothesis_raw"].fillna("") == "").sum()
    if empty_count == len(df):
        issues.append("all hypotheses are empty")
    elif empty_count > 0:
        issues.append(f"{empty_count}/{len(df)} empty hypotheses")

    # Check 3: Chatty / wrapper text in raw output
    chatty_count = 0
    for _, row in df.iterrows():
        raw = str(row.get("hypothesis_raw", ""))
        for pat in CHATTY_PATTERNS:
            if re.search(pat, raw, re.IGNORECASE):
                chatty_count += 1
                break
    if chatty_count > 0:
        issues.append(f"{chatty_count}/{len(df)} hypotheses contain chatbot-style wrapper text")

    # Check 4: Hypothesis much longer than reference (hallucination signal)
    for _, row in df.iterrows():
        ref = str(row.get("reference", ""))
        hyp = str(row.get("hypothesis_raw", ""))
        if len(ref) > 0 and len(hyp) > len(ref) * 3 and len(hyp) > 50:
            issues.append(f"hypothesis for {row['utterance_id']} is {len(hyp)/len(ref):.1f}x longer than reference — possible hallucination")

    passed = len(issues) == 0
    return passed, issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    csv_files = sorted(
        f for f in os.listdir(args.output_dir)
        if f.startswith("predictions_") and f.endswith(".csv")
    )

    if not csv_files:
        print(f"ERROR: No prediction files found in {args.output_dir}")
        sys.exit(1)

    total_passed = 0
    total_failed = 0

    print(f"Validating {len(csv_files)} model(s) in {args.output_dir}\n")
    for csv_file in csv_files:
        model_name = csv_file.replace("predictions_", "").replace(".csv", "")
        csv_path = os.path.join(args.output_dir, csv_file)
        passed, issues = validate_model(csv_path)

        if passed:
            print(f"  ✓ {model_name}")
            total_passed += 1
        else:
            print(f"  ✗ {model_name}")
            for issue in issues:
                print(f"      - {issue}")
            total_failed += 1

    print(f"\n{'='*50}")
    print(f"PASSED: {total_passed}  FAILED: {total_failed}")
    if total_failed > 0:
        print("FIX the failed models before running full inference!")
    print(f"{'='*50}")
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
