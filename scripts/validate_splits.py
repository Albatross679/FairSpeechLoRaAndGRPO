#!/usr/bin/env python3
"""
Validate Fair-Speech train/eval splits: demographic completeness, group sizes,
speaker leakage, audio file existence, text normalization, and distribution drift.

Usage:
    python scripts/validate_splits.py
    python scripts/validate_splits.py --plots
    python scripts/validate_splits.py --train /path/to/fs_train.csv --eval /path/to/fs_eval.csv

Exit code 0 = all checks pass, 1 = one or more critical issues found.
"""

import argparse
import os
import random
import re
import string
import sys

import pandas as pd

# -- Config ------------------------------------------------------------------
OUTPUT_DIR = "/users/PAS2030/srishti/asr_fairness/data"
MIN_GROUP_SIZE = 50
FS_DEMOGRAPHIC_AXES = ["ethnicity", "gender", "age", "first_language"]
CV_DEMOGRAPHIC_AXES = ["accent", "gender", "age"]

# Characters expected in normalized text: lowercase ASCII, digits, punctuation,
# whitespace. The Whisper EnglishTextNormalizer produces this character set.
NORMALIZED_CHARS = set(string.ascii_lowercase + string.digits + string.punctuation + " ")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Fair-Speech train/eval splits."
    )
    parser.add_argument(
        "--train",
        default=os.path.join(OUTPUT_DIR, "fs_train.csv"),
        help="Path to train split CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--eval",
        default=os.path.join(OUTPUT_DIR, "fs_eval.csv"),
        help="Path to eval split CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--cv-manifest",
        default=os.path.join(OUTPUT_DIR, "cv_test_manifest.csv"),
        help="Path to Common Voice test manifest CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=False,
        help="Generate diagnostic plots (default: False)",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory for plots (default: %(default)s)",
    )
    return parser.parse_args()


def check_demographic_completeness(df_train, df_eval, issues, warnings):
    """Check 1: Demographic label completeness for each axis."""
    print(f"\n{'~'*50}")
    print("CHECK 1: Demographic Label Completeness")
    print(f"{'~'*50}")

    for split_name, df in [("train", df_train), ("eval", df_eval)]:
        for axis in FS_DEMOGRAPHIC_AXES:
            if axis not in df.columns:
                warnings.append(f"{split_name}: column '{axis}' not found")
                continue

            missing = df[axis].isna().sum() + (df[axis].astype(str).str.strip() == "").sum()
            pct = 100 * missing / len(df) if len(df) > 0 else 0

            if axis == "ethnicity" and pct > 5:
                issues.append(
                    f"{split_name}/{axis}: {missing}/{len(df)} "
                    f"({pct:.1f}%) missing labels (>5% threshold)"
                )
                print(f"  FAIL: {split_name}/{axis}: {pct:.1f}% missing")
            elif pct > 0:
                print(f"  INFO: {split_name}/{axis}: {pct:.1f}% missing ({missing}/{len(df)})")
            else:
                print(f"  OK:   {split_name}/{axis}: 0% missing")


def check_min_group_size(df_train, df_eval, issues, warnings):
    """Check 2: MIN_GROUP_SIZE per split for each demographic axis."""
    print(f"\n{'~'*50}")
    print(f"CHECK 2: MIN_GROUP_SIZE ({MIN_GROUP_SIZE}) per Split")
    print(f"{'~'*50}")

    for axis in FS_DEMOGRAPHIC_AXES:
        for split_name, df, target in [("train", df_train, warnings), ("eval", df_eval, issues)]:
            if axis not in df.columns:
                continue

            counts = df[axis].value_counts()
            small_groups = counts[counts < MIN_GROUP_SIZE]
            if len(small_groups) > 0:
                for group, count in small_groups.items():
                    if not group or (isinstance(group, str) and group.strip() == ""):
                        continue  # skip empty/missing labels
                    msg = (
                        f"{split_name}/{axis}/{group}: only {count} utterances "
                        f"(< MIN_GROUP_SIZE={MIN_GROUP_SIZE})"
                    )
                    target.append(msg)
                    label = "FAIL" if split_name == "eval" else "WARN"
                    print(f"  {label}: {msg}")
            else:
                print(f"  OK:   {split_name}/{axis}: all groups >= {MIN_GROUP_SIZE}")


def check_speaker_leakage(df_train, df_eval, issues, warnings):
    """Check 3: Speaker and utterance leakage between train and eval."""
    print(f"\n{'~'*50}")
    print("CHECK 3: Speaker/Utterance Leakage")
    print(f"{'~'*50}")

    # Speaker leakage
    if "speaker_id" in df_train.columns and "speaker_id" in df_eval.columns:
        train_speakers = set(df_train["speaker_id"].dropna())
        eval_speakers = set(df_eval["speaker_id"].dropna())
        speaker_overlap = train_speakers & eval_speakers

        if len(speaker_overlap) > 0:
            issues.append(
                f"CRITICAL: Speaker leakage -- {len(speaker_overlap)} speakers "
                f"in both splits: {list(speaker_overlap)[:5]}..."
            )
            print(f"  FAIL: {len(speaker_overlap)} speakers in both splits")
        else:
            print(f"  OK:   No speaker leakage "
                  f"(train={len(train_speakers)}, eval={len(eval_speakers)})")
    else:
        warnings.append("speaker_id column not found in one or both splits")
        print(f"  WARN: speaker_id column not found")

    # Utterance leakage
    if "utterance_id" in df_train.columns and "utterance_id" in df_eval.columns:
        train_utts = set(df_train["utterance_id"])
        eval_utts = set(df_eval["utterance_id"])
        utt_overlap = train_utts & eval_utts

        if len(utt_overlap) > 0:
            issues.append(
                f"CRITICAL: Utterance leakage -- {len(utt_overlap)} utterances "
                f"in both splits"
            )
            print(f"  FAIL: {len(utt_overlap)} utterances in both splits")
        else:
            print(f"  OK:   No utterance leakage "
                  f"(train={len(train_utts)}, eval={len(eval_utts)})")
    else:
        warnings.append("utterance_id column not found in one or both splits")
        print(f"  WARN: utterance_id column not found")


def check_audio_existence(df_train, df_eval, issues, warnings):
    """Check 4: Audio file existence (sample 100 from each split)."""
    print(f"\n{'~'*50}")
    print("CHECK 4: Audio File Existence (sampled)")
    print(f"{'~'*50}")

    random.seed(42)

    for split_name, df in [("train", df_train), ("eval", df_eval)]:
        if "audio_path" not in df.columns:
            warnings.append(f"{split_name}: audio_path column not found")
            print(f"  WARN: {split_name}: no audio_path column")
            continue

        paths = df["audio_path"].tolist()
        sample_size = min(100, len(paths))
        sampled = random.sample(paths, sample_size)

        missing = 0
        for path in sampled:
            if not os.path.isfile(str(path)):
                missing += 1

        if missing > 0:
            issues.append(
                f"{split_name}: {missing}/{sample_size} sampled audio files missing"
            )
            print(f"  FAIL: {split_name}: {missing}/{sample_size} audio files missing")
        else:
            print(f"  OK:   {split_name}: all {sample_size} sampled audio files exist")


def check_text_normalization(df_train, df_eval, issues, warnings):
    """Check 5: Text normalization consistency.

    Checks that:
    - sentence column has no empty values where sentence_raw is non-empty
    - A sample of sentences contain only expected normalized characters
    """
    print(f"\n{'~'*50}")
    print("CHECK 5: Text Normalization Consistency")
    print(f"{'~'*50}")

    random.seed(42)

    for split_name, df in [("train", df_train), ("eval", df_eval)]:
        # Check for empty normalized text where raw is non-empty
        if "sentence" in df.columns and "sentence_raw" in df.columns:
            raw_nonempty = df["sentence_raw"].fillna("").str.strip() != ""
            norm_empty = df["sentence"].fillna("").str.strip() == ""
            failed_norm = (raw_nonempty & norm_empty).sum()

            if failed_norm > 0:
                issues.append(
                    f"{split_name}: {failed_norm} utterances have empty 'sentence' "
                    f"but non-empty 'sentence_raw' (normalization failure)"
                )
                print(f"  FAIL: {split_name}: {failed_norm} normalization failures")
            else:
                print(f"  OK:   {split_name}: no normalization failures")

        elif "sentence" in df.columns:
            print(f"  INFO: {split_name}: no sentence_raw column to compare")
        else:
            warnings.append(f"{split_name}: sentence column not found")
            print(f"  WARN: {split_name}: no sentence column")
            continue

        # Check character set in normalized sentences
        if "sentence" in df.columns:
            sentences = df["sentence"].dropna().tolist()
            sample_size = min(50, len(sentences))
            sampled = random.sample(sentences, sample_size)

            non_normalized = 0
            for sent in sampled:
                sent_str = str(sent)
                if sent_str and not all(c in NORMALIZED_CHARS for c in sent_str):
                    non_normalized += 1

            if non_normalized > 0:
                warnings.append(
                    f"{split_name}: {non_normalized}/{sample_size} sampled sentences "
                    f"contain characters outside expected normalized set"
                )
                print(f"  WARN: {split_name}: {non_normalized}/{sample_size} "
                      f"sentences have unexpected characters")
            else:
                print(f"  OK:   {split_name}: all {sample_size} sampled sentences "
                      f"use normalized character set")


def check_cv_manifest(cv_manifest_path, issues, warnings):
    """Check 6: Common Voice manifest existence."""
    print(f"\n{'~'*50}")
    print("CHECK 6: Common Voice Manifest Existence")
    print(f"{'~'*50}")

    if os.path.isfile(cv_manifest_path):
        df_cv = pd.read_csv(cv_manifest_path)
        print(f"  OK:   {cv_manifest_path} found ({len(df_cv):,} utterances)")
    else:
        warnings.append(
            f"CV manifest not found at {cv_manifest_path} -- "
            f"run prepare_dataset.py first"
        )
        print(f"  WARN: {cv_manifest_path} not found")


def check_distribution_drift(df_train, df_eval, issues, warnings):
    """Check 7: Distribution drift between train and eval for ethnicity axis."""
    print(f"\n{'~'*50}")
    print("CHECK 7: Distribution Drift (ethnicity)")
    print(f"{'~'*50}")

    if "ethnicity" not in df_train.columns or "ethnicity" not in df_eval.columns:
        warnings.append("ethnicity column not found -- skipping drift check")
        print(f"  WARN: ethnicity column not found")
        return

    train_dist = df_train["ethnicity"].value_counts(normalize=True)
    eval_dist = df_eval["ethnicity"].value_counts(normalize=True)

    all_groups = set(train_dist.index) | set(eval_dist.index)
    drift_threshold = 5.0  # percentage points

    any_drift = False
    for group in sorted(all_groups):
        if not group or (isinstance(group, str) and group.strip() == ""):
            continue
        train_pct = train_dist.get(group, 0) * 100
        eval_pct = eval_dist.get(group, 0) * 100
        diff = abs(train_pct - eval_pct)

        if diff > drift_threshold:
            any_drift = True
            warnings.append(
                f"ethnicity/{group}: proportion differs by {diff:.1f}pp "
                f"(train={train_pct:.1f}%, eval={eval_pct:.1f}%)"
            )
            print(f"  WARN: {group}: {diff:.1f}pp drift "
                  f"(train={train_pct:.1f}%, eval={eval_pct:.1f}%)")
        else:
            print(f"  OK:   {group}: {diff:.1f}pp drift "
                  f"(train={train_pct:.1f}%, eval={eval_pct:.1f}%)")

    if not any_drift:
        print(f"  All groups within {drift_threshold}pp threshold.")


def generate_plots(df_train, df_eval, output_dir):
    """Generate optional diagnostic plots (--plots flag)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n{'~'*50}")
    print("GENERATING DIAGNOSTIC PLOTS")
    print(f"{'~'*50}")

    os.makedirs(output_dir, exist_ok=True)

    # -- Plot 1: Ethnicity distribution (train vs eval) ---------------------
    if "ethnicity" in df_train.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        train_dist = df_train["ethnicity"].value_counts(normalize=True).sort_index()
        eval_dist = df_eval["ethnicity"].value_counts(normalize=True).sort_index()

        all_groups = sorted(set(train_dist.index) | set(eval_dist.index))
        all_groups = [g for g in all_groups if g and str(g).strip()]

        x = range(len(all_groups))
        width = 0.35

        train_vals = [train_dist.get(g, 0) * 100 for g in all_groups]
        eval_vals = [eval_dist.get(g, 0) * 100 for g in all_groups]

        ax.bar([i - width / 2 for i in x], train_vals, width, label="Train", color="#4C72B0")
        ax.bar([i + width / 2 for i in x], eval_vals, width, label="Eval", color="#DD8452")

        ax.set_xlabel("Ethnicity")
        ax.set_ylabel("Proportion (%)")
        ax.set_title("Ethnicity Distribution: Train vs Eval")
        ax.set_xticks(list(x))
        ax.set_xticklabels(all_groups, rotation=45, ha="right")
        ax.legend()

        plot_path = os.path.join(output_dir, "split_ethnicity_dist.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {plot_path}")

    # -- Plot 2: Age distribution (train vs eval) ---------------------------
    if "age" in df_train.columns:
        fig, ax = plt.subplots(figsize=(8, 5))

        train_dist = df_train["age"].value_counts(normalize=True).sort_index()
        eval_dist = df_eval["age"].value_counts(normalize=True).sort_index()

        all_groups = sorted(set(train_dist.index) | set(eval_dist.index))
        all_groups = [g for g in all_groups if g and str(g).strip()]

        x = range(len(all_groups))
        width = 0.35

        train_vals = [train_dist.get(g, 0) * 100 for g in all_groups]
        eval_vals = [eval_dist.get(g, 0) * 100 for g in all_groups]

        ax.bar([i - width / 2 for i in x], train_vals, width, label="Train", color="#4C72B0")
        ax.bar([i + width / 2 for i in x], eval_vals, width, label="Eval", color="#DD8452")

        ax.set_xlabel("Age Group")
        ax.set_ylabel("Proportion (%)")
        ax.set_title("Age Distribution: Train vs Eval")
        ax.set_xticks(list(x))
        ax.set_xticklabels(all_groups, rotation=45, ha="right")
        ax.legend()

        plot_path = os.path.join(output_dir, "split_age_dist.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {plot_path}")

    # -- Plot 3: Eval group sizes with MIN_GROUP_SIZE line ------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    group_data = []
    for axis in FS_DEMOGRAPHIC_AXES:
        if axis in df_eval.columns:
            counts = df_eval[axis].value_counts()
            for group, count in counts.items():
                if group and str(group).strip():
                    group_data.append({
                        "axis": axis,
                        "group": f"{axis}/{group}",
                        "count": count,
                    })

    if group_data:
        group_df = pd.DataFrame(group_data).sort_values("count", ascending=True)

        colors = ["#C44E52" if c < MIN_GROUP_SIZE else "#55A868"
                  for c in group_df["count"]]

        ax.barh(range(len(group_df)), group_df["count"], color=colors)
        ax.set_yticks(range(len(group_df)))
        ax.set_yticklabels(group_df["group"], fontsize=7)
        ax.axvline(x=MIN_GROUP_SIZE, color="red", linestyle="--",
                   label=f"MIN_GROUP_SIZE={MIN_GROUP_SIZE}")
        ax.set_xlabel("Number of Utterances")
        ax.set_title("Eval Split: Group Sizes by Demographic Axis")
        ax.legend()

    plot_path = os.path.join(output_dir, "split_group_sizes.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_path}")


def main():
    args = parse_args()

    print(f"{'='*60}")
    print("VALIDATE FAIR-SPEECH TRAIN/EVAL SPLITS")
    print(f"{'='*60}")
    print(f"Train:       {args.train}")
    print(f"Eval:        {args.eval}")
    print(f"CV manifest: {args.cv_manifest}")
    print(f"Plots:       {args.plots}")

    # Load splits
    if not os.path.isfile(args.train):
        print(f"\nERROR: Train file not found: {args.train}")
        print("Run prepare_splits.py first.")
        sys.exit(1)

    if not os.path.isfile(args.eval):
        print(f"\nERROR: Eval file not found: {args.eval}")
        print("Run prepare_splits.py first.")
        sys.exit(1)

    df_train = pd.read_csv(args.train)
    df_eval = pd.read_csv(args.eval)

    n_train_speakers = df_train["speaker_id"].nunique() if "speaker_id" in df_train.columns else "?"
    n_eval_speakers = df_eval["speaker_id"].nunique() if "speaker_id" in df_eval.columns else "?"

    print(f"\nLoaded:")
    print(f"  Train: {len(df_train):,} utterances, {n_train_speakers} speakers")
    print(f"  Eval:  {len(df_eval):,} utterances, {n_eval_speakers} speakers")

    # Run checks
    issues = []
    warnings = []

    check_demographic_completeness(df_train, df_eval, issues, warnings)
    check_min_group_size(df_train, df_eval, issues, warnings)
    check_speaker_leakage(df_train, df_eval, issues, warnings)
    check_audio_existence(df_train, df_eval, issues, warnings)
    check_text_normalization(df_train, df_eval, issues, warnings)
    check_cv_manifest(args.cv_manifest, issues, warnings)
    check_distribution_drift(df_train, df_eval, issues, warnings)

    # Optional plots
    if args.plots:
        generate_plots(df_train, df_eval, args.output_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Train: {len(df_train):,} utterances, {n_train_speakers} speakers")
    print(f"Eval:  {len(df_eval):,} utterances, {n_eval_speakers} speakers")

    for axis in FS_DEMOGRAPHIC_AXES:
        if axis in df_eval.columns:
            n_groups = df_eval[axis].value_counts().shape[0]
            print(f"  {axis}: {n_groups} groups in eval")

    if issues:
        print(f"\nFAILED: {len(issues)} issue(s)")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. FAIL: {issue}")
        sys.exit(1)
    else:
        print(f"\nPASSED: All checks passed")
        if warnings:
            print(f"  ({len(warnings)} warning(s))")
            for w in warnings:
                print(f"    WARN: {w}")
        sys.exit(0)


if __name__ == "__main__":
    main()
