#!/usr/bin/env python3
"""
Prepare Fair-Speech train/eval splits with demographic labels preserved.

Reads fs_manifest.csv (output of prepare_fairspeech.py), discovers speaker IDs
from the raw fairspeech_metadata.tsv, performs speaker-level stratified splitting
using StratifiedGroupKFold (80/20), and saves fs_train.csv and fs_eval.csv.

Common Voice test split is retained as-is (no new split created).

Usage:
    python scripts/prepare_splits.py
    python scripts/prepare_splits.py --seed 42 --output-dir /path/to/output
"""

import argparse
import csv
import os
import sys

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# -- Config ------------------------------------------------------------------
FS_DIR = "/users/PAS2030/srishti/asr_fairness/data/fairspeech"
OUTPUT_DIR = "/users/PAS2030/srishti/asr_fairness/data"
CV_DIR = "/users/PAS2030/srishti/bootcamp/data/commonvoice/cv-corpus-24.0-2025-12-05/en"
SEED = 42
N_SPLITS = 5
MIN_GROUP_SIZE = 50


def discover_speaker_column(meta_path):
    """Discover the speaker/participant ID column from Fair-Speech metadata TSV.

    The Fair-Speech dataset has ~593 participants, each with a de-identified
    unique number. The column with ~500-700 unique values is the speaker ID.
    The hash_name column is per-utterance and must NOT be used as speaker ID.
    """
    print(f"\n{'='*60}")
    print("SPEAKER ID DISCOVERY")
    print(f"{'='*60}")
    print(f"Reading: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames
        rows = list(reader)

    print(f"TSV columns ({len(headers)}): {headers}")
    print(f"Total rows: {len(rows):,}")

    # Compute unique value counts per column
    candidates = []
    for col in headers:
        unique_count = len(set(row[col] for row in rows if row[col].strip()))
        print(f"  {col:40s}: {unique_count:,} unique values")
        # Speaker column should have ~500-700 unique values
        if 400 <= unique_count <= 800 and col != "hash_name":
            candidates.append((col, unique_count))

    if not candidates:
        # Print all column cardinalities and raise error
        print("\nERROR: No column found with ~500-700 unique values.")
        print("Column cardinalities:")
        for col in headers:
            unique_count = len(set(row[col] for row in rows if row[col].strip()))
            print(f"  {col}: {unique_count}")
        sys.exit(1)

    # Pick the best candidate (closest to 593)
    speaker_col = min(candidates, key=lambda x: abs(x[1] - 593))
    print(f"\nDiscovered speaker column: '{speaker_col[0]}' "
          f"({speaker_col[1]} unique values)")

    # Build hash_name -> speaker_id mapping
    mapping = {}
    for row in rows:
        hash_name = row["hash_name"].strip()
        speaker_id = row[speaker_col[0]].strip()
        if hash_name and speaker_id:
            mapping[hash_name] = speaker_id

    print(f"Built mapping: {len(mapping):,} utterances -> "
          f"{len(set(mapping.values())):,} speakers")

    return mapping


def split_fairspeech(df, seed):
    """Perform speaker-level stratified 80/20 split on Fair-Speech data.

    Uses StratifiedGroupKFold with N_SPLITS=5 folds, taking the first fold
    as eval (~20%) and the rest as train (~80%). This ensures:
    - No speaker appears in both train and eval (speaker-disjoint)
    - Ethnicity distribution is preserved across splits (stratified)
    """
    print(f"\n{'='*60}")
    print("STRATIFIED GROUP SPLIT")
    print(f"{'='*60}")

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    train_idx, eval_idx = next(sgkf.split(
        X=df,
        y=df["ethnicity"],
        groups=df["speaker_id"]
    ))

    df_train = df.iloc[train_idx].copy()
    df_eval = df.iloc[eval_idx].copy()

    # Verify zero speaker overlap
    train_speakers = set(df_train["speaker_id"])
    eval_speakers = set(df_eval["speaker_id"])
    overlap = train_speakers & eval_speakers
    assert len(overlap) == 0, (
        f"Speaker leakage detected! {len(overlap)} speakers in both splits: "
        f"{list(overlap)[:5]}..."
    )

    print(f"Train: {len(df_train):,} utterances, {len(train_speakers):,} speakers")
    print(f"Eval:  {len(df_eval):,} utterances, {len(eval_speakers):,} speakers")
    print(f"Speaker overlap: {len(overlap)} (verified zero)")

    return df_train, df_eval


def print_distribution_stats(df_train, df_eval):
    """Print per-axis demographic distribution comparison (train% vs eval%)."""
    print(f"\n{'='*60}")
    print("DISTRIBUTION STATISTICS")
    print(f"{'='*60}")

    for axis in ["ethnicity", "age", "first_language", "gender"]:
        if axis not in df_train.columns:
            continue

        print(f"\n{'~'*50}")
        print(f"{axis.upper()} distribution:")
        print(f"{'~'*50}")

        train_dist = df_train[axis].value_counts(normalize=True).sort_index()
        eval_dist = df_eval[axis].value_counts(normalize=True).sort_index()

        comparison = pd.DataFrame({
            "train%": (train_dist * 100).round(2),
            "eval%": (eval_dist * 100).round(2),
        }).fillna(0)
        comparison["diff_pp"] = (comparison["train%"] - comparison["eval%"]).round(2)
        print(comparison.to_string())

    # Flag groups below MIN_GROUP_SIZE in eval
    print(f"\n{'~'*50}")
    print(f"EVAL GROUP SIZE CHECK (MIN_GROUP_SIZE={MIN_GROUP_SIZE})")
    print(f"{'~'*50}")

    flagged = False
    for axis in ["ethnicity", "age", "first_language", "gender"]:
        if axis not in df_eval.columns:
            continue
        counts = df_eval[axis].value_counts()
        small_groups = counts[counts < MIN_GROUP_SIZE]
        if len(small_groups) > 0:
            flagged = True
            for group, count in small_groups.items():
                print(f"  WARNING: {axis}/{group} has only {count} utterances "
                      f"in eval (< {MIN_GROUP_SIZE})")

    if not flagged:
        print("  All groups meet MIN_GROUP_SIZE threshold in eval split.")

    # Speaker counts per split
    print(f"\n{'~'*50}")
    print("SPEAKER COUNTS")
    print(f"{'~'*50}")
    print(f"  Train speakers: {df_train['speaker_id'].nunique():,}")
    print(f"  Eval speakers:  {df_eval['speaker_id'].nunique():,}")


def check_cv_manifest(cv_manifest_path):
    """Verify Common Voice test manifest exists and print accent distribution."""
    print(f"\n{'='*60}")
    print("COMMON VOICE TEST SPLIT")
    print(f"{'='*60}")
    print("NOTE: CV test split is retained as-is from cv_test_manifest.csv.")
    print("      No new CV splits are created.")

    if os.path.isfile(cv_manifest_path):
        print(f"  FOUND: {cv_manifest_path}")
        df_cv = pd.read_csv(cv_manifest_path)
        print(f"  Utterances: {len(df_cv):,}")

        if "accent" in df_cv.columns:
            print(f"\n  Accent distribution:")
            accent_dist = df_cv["accent"].value_counts()
            for accent, count in accent_dist.items():
                print(f"    {accent:30s}: {count:5,}")
        else:
            print("  WARNING: No 'accent' column found in CV manifest.")
    else:
        print(f"  WARNING: CV manifest not found at {cv_manifest_path}")
        print("  Run prepare_dataset.py first to create it.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Fair-Speech train/eval splits with demographic labels."
    )
    parser.add_argument(
        "--fs-manifest",
        default=os.path.join(OUTPUT_DIR, "fs_manifest.csv"),
        help="Path to Fair-Speech manifest CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--cv-manifest",
        default=os.path.join(OUTPUT_DIR, "cv_test_manifest.csv"),
        help="Path to Common Voice test manifest CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory for train/eval CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for split reproducibility (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"{'='*60}")
    print("PREPARE FAIR-SPEECH TRAIN/EVAL SPLITS")
    print(f"{'='*60}")
    print(f"FS manifest:  {args.fs_manifest}")
    print(f"CV manifest:  {args.cv_manifest}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Seed:         {args.seed}")
    print(f"N_SPLITS:     {N_SPLITS}")

    os.makedirs(args.output_dir, exist_ok=True)

    # -- Step 1: Discover speaker IDs from raw TSV --------------------------
    meta_path = os.path.join(FS_DIR, "fairspeech_metadata.tsv")
    speaker_mapping = discover_speaker_column(meta_path)

    # -- Step 2: Load manifest and add speaker_id ---------------------------
    print(f"\n{'='*60}")
    print("LOADING MANIFEST")
    print(f"{'='*60}")

    df = pd.read_csv(args.fs_manifest)
    print(f"Loaded: {len(df):,} utterances from {args.fs_manifest}")

    # Join speaker_id by matching utterance_id (hash_name) to mapping
    df["speaker_id"] = df["utterance_id"].map(speaker_mapping)

    missing_speaker = df["speaker_id"].isna().sum()
    if missing_speaker > 0:
        print(f"  WARNING: {missing_speaker} utterances have no speaker_id mapping.")
        print(f"  Dropping these utterances from splits.")
        df = df.dropna(subset=["speaker_id"])

    print(f"Utterances with speaker_id: {len(df):,}")
    print(f"Unique speakers: {df['speaker_id'].nunique():,}")

    # -- Step 3: Perform stratified group split -----------------------------
    df_train, df_eval = split_fairspeech(df, args.seed)

    # -- Step 4: Save outputs -----------------------------------------------
    train_path = os.path.join(args.output_dir, "fs_train.csv")
    eval_path = os.path.join(args.output_dir, "fs_eval.csv")

    df_train.to_csv(train_path, index=False)
    df_eval.to_csv(eval_path, index=False)

    print(f"\nSaved: {train_path} ({len(df_train):,} utterances)")
    print(f"Saved: {eval_path} ({len(df_eval):,} utterances)")

    # -- Step 5: Print distribution stats -----------------------------------
    print_distribution_stats(df_train, df_eval)

    # -- Step 6: Check CV manifest ------------------------------------------
    check_cv_manifest(args.cv_manifest)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
