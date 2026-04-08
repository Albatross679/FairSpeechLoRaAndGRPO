"""
Generate training manifests for Fair-Speech and Common Voice datasets.

Creates CSV manifests with standardized columns (utterance_id, audio_path,
sentence, demographic columns) from the raw metadata files, using local
paths on this machine.

Usage:
    python scripts/training/generate_manifests.py
    python scripts/training/generate_manifests.py --output_dir outputs/manifests
"""

import argparse
import csv
import os
import sys

import pandas as pd

# -- Paths --------------------------------------------------------------------

FS_DIR = "datasets/meta_speech_fairness"
FS_METADATA = os.path.join(FS_DIR, "downloaded_file.tsv")
FS_AUDIO_DIR = os.path.join(FS_DIR, "asr_fairness_audio")

CV_DIR = "datasets/cv-corpus-25.0-2026-03-09/en"
CV_CLIPS_DIR = os.path.join(CV_DIR, "clips")

LS_DIR = "datasets/librispeech/LibriSpeech"
LS_SPEAKERS = os.path.join(LS_DIR, "SPEAKERS.TXT")

DEFAULT_OUTPUT_DIR = "outputs/manifests"

# -- Ethnicity standardization (from prepare_fairspeech.py) -------------------

ETHNICITY_MAP = {
    "Black or African American": "Black/AA",
    "White": "White",
    "Native American, American Indian, or Alaska Native": "Native American",
    "Asian, South Asian or Asian American": "Asian",
    "Hispanic, Latino, or Spanish": "Hispanic",
    "Native Hawaiian or Other Pacific Islander": "Pacific Islander",
    "Middle Eastern or North African": "Middle Eastern",
}

# -- Accent standardization (from prepare_dataset.py) -------------------------

ACCENT_MAP = {
    "United States English": "us",
    "England English": "england",
    "India and South Asia (India, Pakistan, Sri Lanka)": "indian",
    "Canadian English": "canada",
    "Australian English": "australia",
    "Southern African (South Africa, Zimbabwe, Namibia)": "african",
    "Scottish English": "scotland",
    "Irish English": "ireland",
    "Hong Kong English": "hongkong",
    "New Zealand English": "newzealand",
    "Filipino": "filipino",
    "Malaysian English": "malaysia",
    "Welsh English": "wales",
    "Singaporean English": "singapore",
}

MIN_GROUP_SIZE = 50


def generate_fairspeech_manifest(output_dir):
    """Generate Fair-Speech manifest CSV from downloaded_file.tsv."""
    print(f"\n{'='*60}")
    print("Generating Fair-Speech manifest")
    print(f"{'='*60}")

    df = pd.read_csv(FS_METADATA, sep="\t")
    print(f"  Raw rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    manifest_rows = []
    missing_audio = 0

    for _, row in df.iterrows():
        hash_name = str(row["hash_name"])
        audio_path = os.path.join(FS_AUDIO_DIR, f"{hash_name}.wav")

        # Check audio file exists
        if not os.path.isfile(audio_path):
            missing_audio += 1
            continue

        # Standardize ethnicity
        raw_ethnicity = str(row.get("ethnicity", "")).strip()
        ethnicity = ETHNICITY_MAP.get(raw_ethnicity, raw_ethnicity)

        transcript = str(row.get("transcription", "")).strip()
        if not transcript:
            continue

        manifest_rows.append({
            "utterance_id": f"fs_{hash_name}",
            "audio_path": os.path.abspath(audio_path),
            "sentence": transcript,
            "ethnicity": ethnicity,
            "gender": str(row.get("gender", "")).strip(),
            "age": str(row.get("age", "")).strip(),
            "first_language": str(row.get("first_language", "")).strip(),
            "dataset": "fairspeech",
        })

    if missing_audio > 0:
        print(f"  WARNING: {missing_audio} audio files not found")

    manifest_df = pd.DataFrame(manifest_rows)
    print(f"  Valid rows: {len(manifest_df):,}")

    # Print ethnicity distribution
    if "ethnicity" in manifest_df.columns:
        print(f"\n  Ethnicity distribution:")
        for group, count in manifest_df["ethnicity"].value_counts().items():
            print(f"    {group}: {count:,}")

    # Filter out groups smaller than MIN_GROUP_SIZE
    group_sizes = manifest_df["ethnicity"].value_counts()
    valid_groups = group_sizes[group_sizes >= MIN_GROUP_SIZE].index
    small_groups = group_sizes[group_sizes < MIN_GROUP_SIZE]
    if len(small_groups) > 0:
        print(f"\n  Dropping {len(small_groups)} groups with < {MIN_GROUP_SIZE} samples:")
        for g, c in small_groups.items():
            print(f"    {g}: {c}")
        manifest_df = manifest_df[manifest_df["ethnicity"].isin(valid_groups)]
        print(f"  After filtering: {len(manifest_df):,} rows")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "fs_train.csv")
    manifest_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(manifest_df):,} rows)")
    return out_path


def generate_cv_manifest(output_dir, split="train"):
    """Generate Common Voice manifest CSV from train/dev/test TSV."""
    print(f"\n{'='*60}")
    print(f"Generating Common Voice manifest (split={split})")
    print(f"{'='*60}")

    tsv_path = os.path.join(CV_DIR, f"{split}.tsv")
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"  Raw rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    manifest_rows = []
    missing_audio = 0

    for _, row in df.iterrows():
        clip_path = str(row.get("path", "")).strip()
        if not clip_path:
            continue

        audio_path = os.path.join(CV_CLIPS_DIR, clip_path)
        if not os.path.isfile(audio_path):
            missing_audio += 1
            continue

        transcript = str(row.get("sentence", "")).strip()
        if not transcript:
            continue

        # Standardize accent
        raw_accent = str(row.get("accents", "")).strip()
        accent = ACCENT_MAP.get(raw_accent, raw_accent) if raw_accent and raw_accent != "nan" else ""

        # Standardize gender
        raw_gender = str(row.get("gender", "")).strip()
        gender = raw_gender if raw_gender and raw_gender != "nan" else ""

        utterance_id = clip_path.replace(".mp3", "").replace(".wav", "")

        # Preserve client_id as speaker_id for speaker-disjoint splits
        speaker_id = str(row.get("client_id", "")).strip()

        manifest_rows.append({
            "utterance_id": f"cv_{utterance_id}",
            "audio_path": os.path.abspath(audio_path),
            "sentence": transcript,
            "accent": accent,
            "gender": gender,
            "age": str(row.get("age", "")).strip() if str(row.get("age", "")).strip() != "nan" else "",
            "speaker_id": speaker_id,
            "dataset": "commonvoice",
        })

    if missing_audio > 0:
        print(f"  WARNING: {missing_audio} audio files not found")

    manifest_df = pd.DataFrame(manifest_rows)
    print(f"  Valid rows: {len(manifest_df):,}")

    # Print accent distribution
    accented = manifest_df[manifest_df["accent"] != ""]
    if len(accented) > 0:
        print(f"\n  Accent distribution (non-empty):")
        for group, count in accented["accent"].value_counts().head(15).items():
            print(f"    {group}: {count:,}")
        print(f"  Unlabeled: {len(manifest_df) - len(accented):,}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"cv_{split}.csv")
    manifest_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(manifest_df):,} rows)")
    return out_path


def generate_librispeech_manifest(output_dir, split="test-clean"):
    """Generate LibriSpeech manifest CSV from directory structure + SPEAKERS.TXT."""
    print(f"\n{'='*60}")
    print(f"Generating LibriSpeech manifest (split={split})")
    print(f"{'='*60}")

    split_dir = os.path.join(LS_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"  SKIP: {split_dir} does not exist")
        return None

    # Parse SPEAKERS.TXT for gender metadata
    speaker_gender = {}
    if os.path.isfile(LS_SPEAKERS):
        with open(LS_SPEAKERS) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    speaker_gender[parts[0]] = parts[1]

    manifest_rows = []
    # Walk speaker/chapter/utterance hierarchy
    for speaker_id in sorted(os.listdir(split_dir)):
        speaker_dir = os.path.join(split_dir, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue
        gender = speaker_gender.get(speaker_id, "")
        for chapter_id in sorted(os.listdir(speaker_dir)):
            chapter_dir = os.path.join(speaker_dir, chapter_id)
            if not os.path.isdir(chapter_dir):
                continue
            # Read transcripts
            trans_file = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}.trans.txt")
            transcripts = {}
            if os.path.isfile(trans_file):
                with open(trans_file) as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            transcripts[parts[0]] = parts[1]
            # Find audio files
            for fname in sorted(os.listdir(chapter_dir)):
                if not fname.endswith(".flac"):
                    continue
                utt_id = fname.replace(".flac", "")
                transcript = transcripts.get(utt_id, "")
                if not transcript:
                    continue
                audio_path = os.path.join(chapter_dir, fname)
                manifest_rows.append({
                    "utterance_id": f"ls_{utt_id}",
                    "audio_path": os.path.abspath(audio_path),
                    "sentence": transcript,
                    "speaker_id": speaker_id,
                    "gender": gender,
                    "dataset": "librispeech",
                })

    manifest_df = pd.DataFrame(manifest_rows)
    print(f"  Valid rows: {len(manifest_df):,}")
    print(f"  Unique speakers: {manifest_df['speaker_id'].nunique()}")

    if "gender" in manifest_df.columns:
        print(f"\n  Gender distribution:")
        for group, count in manifest_df["gender"].value_counts().items():
            print(f"    {group}: {count:,}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"ls_{split.replace('-', '_')}.csv")
    manifest_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(manifest_df):,} rows)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate training manifests")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print(f"Output directory: {args.output_dir}")

    # Generate Fair-Speech manifest
    fs_path = generate_fairspeech_manifest(args.output_dir)

    # Generate Common Voice manifests (train for fine-tuning, dev for eval)
    cv_train_path = generate_cv_manifest(args.output_dir, split="train")
    cv_dev_path = generate_cv_manifest(args.output_dir, split="dev")

    # Generate LibriSpeech manifest (test-clean for held-out evaluation)
    ls_path = generate_librispeech_manifest(args.output_dir, split="test-clean")

    print(f"\n{'='*60}")
    print("All manifests generated")
    print(f"{'='*60}")
    print(f"  Fair-Speech train: {fs_path}")
    print(f"  Common Voice train: {cv_train_path}")
    print(f"  Common Voice dev:   {cv_dev_path}")
    if ls_path:
        print(f"  LibriSpeech test-clean: {ls_path}")


if __name__ == "__main__":
    main()
