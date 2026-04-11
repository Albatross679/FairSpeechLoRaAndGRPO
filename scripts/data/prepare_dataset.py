#!/usr/bin/env python3
"""
Prepare Common Voice v24.0 test split manifest for fairness evaluation.

Reads test.tsv, verifies audio files exist, standardizes demographic labels,
and produces a single CSV manifest with full demographic metadata.

Usage:
    python scripts/data/prepare_dataset.py
"""

import csv
import json
import os
import sys
from collections import Counter

# ── Config ──────────────────────────────────────────────────────────────────
CV_DIR = "/users/PAS2030/srishti/bootcamp/data/commonvoice/cv-corpus-24.0-2025-12-05/en"
OUTPUT_DIR = "/users/PAS2030/srishti/asr_fairness/data"
MIN_GROUP_SIZE = 50

# Gender label mapping
GENDER_MAP = {
    "male_masculine": "male",
    "male": "male",
    "female_feminine": "female",
    "female": "female",
}

# Accent label mapping (standardize to shorter names)
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

# Age label mapping (standardize spelling)
AGE_MAP = {
    "teens": "teens",
    "twenties": "twenties",
    "thirties": "thirties",
    "fourties": "forties",   # Fix CV typo
    "forties": "forties",
    "fifties": "fifties",
    "sixties": "sixties",
    "seventies": "seventies",
    "eighties": "eighties",
    "nineties": "nineties",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clips_dir = os.path.join(CV_DIR, "clips")

    # ── Read test.tsv ───────────────────────────────────────────────
    test_tsv = os.path.join(CV_DIR, "test.tsv")
    rows = []
    with open(test_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    print(f"Total entries in test.tsv: {len(rows):,}")

    # ── Build manifest ──────────────────────────────────────────────
    manifest = []
    missing_audio = 0

    for row in rows:
        audio_file = row["path"]
        audio_path = os.path.join(clips_dir, audio_file)

        if not os.path.isfile(audio_path):
            missing_audio += 1
            continue

        # Standardize demographics
        raw_gender = row.get("gender", "").strip().lower()
        raw_accent = row.get("accents", "").strip()
        raw_age = row.get("age", "").strip().lower()

        gender = GENDER_MAP.get(raw_gender, "")
        accent = ACCENT_MAP.get(raw_accent, raw_accent.lower().replace(" ", "_") if raw_accent else "")
        age = AGE_MAP.get(raw_age, "")

        manifest.append({
            "utterance_id": audio_file.replace(".mp3", ""),
            "audio_path": audio_path,
            "sentence": row["sentence"],
            "gender": gender,
            "accent": accent,
            "accent_raw": raw_accent,
            "age": age,
            "client_id": row.get("client_id", "")[:8],  # Truncated for privacy
        })

    print(f"Audio found:              {len(manifest):,}")
    print(f"Audio missing:            {missing_audio:,}")

    # ── Save full manifest ──────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "cv_test_manifest.csv")
    fieldnames = ["utterance_id", "audio_path", "sentence", "gender", "accent", "accent_raw", "age", "client_id"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nSaved: {csv_path} ({len(manifest):,} utterances)")

    # ── Print demographic stats ─────────────────────────────────────
    summary = {"total": len(manifest), "demographics": {}}

    for axis, field in [("gender", "gender"), ("accent", "accent"), ("age", "age")]:
        counts = Counter(r[field] for r in manifest if r[field])
        labeled = sum(counts.values())

        print(f"\n{'─'*50}")
        print(f"{axis.upper()} — {labeled:,} labeled ({100*labeled/len(manifest):.1f}%)")
        print(f"{'─'*50}")

        viable_groups = {}
        for group, count in counts.most_common():
            marker = "✓" if count >= MIN_GROUP_SIZE else "✗"
            print(f"  {group:20s}: {count:5,} {marker}")
            if count >= MIN_GROUP_SIZE:
                viable_groups[group] = count

        summary["demographics"][axis] = {
            "total_labeled": labeled,
            "viable_groups": viable_groups,
        }

    # ── Save summary ────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "cv_test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
