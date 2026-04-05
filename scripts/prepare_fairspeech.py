#!/usr/bin/env python3
"""
Prepare Fair-Speech dataset manifest for fairness evaluation.

Reads fairspeech_metadata.tsv, verifies audio files exist, standardizes
demographic labels, applies text normalization, and produces a CSV manifest.

Usage:
    python scripts/prepare_fairspeech.py
"""

import csv
import json
import os
import sys
from collections import Counter

# ── Config ──────────────────────────────────────────────────────────────────
FS_DIR = "/users/PAS2030/srishti/asr_fairness/data/fairspeech"
AUDIO_DIR = os.path.join(FS_DIR, "asr_fairness_audio")
OUTPUT_DIR = "/users/PAS2030/srishti/asr_fairness/data"
MIN_GROUP_SIZE = 50

# ── Text normalization (same as CV pipeline) ────────────────────────────────
try:
    from whisper.normalizers import EnglishTextNormalizer
except ImportError:
    from whisper_normalizer.english import EnglishTextNormalizer
_normalizer = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    """Normalize text for WER using Whisper's English normalizer."""
    if not text or not isinstance(text, str):
        return ""
    return _normalizer(text)


# ── Ethnicity label standardization ─────────────────────────────────────────
# Shorten the long Fair-Speech labels for cleaner columns / plots
ETHNICITY_MAP = {
    "Black or African American": "Black/AA",
    "White": "White",
    "Native American, American Indian, or Alaska Native": "Native American",
    "Asian, South Asian or Asian American": "Asian",
    "Hispanic, Latino, or Spanish": "Hispanic",
    "Native Hawaiian or Other Pacific Islander": "Pacific Islander",
    "Middle Eastern or North African": "Middle Eastern",
}

# ── Age label standardization ──────────────────────────────────────────────
AGE_MAP = {
    "18 - 22": "18-22",
    "23 - 30": "23-30",
    "31 - 45": "31-45",
    "46 - 65": "46-65",
}

# ── L1 grouping (keep individual languages with n >= 50; rest → Other) ─────
# Based on actual data: English(21528), Spanish(1862), Mandarin(422),
# Hindi(233), Cantonese(191), Tagalog(190), French(174), Portuguese(172),
# Other(170), Arabic(140), Indonesian(137), Urdu(136), German(136),
# Russian(98), Malay(98), Dutch(93), Vietnamese(86), Korean(69),
# Nepali(50), Marathi(50), Filipino(50), ...
L1_KEEP = {
    "English", "Spanish", "Mandarin", "Hindi", "Cantonese", "Tagalog",
    "French", "Portuguese", "Arabic", "Indonesian", "Urdu", "German",
    "Russian", "Malay", "Dutch", "Vietnamese", "Korean", "Nepali",
    "Marathi", "Filipino",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Read metadata ───────────────────────────────────────────────
    meta_path = os.path.join(FS_DIR, "fairspeech_metadata.tsv")
    rows = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    print(f"Total entries in metadata: {len(rows):,}")

    # ── Build manifest ──────────────────────────────────────────────
    manifest = []
    missing_audio = 0

    for row in rows:
        hash_name = row["hash_name"].strip()
        audio_path = os.path.join(AUDIO_DIR, f"{hash_name}.wav")

        if not os.path.isfile(audio_path):
            missing_audio += 1
            continue

        # Standardize demographics
        raw_ethnicity = row.get("ethnicity", "").strip()
        raw_gender = row.get("gender", "").strip().lower()
        raw_age = row.get("age", "").strip()
        raw_l1 = row.get("first_language", "").strip()
        raw_ses = row.get("socioeconomic_bkgd", "").strip()
        raw_text = row.get("transcription", "").strip()

        ethnicity = ETHNICITY_MAP.get(raw_ethnicity, "")
        gender = raw_gender if raw_gender in ("male", "female") else ""
        age = AGE_MAP.get(raw_age, "")
        first_language = raw_l1 if raw_l1 in L1_KEEP else ("Other" if raw_l1 else "")
        # Also keep a column for the L1 broad grouping used in H5
        l1_group = raw_l1 if raw_l1 in {"English", "Spanish", "Mandarin", "Hindi"} else (
            "Other" if raw_l1 else ""
        )
        socioeconomic = raw_ses if raw_ses in ("Low", "Medium", "Affluent") else ""

        # Normalize reference text (same normalizer as CV pipeline)
        sentence_normalized = normalize_text(raw_text)

        manifest.append({
            "utterance_id": hash_name,
            "audio_path": audio_path,
            "sentence": sentence_normalized,
            "sentence_raw": raw_text,
            "ethnicity": ethnicity,
            "ethnicity_raw": raw_ethnicity,
            "gender": gender,
            "age": age,
            "first_language": first_language,
            "first_language_raw": raw_l1,
            "l1_group": l1_group,
            "socioeconomic": socioeconomic,
        })

    print(f"Audio found:              {len(manifest):,}")
    print(f"Audio missing:            {missing_audio:,}")

    # ── Save full manifest ──────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "fs_manifest.csv")
    fieldnames = [
        "utterance_id", "audio_path", "sentence", "sentence_raw",
        "ethnicity", "ethnicity_raw", "gender", "age",
        "first_language", "first_language_raw", "l1_group",
        "socioeconomic",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nSaved: {csv_path} ({len(manifest):,} utterances)")

    # ── Print demographic stats ─────────────────────────────────────
    summary = {"total": len(manifest), "demographics": {}}

    for axis, field in [
        ("ethnicity", "ethnicity"),
        ("gender", "gender"),
        ("age", "age"),
        ("first_language", "first_language"),
        ("l1_group", "l1_group"),
        ("socioeconomic", "socioeconomic"),
    ]:
        counts = Counter(r[field] for r in manifest if r[field])
        labeled = sum(counts.values())

        print(f"\n{'─'*50}")
        print(f"{axis.upper()} — {labeled:,} labeled ({100*labeled/len(manifest):.1f}%)")
        print(f"{'─'*50}")

        viable_groups = {}
        for group, count in counts.most_common():
            marker = "✓" if count >= MIN_GROUP_SIZE else "✗"
            print(f"  {group:30s}: {count:5,} {marker}")
            if count >= MIN_GROUP_SIZE:
                viable_groups[group] = count

        summary["demographics"][axis] = {
            "total_labeled": labeled,
            "viable_groups": viable_groups,
        }

    # ── Save summary ────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "fs_test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # ── Spot-check: verify 10 random audio files ────────────────────
    import random
    random.seed(42)
    sample = random.sample(manifest, min(10, len(manifest)))
    print(f"\nSpot-check ({len(sample)} samples):")
    for s in sample:
        exists = os.path.isfile(s["audio_path"])
        size = os.path.getsize(s["audio_path"]) if exists else 0
        print(f"  {s['utterance_id']}: exists={exists}, size={size:,}B, "
              f"eth={s['ethnicity']}, age={s['age']}, l1={s['first_language']}")


if __name__ == "__main__":
    main()
