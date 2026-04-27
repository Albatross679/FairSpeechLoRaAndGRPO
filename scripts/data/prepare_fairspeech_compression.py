#!/usr/bin/env python3
"""Build FairSpeech manifests and derived audio variants for compression tests.

This script is infrastructure-only: by default it writes manifests and summaries
without doing an expensive full audio conversion. Pass ``--generate-audio`` when
ready to create WAV files for the selected variants.

Example dry run / manifest build:
    python3 scripts/data/prepare_fairspeech_compression.py

Example pilot audio generation:
    python3 scripts/data/prepare_fairspeech_compression.py \
      --max-samples 21 --generate-audio --variants all
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = PROJECT_ROOT / "datasets" / "meta_speech_fairness" / "downloaded_file.tsv"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "datasets" / "meta_speech_fairness" / "asr_fairness_audio"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "fairspeech_compression"
EXPECTED_FULL_COUNT = 26_471
TARGET_SAMPLE_RATE = 16_000

ETHNICITY_MAP = {
    "Black or African American": "Black/AA",
    "White": "White",
    "Native American, American Indian, or Alaska Native": "Native American",
    "Asian, South Asian or Asian American": "Asian",
    "Hispanic, Latino, or Spanish": "Hispanic",
    "Native Hawaiian or Other Pacific Islander": "Pacific Islander",
    "Middle Eastern or North African": "Middle Eastern",
}

AGE_MAP = {
    "18 - 22": "18-22",
    "23 - 30": "23-30",
    "31 - 45": "31-45",
    "46 - 65": "46-65",
}

L1_KEEP = {
    "English", "Spanish", "Mandarin", "Hindi", "Cantonese", "Tagalog",
    "French", "Portuguese", "Arabic", "Indonesian", "Urdu", "German",
    "Russian", "Malay", "Dutch", "Vietnamese", "Korean", "Nepali",
    "Marathi", "Filipino",
}


@dataclass(frozen=True)
class AudioVariant:
    key: str
    label: str
    pipeline: str
    kind: str
    bottleneck_rate: int | None = None
    mp3_bitrate: str | None = None


VARIANTS: "OrderedDict[str, AudioVariant]" = OrderedDict(
    (
        (
            "baseline",
            AudioVariant(
                key="baseline",
                label="Baseline 16 kHz PCM WAV",
                pipeline="16 kHz PCM WAV -> model",
                kind="baseline",
            ),
        ),
        (
            "bottleneck_12k",
            AudioVariant(
                key="bottleneck_12k",
                label="12 kHz bottleneck",
                pipeline="16 kHz -> 12 kHz -> 16 kHz -> model",
                kind="bottleneck",
                bottleneck_rate=12_000,
            ),
        ),
        (
            "bottleneck_8k",
            AudioVariant(
                key="bottleneck_8k",
                label="8 kHz bottleneck",
                pipeline="16 kHz -> 8 kHz -> 16 kHz -> model",
                kind="bottleneck",
                bottleneck_rate=8_000,
            ),
        ),
        (
            "mp3_64k",
            AudioVariant(
                key="mp3_64k",
                label="MP3 64 kbps",
                pipeline="16 kHz WAV -> MP3 64 kbps -> 16 kHz WAV -> model",
                kind="mp3_roundtrip",
                mp3_bitrate="64k",
            ),
        ),
        (
            "mp3_32k",
            AudioVariant(
                key="mp3_32k",
                label="MP3 32 kbps",
                pipeline="16 kHz WAV -> MP3 32 kbps -> 16 kHz WAV -> model",
                kind="mp3_roundtrip",
                mp3_bitrate="32k",
            ),
        ),
        (
            "mp3_16k",
            AudioVariant(
                key="mp3_16k",
                label="MP3 16 kbps",
                pipeline="16 kHz WAV -> MP3 16 kbps -> 16 kHz WAV -> model",
                kind="mp3_roundtrip",
                mp3_bitrate="16k",
            ),
        ),
    )
)


def normalize_text(text: str) -> str:
    """Normalize reference text using Whisper if available, otherwise a light fallback."""
    if not text:
        return ""
    try:
        from whisper.normalizers import EnglishTextNormalizer  # type: ignore

        return EnglishTextNormalizer()(text)
    except Exception:
        text = text.lower()
        text = re.sub(r"[^a-z0-9' ]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def wav_metadata(path: Path) -> dict[str, int | float | str]:
    """Read WAV header metadata with the stdlib wave module."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth_bytes = wf.getsampwidth()
        frames = wf.getnframes()
    bit_depth = sampwidth_bytes * 8
    duration = frames / sample_rate if sample_rate else 0.0
    bitrate = sample_rate * channels * bit_depth
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "bit_depth": bit_depth,
        "frames": frames,
        "duration_seconds": round(duration, 6),
        "bitrate_bps": bitrate,
        "codec": f"PCM_S{bit_depth}LE" if bit_depth else "PCM",
    }


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def standardize_row(row: dict[str, str], audio_dir: Path) -> dict[str, str | int | float]:
    hash_name = row["hash_name"].strip()
    source_audio = audio_dir / f"{hash_name}.wav"
    meta = wav_metadata(source_audio)

    raw_ethnicity = row.get("ethnicity", "").strip()
    raw_gender = row.get("gender", "").strip().lower()
    raw_age = row.get("age", "").strip()
    raw_l1 = row.get("first_language", "").strip()
    raw_ses = row.get("socioeconomic_bkgd", "").strip()
    raw_text = row.get("transcription", "").strip()

    first_language = raw_l1 if raw_l1 in L1_KEEP else ("Other" if raw_l1 else "")
    l1_group = raw_l1 if raw_l1 in {"English", "Spanish", "Mandarin", "Hindi"} else (
        "Other" if raw_l1 else ""
    )

    return {
        "utterance_id": hash_name,
        "hash_name": hash_name,
        "audio_path": str(source_audio),
        "source_audio_path": str(source_audio),
        "sentence": normalize_text(raw_text),
        "sentence_raw": raw_text,
        "ethnicity": ETHNICITY_MAP.get(raw_ethnicity, raw_ethnicity),
        "ethnicity_raw": raw_ethnicity,
        "gender": raw_gender if raw_gender in {"male", "female"} else "",
        "age": AGE_MAP.get(raw_age, raw_age),
        "first_language": first_language,
        "first_language_raw": raw_l1,
        "l1_group": l1_group,
        "socioeconomic": raw_ses if raw_ses in {"Low", "Medium", "Affluent"} else "",
        "source_sample_rate": meta["sample_rate"],
        "source_channels": meta["channels"],
        "source_bit_depth": meta["bit_depth"],
        "source_codec": meta["codec"],
        "source_bitrate_bps": meta["bitrate_bps"],
        "source_duration_seconds": meta["duration_seconds"],
        "source_num_frames": meta["frames"],
        "duration_seconds": meta["duration_seconds"],
    }


def build_source_manifest(metadata_path: Path, audio_dir: Path, max_samples: int | None = None) -> list[dict[str, str | int | float]]:
    raw_rows = read_metadata(metadata_path)
    if max_samples is not None:
        raw_rows = raw_rows[:max_samples]

    rows: list[dict[str, str | int | float]] = []
    missing: list[str] = []
    for row in raw_rows:
        hash_name = row["hash_name"].strip()
        audio_path = audio_dir / f"{hash_name}.wav"
        if not audio_path.is_file():
            missing.append(hash_name)
            continue
        rows.append(standardize_row(row, audio_dir))
    if missing:
        print(f"WARNING: skipped {len(missing):,} metadata rows with missing WAV files", file=sys.stderr)
    return rows


def parse_variant_keys(raw: str) -> list[str]:
    if raw == "all":
        return list(VARIANTS)
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [key for key in keys if key not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Available: {list(VARIANTS)}")
    return keys


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_audio_path(output_dir: Path, variant_key: str, utterance_id: str) -> Path:
    return output_dir / "audio" / variant_key / f"{utterance_id}.wav"


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def ffmpeg_base() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg is required for audio generation but was not found on PATH")
    return exe


def generate_variant_audio(src: Path, dst: Path, spec: AudioVariant, overwrite: bool = False, dry_run: bool = False) -> str:
    """Create one derived WAV. Returns status: created/skipped/dry-run."""
    if dst.exists() and not overwrite:
        return "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_base()

    if spec.kind == "baseline":
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
            "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst),
        ]
        run_command(cmd, dry_run=dry_run)
        return "dry-run" if dry_run else "created"

    if spec.kind == "bottleneck":
        if spec.bottleneck_rate is None:
            raise ValueError(f"Variant {spec.key} missing bottleneck_rate")
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
            "-ac", "1", "-af", f"aresample={spec.bottleneck_rate},aresample={TARGET_SAMPLE_RATE}",
            "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst),
        ]
        run_command(cmd, dry_run=dry_run)
        return "dry-run" if dry_run else "created"

    if spec.kind == "mp3_roundtrip":
        if spec.mp3_bitrate is None:
            raise ValueError(f"Variant {spec.key} missing mp3_bitrate")
        with tempfile.TemporaryDirectory(prefix="fs-mp3-") as tmp:
            tmp_mp3 = Path(tmp) / f"{src.stem}.{spec.key}.mp3"
            encode = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "libmp3lame",
                "-b:a", spec.mp3_bitrate, str(tmp_mp3),
            ]
            decode = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(tmp_mp3),
                "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst),
            ]
            run_command(encode, dry_run=dry_run)
            run_command(decode, dry_run=dry_run)
        return "dry-run" if dry_run else "created"

    raise ValueError(f"Unknown variant kind: {spec.kind}")


def variant_manifest_rows(
    source_rows: list[dict[str, str | int | float]],
    output_dir: Path,
    variant_key: str,
    generated_audio: bool,
) -> list[dict[str, object]]:
    spec = VARIANTS[variant_key]
    rows: list[dict[str, object]] = []
    for row in source_rows:
        utterance_id = str(row["utterance_id"])
        derived = output_audio_path(output_dir, variant_key, utterance_id)
        new_row: dict[str, object] = dict(row)
        new_row["variant"] = variant_key
        new_row["variant_label"] = spec.label
        new_row["variant_pipeline"] = spec.pipeline
        new_row["expected_sample_rate"] = TARGET_SAMPLE_RATE
        new_row["expected_channels"] = 1
        new_row["expected_bit_depth"] = 16
        new_row["derived_audio_path"] = str(derived)
        if generated_audio or variant_key != "baseline":
            new_row["audio_path"] = str(derived)
        else:
            # Manifest-only baseline can point directly at the clean source WAV.
            new_row["audio_path"] = row["source_audio_path"]
        new_row["audio_exists"] = Path(str(new_row["audio_path"])).is_file()
        rows.append(new_row)
    return rows


def build_summary(rows: list[dict[str, object]], variant_keys: Iterable[str]) -> dict[str, object]:
    durations = [float(r["duration_seconds"]) for r in rows]
    summary: dict[str, object] = {
        "total_rows": len(rows),
        "variants": list(variant_keys),
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "mean": round(sum(durations) / len(durations), 6) if durations else None,
            "total_hours": round(sum(durations) / 3600, 6) if durations else None,
        },
        "source_sample_rates": dict(Counter(str(r["source_sample_rate"]) for r in rows)),
        "source_channels": dict(Counter(str(r["source_channels"]) for r in rows)),
        "source_bit_depths": dict(Counter(str(r["source_bit_depth"]) for r in rows)),
        "ethnicity": dict(Counter(str(r["ethnicity_raw"]) for r in rows)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FairSpeech compression manifests/audio")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", default="all", help="Comma-separated variant keys or 'all'")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for pilot/debug runs")
    parser.add_argument("--generate-audio", action="store_true", help="Actually create derived WAV files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing derived WAV files")
    parser.add_argument("--dry-run-commands", action="store_true", help="Print ffmpeg commands without running them")
    parser.add_argument("--expected-full-count", type=int, default=EXPECTED_FULL_COUNT)
    args = parser.parse_args()

    variant_keys = parse_variant_keys(args.variants)
    source_rows = build_source_manifest(args.metadata, args.audio_dir, args.max_samples)
    if args.max_samples is None and len(source_rows) != args.expected_full_count:
        raise SystemExit(
            f"Expected {args.expected_full_count:,} source rows, found {len(source_rows):,}. "
            "Use --max-samples for pilot/debug runs."
        )

    manifest_dir = args.output_dir / "manifests"
    summary_dir = args.output_dir / "summaries"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = manifest_dir / "fairspeech_source_manifest.csv"
    write_csv(source_manifest, source_rows)
    print(f"Wrote source manifest: {source_manifest} ({len(source_rows):,} rows)")

    generation_counts: dict[str, Counter[str]] = {}
    for variant_key in variant_keys:
        spec = VARIANTS[variant_key]
        counts: Counter[str] = Counter()
        if args.generate_audio:
            for row in source_rows:
                src = Path(str(row["source_audio_path"]))
                dst = output_audio_path(args.output_dir, variant_key, str(row["utterance_id"]))
                status = generate_variant_audio(src, dst, spec, overwrite=args.overwrite, dry_run=args.dry_run_commands)
                counts[status] += 1
            generation_counts[variant_key] = counts
            print(f"Generated {variant_key}: {dict(counts)}")

        rows = variant_manifest_rows(
            source_rows,
            args.output_dir,
            variant_key,
            generated_audio=args.generate_audio and not args.dry_run_commands,
        )
        variant_manifest = manifest_dir / f"fairspeech_{variant_key}_manifest.csv"
        write_csv(variant_manifest, rows)
        print(f"Wrote {variant_key} manifest: {variant_manifest} ({len(rows):,} rows)")

    summary = build_summary(source_rows, variant_keys)
    summary["generated_audio"] = args.generate_audio and not args.dry_run_commands
    summary["generation_counts"] = {k: dict(v) for k, v in generation_counts.items()}
    summary["variant_specs"] = {key: asdict(VARIANTS[key]) for key in variant_keys}
    summary_path = summary_dir / "fairspeech_compression_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
