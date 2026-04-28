#!/usr/bin/env python3
"""Validate FairSpeech compression manifests and derived WAV folders.

This is a fast, CPU-only gate. It checks row identity, required columns, audio
existence, final WAV format, duration sanity, and optional silence detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import wave
from collections import Counter
from pathlib import Path

REQUIRED_COLUMNS = {
    "utterance_id",
    "audio_path",
    "source_audio_path",
    "sentence",
    "ethnicity",
    "ethnicity_raw",
    "variant",
    "duration_seconds",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def wav_metadata(path: Path) -> dict[str, int | float | str]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        frames = wf.getnframes()
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "bit_depth": sampwidth * 8,
        "frames": frames,
        "duration_seconds": frames / sample_rate if sample_rate else 0.0,
    }


def is_probably_silent(path: Path, max_frames: int | None = None) -> bool:
    """Return True when all sampled PCM frames are zero.

    When ``max_frames`` is set, sample beginning/middle/end chunks instead of
    reading the whole file. That keeps full-corpus validation practical while
    still catching fully silent generated payloads.
    """
    with wave.open(str(path), "rb") as wf:
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()
        if max_frames is None or total_frames <= max_frames:
            data = wf.readframes(total_frames)
        else:
            chunk_frames = max(1, max_frames // 3)
            starts = [
                0,
                max(0, total_frames // 2 - chunk_frames // 2),
                max(0, total_frames - chunk_frames),
            ]
            chunks = []
            for start in starts:
                wf.setpos(start)
                chunks.append(wf.readframes(min(chunk_frames, total_frames - start)))
            data = b"".join(chunks)
    if not data:
        return True
    if sampwidth == 2:
        count = len(data) // 2
        samples = struct.unpack("<" + "h" * count, data[: count * 2])
        return max(abs(x) for x in samples) == 0
    # Fallback for unusual sample widths: byte-level all-zero check.
    return all(b == 0 for b in data)


def validate_manifest(
    path: Path,
    expected_count: int | None,
    allow_missing_audio: bool,
    check_silence: bool,
    silence_sample_frames: int | None,
    max_audio_checks: int | None,
) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    rows = read_csv(path)

    columns = set(rows[0].keys()) if rows else set()
    missing_cols = sorted(REQUIRED_COLUMNS - columns)
    if missing_cols:
        errors.append(f"missing required columns: {missing_cols}")

    if expected_count is not None and len(rows) != expected_count:
        errors.append(f"expected {expected_count:,} rows, found {len(rows):,}")

    ids = [row.get("utterance_id", "") for row in rows]
    duplicate_ids = [uid for uid, count in Counter(ids).items() if uid and count > 1]
    if duplicate_ids:
        errors.append(f"duplicate utterance_id values: {duplicate_ids[:5]}")

    audio_missing = 0
    wrong_format = 0
    duration_mismatch = 0
    silent = 0
    checked = 0
    variants = Counter(row.get("variant", "") for row in rows)
    ethnicities = Counter(row.get("ethnicity_raw", "") for row in rows)

    for row in rows:
        if max_audio_checks is not None and checked >= max_audio_checks:
            break
        audio_path = Path(row.get("audio_path", ""))
        if not audio_path.is_file():
            audio_missing += 1
            continue
        checked += 1
        try:
            meta = wav_metadata(audio_path)
        except Exception as exc:  # noqa: BLE001 - validation should report all failures
            wrong_format += 1
            warnings.append(f"{audio_path}: could not read WAV header: {exc}")
            continue

        if meta["sample_rate"] != 16_000 or meta["channels"] != 1 or meta["bit_depth"] != 16:
            wrong_format += 1
            warnings.append(f"{audio_path}: expected 16 kHz mono PCM16, got {meta}")

        expected_dur = row.get("duration_seconds") or row.get("source_duration_seconds")
        if expected_dur:
            try:
                if abs(float(expected_dur) - float(meta["duration_seconds"])) > 0.25:
                    duration_mismatch += 1
            except ValueError:
                warnings.append(f"{audio_path}: invalid duration value {expected_dur!r}")

        if check_silence and is_probably_silent(audio_path, max_frames=silence_sample_frames):
            silent += 1
            warnings.append(f"{audio_path}: appears silent")

    if audio_missing and not allow_missing_audio:
        errors.append(f"missing audio files: {audio_missing:,}")
    if wrong_format:
        errors.append(f"wrong or unreadable WAV format files: {wrong_format:,}")
    if duration_mismatch:
        errors.append(f"duration mismatches >0.25s: {duration_mismatch:,}")
    if silent:
        errors.append(f"silent files: {silent:,}")

    summary = {
        "manifest": str(path),
        "rows": len(rows),
        "checked_audio_files": checked,
        "missing_audio_files": audio_missing,
        "wrong_format_files": wrong_format,
        "duration_mismatches": duration_mismatch,
        "silent_files": silent,
        "variants": dict(variants),
        "ethnicity_raw_counts": dict(ethnicities),
        "required_columns_present": not missing_cols,
        "allow_missing_audio": allow_missing_audio,
        "warnings": warnings[:50],
        "status": "pass" if not errors else "fail",
    }
    return summary, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FairSpeech compression manifests/audio")
    parser.add_argument("manifests", nargs="+", type=Path, help="Manifest CSV paths")
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--allow-missing-audio", action="store_true")
    parser.add_argument("--check-silence", action="store_true")
    parser.add_argument(
        "--silence-sample-frames",
        type=int,
        default=None,
        help="When checking silence, sample this many frames across each file instead of reading full audio.",
    )
    parser.add_argument("--max-audio-checks", type=int, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    all_summaries = []
    all_errors: list[str] = []
    for manifest in args.manifests:
        summary, errors = validate_manifest(
            manifest,
            expected_count=args.expected_count,
            allow_missing_audio=args.allow_missing_audio,
            check_silence=args.check_silence,
            silence_sample_frames=args.silence_sample_frames,
            max_audio_checks=args.max_audio_checks,
        )
        all_summaries.append(summary)
        for error in errors:
            all_errors.append(f"{manifest}: {error}")
        print(f"{summary['status'].upper()}: {manifest} ({summary['rows']:,} rows)", flush=True)

    combined = {
        "status": "pass" if not all_errors else "fail",
        "manifests": all_summaries,
        "errors": all_errors,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"Wrote summary: {args.summary_json}")

    if all_errors:
        for error in all_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
