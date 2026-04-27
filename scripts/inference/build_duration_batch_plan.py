#!/usr/bin/env python3
"""Build duration-bucketed batch plans for FairSpeech inference.

The plan uses padded-audio seconds as the first-order cost:
    batch_cost = num_samples * max_duration_in_batch

This creates variable-size batches: short clips get more samples per batch,
long clips get fewer. It does not profile a GPU; pilot-gate profiling should run
these plans on the target VM and then lock per-model budgets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_BUCKET_EDGES = [3, 5, 7, 10, 15, 20, 30, 45, 66]
DEFAULT_DURATION_COLUMNS = ["duration_seconds", "source_duration_seconds"]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for idx, row in enumerate(rows):
        row["_row_index"] = str(idx)
    return rows


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


def get_duration(row: dict[str, str], duration_col: str | None = None) -> float:
    candidates = [duration_col] if duration_col else DEFAULT_DURATION_COLUMNS
    for col in candidates:
        if col and row.get(col):
            return float(row[col])
    raise KeyError(f"No duration column found; tried {candidates}")


def parse_edges(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def bucket_label(duration: float, edges: list[float]) -> str:
    lower = 0.0
    for edge in edges:
        if duration <= edge:
            return f"{lower:g}-{edge:g}s"
        lower = edge
    return f">{edges[-1]:g}s"


def padded_seconds(durations: list[float]) -> float:
    if not durations:
        return 0.0
    return len(durations) * max(durations)


def build_batches(
    rows: list[dict[str, str]],
    max_padded_seconds: float,
    max_samples: int,
    bucket_edges: list[float] | None = None,
    duration_col: str | None = None,
) -> list[dict[str, object]]:
    """Return JSON-serializable variable-size batch records."""
    if max_padded_seconds <= 0:
        raise ValueError("max_padded_seconds must be > 0")
    if max_samples <= 0:
        raise ValueError("max_samples must be > 0")
    edges = bucket_edges or DEFAULT_BUCKET_EDGES

    grouped: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
    for row in rows:
        dur = get_duration(row, duration_col)
        grouped[bucket_label(dur, edges)].append((row, dur))

    batches: list[dict[str, object]] = []
    for label in sorted(grouped, key=lambda x: (float("inf") if x.startswith(">") else float(x.split("-")[0]))):
        items = sorted(grouped[label], key=lambda item: item[1])
        current: list[tuple[dict[str, str], float]] = []
        for row, dur in items:
            candidate_durations = [x[1] for x in current] + [dur]
            candidate_cost = padded_seconds(candidate_durations)
            over_budget = candidate_cost > max_padded_seconds and current
            over_count = len(current) >= max_samples
            if over_budget or over_count:
                batches.append(make_batch_record(current, label, len(batches)))
                current = []
            current.append((row, dur))
        if current:
            batches.append(make_batch_record(current, label, len(batches)))

    return batches


def make_batch_record(items: list[tuple[dict[str, str], float]], label: str, batch_idx: int) -> dict[str, object]:
    row_indices = [int(row["_row_index"]) for row, _ in items]
    utterance_ids = [row.get("utterance_id", row.get("hash_name", "")) for row, _ in items]
    durations = [dur for _, dur in items]
    return {
        "batch_id": f"b{batch_idx:06d}",
        "duration_bucket": label,
        "row_indices": row_indices,
        "utterance_ids": utterance_ids,
        "n_samples": len(items),
        "max_duration_seconds": round(max(durations), 6),
        "sum_duration_seconds": round(sum(durations), 6),
        "padded_audio_seconds": round(padded_seconds(durations), 6),
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def build_calibration_subset(
    rows: list[dict[str, str]],
    group_col: str = "ethnicity",
    duration_col: str | None = None,
    per_group: int = 4,
) -> list[dict[str, str]]:
    """Pick short, median, long, and max-duration examples per demographic group."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group = row.get(group_col, "") or "__missing__"
        grouped[group].append(row)

    selected: dict[str, dict[str, str]] = {}
    for _, group_rows in grouped.items():
        ordered = sorted(group_rows, key=lambda row: get_duration(row, duration_col))
        if not ordered:
            continue
        positions = [0, len(ordered) // 2, max(0, int(math.ceil(len(ordered) * 0.9)) - 1), len(ordered) - 1]
        for pos in positions[:per_group]:
            row = ordered[pos]
            uid = row.get("utterance_id", row.get("hash_name", row["_row_index"]))
            selected[uid] = row
    return sorted(selected.values(), key=lambda row: int(row["_row_index"]))


def profile_schema() -> dict[str, object]:
    return {
        "fields": {
            "model": "run_inference.py model key",
            "manifest": "manifest used for profiling",
            "batch_plan": "JSONL batch plan used for profiling",
            "max_padded_seconds": "candidate padded-audio-seconds budget",
            "max_samples": "candidate max sample cap",
            "num_batches_profiled": "number of calibration batches actually run",
            "peak_vram_mib": "maximum GPU memory observed during run",
            "throughput_audio_seconds_per_second": "processed audio seconds / wall time",
            "throughput_utterances_per_second": "utterances / wall time",
            "oom": "true if candidate failed due OOM",
            "failure": "error string for any non-OOM failure",
            "selected": "true only for locked safe policy",
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build duration-bucketed FairSpeech batch plans")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--calibration-csv", type=Path, default=None)
    parser.add_argument("--profile-schema-json", type=Path, default=None)
    parser.add_argument("--max-padded-seconds", type=float, default=160.0)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--duration-col", default=None)
    parser.add_argument("--bucket-edges", default=",".join(str(x) for x in DEFAULT_BUCKET_EDGES))
    parser.add_argument("--group-col", default="ethnicity")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    edges = parse_edges(args.bucket_edges)
    batches = build_batches(
        rows,
        max_padded_seconds=args.max_padded_seconds,
        max_samples=args.max_samples,
        bucket_edges=edges,
        duration_col=args.duration_col,
    )
    write_jsonl(args.output_jsonl, batches)

    if args.calibration_csv:
        calibration = build_calibration_subset(rows, group_col=args.group_col, duration_col=args.duration_col)
        write_csv(args.calibration_csv, calibration)
        print(f"Wrote calibration subset: {args.calibration_csv} ({len(calibration):,} rows)")

    if args.profile_schema_json:
        args.profile_schema_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_schema_json.write_text(json.dumps(profile_schema(), indent=2), encoding="utf-8")
        print(f"Wrote profile schema: {args.profile_schema_json}")

    summary = {
        "manifest": str(args.manifest),
        "output_jsonl": str(args.output_jsonl),
        "num_rows": len(rows),
        "num_batches": len(batches),
        "max_padded_seconds_budget": args.max_padded_seconds,
        "max_samples_cap": args.max_samples,
        "max_observed_padded_seconds": max((b["padded_audio_seconds"] for b in batches), default=0),
        "max_observed_samples": max((b["n_samples"] for b in batches), default=0),
        "median_batch_samples": statistics.median([int(b["n_samples"]) for b in batches]) if batches else 0,
        "duration_buckets": dict(sorted({b["duration_bucket"]: 0 for b in batches}.items())),
    }
    for batch in batches:
        summary["duration_buckets"][str(batch["duration_bucket"])] += 1

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {args.summary_json}")

    print(f"Wrote batch plan: {args.output_jsonl} ({len(batches):,} batches for {len(rows):,} rows)")


if __name__ == "__main__":
    main()
