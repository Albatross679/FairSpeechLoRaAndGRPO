#!/usr/bin/env python3
"""Pilot-gate profiler for duration-bucketed inference batch policies.

This script is intended to run on the target GPU VM after infrastructure is
built. It tries candidate padded-audio-second budgets and max-sample caps,
invokes run_inference.py on a calibration manifest, records peak VRAM and
throughput, and writes a profile manifest. It does not choose final scientific
results; it only locks safe runtime policy before full inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_csv_numbers(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def read_manifest_count_and_seconds(path: Path) -> tuple[int, float]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    total_seconds = 0.0
    for row in rows:
        value = row.get("duration_seconds") or row.get("source_duration_seconds") or "0"
        total_seconds += float(value or 0)
    return len(rows), total_seconds


def query_gpu_memory() -> tuple[int | None, int | None]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None, None
    cmd = [
        nvidia_smi,
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip().splitlines()[0]
        used, total = [int(x.strip()) for x in out.split(",")]
        return used, total
    except Exception:
        return None, None


def assert_gpu_safe(threshold: float) -> None:
    used, total = query_gpu_memory()
    if used is None or total in (None, 0):
        print("WARNING: nvidia-smi unavailable; cannot pre-check GPU memory", file=sys.stderr)
        return
    frac = used / total
    if frac > threshold:
        raise SystemExit(
            f"GPU memory already {used}/{total} MiB ({frac:.1%}), above safety threshold {threshold:.0%}."
        )


def monitor_peak_vram(stop: threading.Event, interval: float, sink: dict[str, int | None]) -> None:
    peak = None
    while not stop.is_set():
        used, _ = query_gpu_memory()
        if used is not None:
            peak = used if peak is None else max(peak, used)
        time.sleep(interval)
    used, _ = query_gpu_memory()
    if used is not None:
        peak = used if peak is None else max(peak, used)
    sink["peak_vram_mib"] = peak


def build_plan(manifest: Path, output_jsonl: Path, max_padded_seconds: float, max_samples: int) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "inference" / "build_duration_batch_plan.py"),
        "--manifest", str(manifest),
        "--output-jsonl", str(output_jsonl),
        "--max-padded-seconds", str(max_padded_seconds),
        "--max-samples", str(max_samples),
    ]
    subprocess.run(cmd, check=True)


def run_candidate(args, max_padded_seconds: float, max_samples: int, work_dir: Path) -> dict[str, object]:
    plan_path = work_dir / f"plan_s{max_padded_seconds:g}_n{max_samples}.jsonl"
    build_plan(args.manifest, plan_path, max_padded_seconds, max_samples)

    output_dir = work_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    variant = f"profile_s{max_padded_seconds:g}_n{max_samples}"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "inference" / "run_inference.py"),
        "--model", args.model,
        "--manifest", str(args.manifest),
        "--output_dir", str(output_dir),
        "--device", args.device,
        "--batch_size", "1",
        "--batch_plan", str(plan_path),
        "--audio_variant", variant,
        "--dataset", "fs",
    ]

    if "cuda" in args.device:
        assert_gpu_safe(args.gpu_safety_threshold)

    n_rows, audio_seconds = read_manifest_count_and_seconds(args.manifest)
    stop = threading.Event()
    monitor_sink: dict[str, int | None] = {"peak_vram_mib": None}
    monitor = threading.Thread(target=monitor_peak_vram, args=(stop, args.monitor_interval, monitor_sink))

    start = time.time()
    monitor.start()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stop.set()
    monitor.join(timeout=5)
    elapsed = time.time() - start

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    oom = "out of memory" in combined.lower() or "cuda oom" in combined.lower()
    failure = None if proc.returncode == 0 else combined[-4000:]
    return {
        "model": args.model,
        "manifest": str(args.manifest),
        "batch_plan": str(plan_path),
        "max_padded_seconds": max_padded_seconds,
        "max_samples": max_samples,
        "num_rows": n_rows,
        "audio_seconds": round(audio_seconds, 6),
        "elapsed_seconds": round(elapsed, 6),
        "peak_vram_mib": monitor_sink.get("peak_vram_mib"),
        "throughput_audio_seconds_per_second": round(audio_seconds / elapsed, 6) if elapsed > 0 else None,
        "throughput_utterances_per_second": round(n_rows / elapsed, 6) if elapsed > 0 else None,
        "returncode": proc.returncode,
        "oom": oom,
        "failure": failure,
        "selected": False,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile duration-bucketed inference batch policy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Calibration manifest CSV")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--candidate-seconds", default="40,80,120,160")
    parser.add_argument("--candidate-max-samples", default="4,8,16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--monitor-interval", type=float, default=0.5)
    parser.add_argument("--gpu-safety-threshold", type=float, default=0.80)
    parser.add_argument("--stop-after-first-failure", action="store_true")
    args = parser.parse_args()

    seconds = parse_csv_numbers(args.candidate_seconds, float)
    max_samples_values = parse_csv_numbers(args.candidate_max_samples, int)
    profile_rows: list[dict[str, object]] = []

    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        ctx = tempfile.TemporaryDirectory(dir=str(args.work_dir), prefix="batch-profile-")
    else:
        ctx = tempfile.TemporaryDirectory(prefix="batch-profile-")

    with ctx as tmp:
        work_dir = Path(tmp)
        for sec in seconds:
            for max_samples in max_samples_values:
                print(f"Profiling {args.model}: max_padded_seconds={sec}, max_samples={max_samples}")
                row = run_candidate(args, sec, max_samples, work_dir)
                profile_rows.append(row)
                write_jsonl(args.output_jsonl, profile_rows)
                if row["returncode"] != 0 and args.stop_after_first_failure:
                    print("Stopping after first failed candidate")
                    return

    # Select largest non-OOM successful candidate by padded seconds then max_samples.
    successful = [r for r in profile_rows if r["returncode"] == 0 and not r["oom"]]
    if successful:
        selected = sorted(successful, key=lambda r: (float(r["max_padded_seconds"]), int(r["max_samples"]))) [-1]
        selected["selected"] = True
        write_jsonl(args.output_jsonl, profile_rows)
        print(f"Selected safe candidate: seconds={selected['max_padded_seconds']}, max_samples={selected['max_samples']}")
    print(f"Wrote profile manifest: {args.output_jsonl}")


if __name__ == "__main__":
    main()
