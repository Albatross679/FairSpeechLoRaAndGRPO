#!/usr/bin/env python
"""VRAM/throughput tuning grid runner for Phase 3 full-scale LoRA training.

Runs a sequence of training cells as fresh subprocesses, measures each with
the Plan 01 gpu.log + timing.txt + training_config.json artifact triple,
picks a winner by explicit criteria, and freezes it as `vram_config.json`.

The grid is 6 cells by default — see GRID_CELLS below. Each cell runs as a
fresh `python train_standard_lora.py` subprocess to avoid CUDA state
carry-over (fragmentation, cached allocators, leaked tensors). Between cells
the tuner waits for VRAM to drain below 1 GiB before launching the next.

Usage:
    # Full grid:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python scripts/training/tune_vram.py \\
        --output_dir outputs/full-lora-vram-tune \\
        --max_steps 50 \\
        --gpu_poll_interval_s 1

    # Smoke-test: print commands without running:
    python scripts/training/tune_vram.py --dry_run

    # Reproducibility re-run (single cell):
    python scripts/training/tune_vram.py --only_cell D \\
        --output_dir outputs/full-lora-vram-tune-repro --max_steps 50
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

# Allow `from scripts.training...` imports when this script is invoked directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.training.validate_dryrun_gates import (  # noqa: E402
    parse_gpu_log, parse_timing, compute_gpu_util_stats, extract_step_times,
)


# -- Grid definition ---------------------------------------------------------

# Each cell's `flags` becomes CLI arguments to train_standard_lora.py, merged
# with SHARED_BASE_FLAGS. effective_batch is held constant at 16 across cells.
GRID_CELLS = [
    {
        "id": "A",
        "label": "baseline (fixed 4x4, GC on, adamw_torch)",
        "complexity_score": 0,
        "flags": {
            "batch_size": 4,
            "grad_accum": 4,
            "optim": "adamw_torch",
            # no --no_grad_checkpoint => GC on
            # no --dynamic_batch
        },
    },
    {
        "id": "B",
        "label": "fixed 4x4 + fused adamw (GC on)",
        "complexity_score": 0,
        "flags": {
            "batch_size": 4,
            "grad_accum": 4,
            "optim": "adamw_torch_fused",
        },
    },
    {
        "id": "C",
        "label": "fixed 8x2 + fused adamw (GC on)",
        "complexity_score": 0,
        "flags": {
            "batch_size": 8,
            "grad_accum": 2,
            "optim": "adamw_torch_fused",
        },
    },
    {
        "id": "D",
        "label": "dynamic fb=180 max=64 + fused (GC on)",
        "complexity_score": 1,
        "flags": {
            "batch_size": 1,
            "grad_accum": 4,
            "optim": "adamw_torch_fused",
            "dynamic_batch": True,
            "frame_budget": 180,
            "max_batch_size": 64,
        },
    },
    {
        "id": "E",
        "label": "dynamic fb=300 max=96 + fused (GC on)",
        "complexity_score": 1,
        "flags": {
            "batch_size": 1,
            "grad_accum": 4,
            "optim": "adamw_torch_fused",
            "dynamic_batch": True,
            "frame_budget": 300,
            "max_batch_size": 96,
        },
    },
    {
        "id": "F",
        "label": "dynamic fb=180 max=64 + fused + NO GC",
        "complexity_score": 2,
        "flags": {
            "batch_size": 1,
            "grad_accum": 4,
            "optim": "adamw_torch_fused",
            "dynamic_batch": True,
            "frame_budget": 180,
            "max_batch_size": 64,
            "no_grad_checkpoint": True,
        },
    },
]


# Flags applied to every cell. --full_dataset is intentionally NOT here:
# computing audio durations on the full 1.04M train set for the dynamic-batch
# cells would take hours (sf.info per file on overlayfs with 4 CPUs). A
# 5000-sample subset with SEED=42 is deterministic across cells, gives 50
# training steps worth of real data even at max_batch_size=96, and reduces
# per-cell startup from ~minutes to ~30s. Per-step compute/VRAM/util is
# identical at subset vs full-dataset scale.
SHARED_BASE_FLAGS = {
    "mode": "train",
    "fs_manifest": "outputs/manifests/fs_train.csv",
    "cv_manifest": "outputs/manifests/cv_train.csv",
    "locked_config_path": "outputs/standard-lora/locked_config.json",
    "lr": 9.8e-4,
    "dataloader_num_workers": 4,
    "dataloader_prefetch_factor": 2,
    "wandb_project": "none",
    "save_total_limit": 1,
    "subset_size": 5000,
}


# -- Helpers -----------------------------------------------------------------

def iso_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_cmd(cmd, **kwargs):
    """Run a command, return CompletedProcess. Does not raise on non-zero."""
    return subprocess.run(cmd, **kwargs)


def get_gpu_mem_used_mib():
    """Return current GPU0 memory.used in MiB, or None if nvidia-smi fails."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            text=True, timeout=10,
        ).strip()
        return int(out.splitlines()[0])
    except Exception as e:
        print(f"  WARN: nvidia-smi query failed: {e}")
        return None


def get_gpu_name_and_total():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits", "-i", "0"],
            text=True, timeout=10,
        ).strip()
        name, total = [p.strip() for p in out.split(",")]
        return name, int(total)
    except Exception:
        return "unknown", 0


def wait_for_vram_drain(threshold_mib=1024, max_wait_s=30):
    """Block until GPU0 memory.used drops below threshold, or timeout."""
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        used = get_gpu_mem_used_mib()
        if used is None:
            time.sleep(1)
            continue
        if used < threshold_mib:
            return used
        time.sleep(1)
    return get_gpu_mem_used_mib()


def build_cell_command(cell, shared, output_dir, max_steps):
    """Build the python train_standard_lora.py argv for a cell."""
    merged = dict(shared)
    merged.update(cell["flags"])
    merged["output_dir"] = output_dir
    merged["max_steps"] = max_steps

    cmd = [sys.executable, "scripts/training/train_standard_lora.py"]
    for k, v in merged.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
            # False -> omit (argparse store_true default)
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))
    return cmd


def poll_nvidia_smi_to_file(log_path, interval_s=1):
    """Start a background subprocess polling nvidia-smi every interval_s.

    Writes CSV: `epoch_ts,util_pct,mem_mib` one row per poll. Returns the
    Popen handle; caller must .terminate() and .wait() when done.
    """
    # Use a small bash loop so timestamps come from the shell clock, matching
    # Plan 01's format. We avoid `nvidia-smi -l` because some drivers have
    # issues with its internal timer, and we want per-second rows regardless.
    script = (
        f"while true; do "
        f"ts=$(date +%s); "
        f"row=$(nvidia-smi --query-gpu=utilization.gpu,memory.used "
        f"--format=csv,noheader,nounits -i 0 2>/dev/null); "
        f"echo \"$ts,$row\" >> {log_path}; "
        f"sleep {interval_s}; "
        f"done"
    )
    p = subprocess.Popen(
        ["bash", "-c", script],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return p


def stop_poller(poller):
    try:
        poller.terminate()
        poller.wait(timeout=5)
    except Exception:
        try:
            poller.kill()
        except Exception:
            pass


def compute_real_samples_per_step(cell, train_log, metrics):
    """Return the real samples per full training step for this cell.

    Fixed-batching cells: trivially `effective_batch_size` (from
    training_config.json), which is `batch_size * grad_accum`.

    Dynamic-batching cells: training_config.json.effective_batch_size is
    always `1 * grad_accum = 4` because the TrainingArguments value only
    reflects the trainer-level knob, not the FrameBudgetBatchSampler's
    actual output. Parse the trainer's stdout for the printed
    "{N} batches, avg {Y} samples/batch" line emitted by the dynamic-
    batch setup block in train_standard_lora.py:run_train() and multiply
    Y by grad_accum.

    Returns None if no samples/step can be inferred (unknown config).
    """
    flags = cell.get("flags", {}) or {}
    if not flags.get("dynamic_batch"):
        eb = metrics.get("effective_batch_size")
        return int(eb) if eb else None

    grad_accum = int(flags.get("grad_accum") or 1)
    try:
        with open(train_log, "r", errors="replace") as f:
            text = f.read()
    except OSError:
        return None

    import re
    m = re.search(r"(\d+)\s+batches,\s*avg\s+([\d.]+)\s+samples/batch", text)
    if not m:
        return None
    avg_samples_per_batch = float(m.group(2))
    return int(round(avg_samples_per_batch * grad_accum))


# -- Per-cell execution ------------------------------------------------------

def run_cell(cell, args, shared_flags):
    """Run a single cell, return its metrics.json dict."""
    cell_dir = os.path.join(args.output_dir, f"cell-{cell['id']}")
    os.makedirs(cell_dir, exist_ok=True)

    gpu_log = os.path.join(cell_dir, "gpu.log")
    timing_path = os.path.join(cell_dir, "timing.txt")
    train_log = os.path.join(cell_dir, "train.log")
    metrics_path = os.path.join(cell_dir, "metrics.json")

    # Pre-flight VRAM drain
    residual = wait_for_vram_drain(threshold_mib=1024, max_wait_s=30)
    if residual is not None and residual > 1024:
        print(f"  WARN: cell {cell['id']} launching with {residual} MiB residual VRAM")

    cmd = build_cell_command(cell, shared_flags, cell_dir, args.max_steps)
    print(f"\n{'='*70}\ncell-{cell['id']}: {cell['label']}\n{'='*70}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  cell_dir: {cell_dir}")

    # Reset gpu.log
    open(gpu_log, "w").close()

    # Start GPU poller
    poller = poll_nvidia_smi_to_file(gpu_log, interval_s=args.gpu_poll_interval_s)

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Pin each cell to GPU0 to avoid accidental multi-GPU scheduling
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    train_start = int(time.time())
    with open(train_log, "w") as lf:
        proc = subprocess.run(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
        )
    train_end = int(time.time())
    wallclock = train_end - train_start

    # Stop poller
    stop_poller(poller)

    # Write timing.txt (same format as Plan 01)
    with open(timing_path, "w") as f:
        f.write(f"TRAIN_START_EPOCH={train_start}\n")
        f.write(f"TRAIN_END_EPOCH={train_end}\n")
        f.write(f"WALLCLOCK_SEC={wallclock}\n")
        f.write(f"EXIT_CODE={proc.returncode}\n")

    print(f"  wallclock={wallclock}s exit={proc.returncode}")

    # Assemble metrics
    metrics = {
        "cell_id": cell["id"],
        "label": cell["label"],
        "flags": cell["flags"],
        "cmd": cmd,
        "wallclock_sec": wallclock,
        "exit_code": proc.returncode,
        "complexity_score": cell["complexity_score"],
    }

    # Parse training_config.json if it exists
    tc_path = os.path.join(cell_dir, "training_config.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
        metrics["peak_vram_gb"] = float(tc.get("peak_vram_gb", 0.0))
        metrics["final_loss"] = float(tc.get("final_train_loss", float("nan")))
        metrics["gradient_checkpointing_enabled"] = tc.get("gradient_checkpointing_enabled")
        metrics["optim_runtime"] = tc.get("optim")
        metrics["effective_batch_size"] = tc.get("effective_batch_size")
        metrics["effective_learning_rate"] = tc.get("effective_learning_rate")
        metrics["vram_config_source"] = tc.get("vram_config_source")
        metrics["steps_per_epoch"] = tc.get("steps_per_epoch")
    else:
        metrics["peak_vram_gb"] = None
        metrics["final_loss"] = None
        metrics["gradient_checkpointing_enabled"] = None
        metrics["optim_runtime"] = None

    # GPU-Util stats from gpu.log
    rows = parse_gpu_log(gpu_log)
    # Use a shorter steady_offset than Plan 01 (15s) because 50-step runs have
    # a shorter total wall-clock and we'd otherwise lose half the samples.
    stats = compute_gpu_util_stats(rows, train_start, train_end, steady_offset=15)
    if stats and not stats.get("insufficient"):
        metrics["mean_gpu_util"] = stats["mean_gpu_util"]
        metrics["median_gpu_util"] = stats["median_gpu_util"]
        metrics["p10_gpu_util"] = stats["p10"]
        metrics["mem_peak_mib"] = stats["mem_peak_mib"]
        metrics["n_steady_samples"] = stats["n_steady"]
    else:
        metrics["mean_gpu_util"] = None
        metrics["median_gpu_util"] = None
        metrics["p10_gpu_util"] = None
        metrics["mem_peak_mib"] = None
        metrics["n_steady_samples"] = stats["n_steady"] if stats else 0

    # Per-step times from trainer stdout
    step_times = extract_step_times(train_log)
    if step_times:
        # skip first 5 steps (warmup)
        warm = step_times[5:] if len(step_times) > 10 else step_times
        warm_sorted = sorted(warm)
        median_step_time_s = warm_sorted[len(warm_sorted) // 2]
        metrics["median_step_time_s"] = float(median_step_time_s)
        # Real samples/sec is the comparable throughput metric across cells.
        # Fixed-batch: effective_batch_size (from training_config.json) / step_time
        # Dynamic-batch: dataloader_batches has avg-samples-per-batch that gets
        #   multiplied by grad_accum — parse it from the trainer log since the
        #   training_config.json effective_batch_size is 1*grad_accum=4 (wrong
        #   for dynamic, it only captures the trainer-level knob, not the real
        #   FrameBudgetBatchSampler output).
        samples_per_step = compute_real_samples_per_step(cell, train_log, metrics)
        metrics["real_samples_per_step"] = samples_per_step
        metrics["samples_per_sec"] = (
            float(samples_per_step) / float(median_step_time_s)
            if samples_per_step else None
        )
        # Keep tokens_per_sec for backwards-compat, but now it equals
        # samples_per_sec so the sort ordering is consistent.
        metrics["tokens_per_sec"] = metrics["samples_per_sec"]
    else:
        metrics["median_step_time_s"] = None
        metrics["tokens_per_sec"] = None
        metrics["samples_per_sec"] = None
        metrics["real_samples_per_step"] = None

    # Verdict
    verdict = "pass"
    reason = []
    if proc.returncode != 0:
        verdict = "fail_exit"
        reason.append(f"exit_code={proc.returncode}")
    elif metrics["peak_vram_gb"] is None:
        verdict = "fail_no_config"
        reason.append("training_config.json missing")
    elif metrics["peak_vram_gb"] >= 44.0:
        verdict = "fail_vram"
        reason.append(f"peak_vram_gb={metrics['peak_vram_gb']:.2f} >= 44")
    elif metrics["final_loss"] is None or not (metrics["final_loss"] < 3.0):
        verdict = "fail_loss"
        reason.append(f"final_loss={metrics['final_loss']}")
    elif metrics["median_step_time_s"] is None:
        verdict = "fail_step_time_missing"
        reason.append("no step-time samples extracted from train.log")
    elif metrics["median_step_time_s"] >= 30.0:
        verdict = "fail_step_time"
        reason.append(f"median_step_time_s={metrics['median_step_time_s']:.2f} >= 30")

    metrics["verdict"] = verdict
    metrics["verdict_reason"] = "; ".join(reason) if reason else "ok"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"  verdict: {verdict}  peak_vram_gb={metrics.get('peak_vram_gb')}  "
          f"mean_gpu_util={metrics.get('mean_gpu_util')}  "
          f"median_step_time_s={metrics.get('median_step_time_s')}  "
          f"final_loss={metrics.get('final_loss')}")

    return metrics


# -- Reporting & winner selection --------------------------------------------

def write_grid_csv(metrics_list, path):
    cols = [
        "cell_id", "label", "verdict", "peak_vram_gb", "mean_gpu_util",
        "median_step_time_s", "real_samples_per_step", "samples_per_sec",
        "tokens_per_sec", "final_loss",
        "gradient_checkpointing_enabled", "optim_runtime", "effective_batch_size",
        "n_steady_samples", "wallclock_sec", "complexity_score",
        "flags_json", "verdict_reason",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in metrics_list:
            w.writerow([
                m.get("cell_id"),
                m.get("label"),
                m.get("verdict"),
                f"{m['peak_vram_gb']:.3f}" if isinstance(m.get("peak_vram_gb"), (int, float)) else "",
                f"{m['mean_gpu_util']:.2f}" if isinstance(m.get("mean_gpu_util"), (int, float)) else "",
                f"{m['median_step_time_s']:.3f}" if isinstance(m.get("median_step_time_s"), (int, float)) else "",
                m.get("real_samples_per_step") if m.get("real_samples_per_step") is not None else "",
                f"{m['samples_per_sec']:.3f}" if isinstance(m.get("samples_per_sec"), (int, float)) else "",
                f"{m['tokens_per_sec']:.3f}" if isinstance(m.get("tokens_per_sec"), (int, float)) else "",
                f"{m['final_loss']:.4f}" if isinstance(m.get("final_loss"), (int, float)) else "",
                m.get("gradient_checkpointing_enabled"),
                m.get("optim_runtime"),
                m.get("effective_batch_size"),
                m.get("n_steady_samples"),
                m.get("wallclock_sec"),
                m.get("complexity_score"),
                json.dumps(m.get("flags"), sort_keys=True),
                m.get("verdict_reason"),
            ])


def fmt_num(x, ndigits=2, default="-"):
    if x is None:
        return default
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return default


def write_grid_markdown(metrics_list, winner_id, winner_reason, path):
    lines = [
        "# Plan 03-02 VRAM Tuning Grid — Results",
        "",
        f"Generated: {iso_now()}",
        "",
        "## Grid",
        "",
        "| Cell | Label | Verdict | VRAM (GB) | Util % | step_s | tok/s | final_loss | GC | optim |",
        "|------|-------|---------|-----------|--------|--------|-------|------------|----|-------|",
    ]
    for m in metrics_list:
        marker = " **WIN**" if m.get("cell_id") == winner_id else ""
        lines.append(
            f"| {m.get('cell_id')}{marker} | {m.get('label')} | {m.get('verdict')} | "
            f"{fmt_num(m.get('peak_vram_gb'), 2)} | "
            f"{fmt_num(m.get('mean_gpu_util'), 1)} | "
            f"{fmt_num(m.get('median_step_time_s'), 2)} | "
            f"{fmt_num(m.get('tokens_per_sec'), 2)} | "
            f"{fmt_num(m.get('final_loss'), 4)} | "
            f"{m.get('gradient_checkpointing_enabled')} | "
            f"{m.get('optim_runtime')} |"
        )
    lines += [
        "",
        "## Winner",
        "",
        f"- **cell-{winner_id}** — {winner_reason}",
        "",
        "## Selection rule",
        "",
        "Sort passing cells by: `-mean_gpu_util, -tokens_per_sec, +median_step_time_s, +complexity_score`.",
        "Head of the sorted list is the winner. Ties within 2 percentage points on util prefer the simpler config.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def select_winner(metrics_list):
    passing = [m for m in metrics_list if m.get("verdict") == "pass"]
    if not passing:
        return None, "no cells passed"

    def sort_key(m):
        return (
            -(m.get("mean_gpu_util") or 0.0),
            -(m.get("tokens_per_sec") or 0.0),
            (m.get("median_step_time_s") or float("inf")),
            (m.get("complexity_score") or 0),
        )

    passing_sorted = sorted(passing, key=sort_key)
    winner = passing_sorted[0]

    # Occam tiebreaker: among ALL passing cells within 2 percentage points of
    # the utilization leader, prefer the one with the lowest complexity_score
    # (fixed batching > dynamic; GC on > GC off). Tiebreak within the simpler
    # band by the original sort key (util, tok/s, step_time). This matches the
    # plan's explicit rule — the previous implementation only compared the top
    # two cells and could skip a simpler cell that ranked 3rd or later.
    lead_u = winner.get("mean_gpu_util") or 0.0
    tie_band = [m for m in passing_sorted
                if (lead_u - (m.get("mean_gpu_util") or 0.0)) <= 2.0]
    if tie_band:
        min_complexity = min((m.get("complexity_score") or 0) for m in tie_band)
        simpler = [m for m in tie_band
                   if (m.get("complexity_score") or 0) == min_complexity]
        # Preserve the original sort order within the simpler group
        winner = simpler[0]

    reason = (
        f"mean_gpu_util={winner.get('mean_gpu_util'):.1f}%, "
        f"peak_vram_gb={winner.get('peak_vram_gb'):.2f}, "
        f"median_step_time_s={winner.get('median_step_time_s'):.2f}s, "
        f"tokens_per_sec={winner.get('tokens_per_sec'):.2f}, "
        f"complexity={winner.get('complexity_score')}"
    )
    return winner, reason


def compute_verdict(winner):
    if winner is None:
        return "blocked", "no cells passed — review grid_results.md"
    util = winner.get("mean_gpu_util") or 0.0
    if util >= 70.0:
        return "go", (
            f"mean_gpu_util {util:.1f}% >= 70% — compute-bound, healthy; "
            f"proceed to Plan 03-03"
        )
    if util >= 55.0:
        return "go_with_warning", (
            f"mean_gpu_util {util:.1f}% in [55, 70) — I/O is the ceiling on "
            f"this host (4-core cgroup + overlayfs); Plan 03-03 can proceed "
            f"but wall-clock will be dataloader-bound"
        )
    return "blocked", (
        f"mean_gpu_util {util:.1f}% < 55% — I/O remediation required before "
        f"Plan 03-03 launch; see outputs/full-lora-vram-tune/io_remediation_report.md"
    )


def write_vram_config(winner, verdict, verdict_reason, shared_flags, args, path):
    gpu_name, gpu_total = get_gpu_name_and_total()
    try:
        nproc = int(subprocess.check_output(["nproc"], text=True).strip())
    except Exception:
        nproc = None

    # Merge shared base flags with winner-specific flags for the "flags"
    # section Plan 03-03 will consume via --vram_config.
    flags = dict(shared_flags)
    flags.update(winner["flags"])
    # Remove keys that should NOT auto-propagate into a Plan 03-03 launch
    # (subset_size -> Plan 03-03 uses --full_dataset; mode/manifests are
    # launch-time concerns, not tuning concerns).
    for k in ("subset_size", "mode", "fs_manifest", "cv_manifest"):
        flags.pop(k, None)

    # grid_spec_sha256: hash of the GRID_CELLS + SHARED_BASE_FLAGS json
    spec_payload = json.dumps(
        {"cells": GRID_CELLS, "shared": SHARED_BASE_FLAGS},
        sort_keys=True,
    ).encode()
    grid_sha = hashlib.sha256(spec_payload).hexdigest()

    config = {
        "source": f"phase-03 VRAM tuning, cell {winner['cell_id']}",
        "target_host": f"{gpu_name} ({gpu_total} MiB)",
        "target_host_cpu_cores": nproc,
        "target_host_storage": "overlayfs",
        "tuned_at": iso_now(),
        "grid_spec_sha256": grid_sha,
        "flags": flags,
        "measured": {
            "peak_vram_gb": winner.get("peak_vram_gb"),
            "mean_gpu_util": winner.get("mean_gpu_util"),
            "median_step_time_s": winner.get("median_step_time_s"),
            "tokens_per_sec": winner.get("tokens_per_sec"),
            "final_loss": winner.get("final_loss"),
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return config


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/full-lora-vram-tune")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--gpu_poll_interval_s", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print per-cell commands without running")
    parser.add_argument("--only_cell", type=str, default=None,
                        help="Run only the named cell (A..F). Used for reproducibility re-runs.")
    args = parser.parse_args()

    cells = GRID_CELLS
    if args.only_cell:
        cells = [c for c in GRID_CELLS if c["id"] == args.only_cell]
        if not cells:
            print(f"ERROR: cell '{args.only_cell}' not in grid (ids: "
                  f"{[c['id'] for c in GRID_CELLS]})")
            sys.exit(2)

    if args.dry_run:
        # Print one line per cell, prefixed with `cell-{id}` so the verify
        # command `grep -c ^cell-` can count them.
        for cell in cells:
            cmd = build_cell_command(
                cell, SHARED_BASE_FLAGS,
                os.path.join(args.output_dir, f"cell-{cell['id']}"),
                args.max_steps,
            )
            print(f"cell-{cell['id']}: {' '.join(cmd)}")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)

    # Pre-flight: GPU idle check
    used = get_gpu_mem_used_mib()
    if used is None or used > 1024:
        print(f"ERROR: GPU not idle (memory.used={used} MiB). "
              f"Refusing to start the grid. Check `nvidia-smi`.")
        sys.exit(3)
    print(f"[preflight] GPU memory used: {used} MiB (OK)")

    all_metrics = []
    for cell in cells:
        t0 = time.time()
        m = run_cell(cell, args, SHARED_BASE_FLAGS)
        t1 = time.time()
        print(f"  cell-{cell['id']} total tune time: {t1-t0:.1f}s")
        all_metrics.append(m)

        # Drain residual VRAM before next cell
        drained = wait_for_vram_drain(threshold_mib=1024, max_wait_s=30)
        print(f"  post-cell residual VRAM: {drained} MiB")
        if drained is not None and drained > 1024:
            print(f"  WARN: VRAM did not drain below 1 GiB after cell {cell['id']}; continuing anyway")

    # Reporting
    grid_csv = os.path.join(args.output_dir, "grid_results.csv")
    grid_md = os.path.join(args.output_dir, "grid_results.md")
    vram_config_path = os.path.join(args.output_dir, "vram_config.json")

    if args.only_cell:
        # In reproducibility mode, don't overwrite the main grid artifacts.
        repro_metrics_path = os.path.join(args.output_dir, "repro_metrics.json")
        with open(repro_metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"\n[repro] wrote {repro_metrics_path}")
        return 0

    winner, winner_reason = select_winner(all_metrics)
    if winner is None:
        print(f"\nERROR: {winner_reason}")
        write_grid_csv(all_metrics, grid_csv)
        write_grid_markdown(all_metrics, winner_id="-", winner_reason=winner_reason, path=grid_md)
        sys.exit(4)

    verdict, verdict_reason = compute_verdict(winner)
    write_grid_csv(all_metrics, grid_csv)
    write_grid_markdown(all_metrics, winner_id=winner["cell_id"],
                        winner_reason=winner_reason, path=grid_md)
    config = write_vram_config(
        winner, verdict, verdict_reason, SHARED_BASE_FLAGS, args, vram_config_path,
    )

    print(f"\n{'='*70}\nWINNER: cell-{winner['cell_id']} ({winner['label']})\n{'='*70}")
    print(f"  {winner_reason}")
    print(f"  verdict: {verdict}")
    print(f"  verdict_reason: {verdict_reason}")
    print(f"\n  Artifacts:")
    print(f"    {grid_csv}")
    print(f"    {grid_md}")
    print(f"    {vram_config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
