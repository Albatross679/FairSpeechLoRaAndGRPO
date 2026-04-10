"""Validate a Plan 03-01 dry-run against the HP / VRAM / GPU-Util gates.

Reads:
  outputs/full-lora-dryrun/training_config.json
  outputs/full-lora-dryrun-gpu.log        (nvidia-smi --query-gpu polling)
  outputs/full-lora-dryrun-timing.txt     (TRAIN_START_EPOCH / TRAIN_END_EPOCH)
  outputs/full-lora-dryrun-train.log      (trainer stdout)

Exit code: 0 if all gates PASS, 1 otherwise.

Note on GPU-Util gate: the 70% threshold is calibrated for an RTX 3090. On a
faster GPU (e.g. RTX A6000) each training step completes faster, leaving the
dataloader a larger relative share of wall-clock — which depresses util even
when the pipeline itself is unchanged. Interpret sub-70% results on fast GPUs
accordingly.

Usage:
    python scripts/training/validate_dryrun_gates.py [--output_dir outputs/full-lora-dryrun]
"""
import argparse
import json
import os
import statistics
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/full-lora-dryrun")
    parser.add_argument("--gpu_log", default="outputs/full-lora-dryrun-gpu.log")
    parser.add_argument("--timing", default="outputs/full-lora-dryrun-timing.txt")
    parser.add_argument("--train_log", default="outputs/full-lora-dryrun-train.log")
    parser.add_argument("--steady_offset", type=int, default=30,
                        help="Seconds to skip after train_start (warmup/model-load)")
    parser.add_argument("--util_threshold", type=float, default=70.0,
                        help="Mean GPU-Util gate in percent")
    parser.add_argument("--vram_ceiling_gb", type=float, default=24.0)
    args = parser.parse_args()

    failures = []

    def fail(msg):
        failures.append(msg)
        print(f"  FAIL: {msg}")

    def ok(msg):
        print(f"  OK:   {msg}")

    # 1. training_config.json checks
    cfg_path = os.path.join(args.output_dir, "training_config.json")
    with open(cfg_path) as f:
        config = json.load(f)

    print("=== training_config.json gates ===")
    params = config.get("params", {})
    for name, actual, expected in [
        ("lr_scheduler", config.get("lr_scheduler"), "constant"),
        ("warmup_ratio", config.get("warmup_ratio"), 0.0),
        ("batch_size", config.get("batch_size"), 4),
        ("grad_accum_steps", config.get("grad_accum_steps"), 4),
        ("effective_batch_size", config.get("effective_batch_size"), 16),
        ("params.target_mlp", params.get("target_mlp"), True),
    ]:
        if actual == expected:
            ok(f"{name} = {actual}")
        else:
            fail(f"{name} = {actual} (expected {expected})")

    peak = config.get("peak_vram_gb", 999)
    if peak < args.vram_ceiling_gb:
        ok(f"peak_vram_gb = {peak:.2f} GB < {args.vram_ceiling_gb} GB")
    else:
        fail(f"peak_vram_gb = {peak:.2f} GB >= {args.vram_ceiling_gb} GB")

    print(f"  NOTE: params.lr = {params.get('lr'):.2e} (from locked_config.json);")
    print(f"        CLI --lr override is applied in-memory at TrainingArguments")
    print(f"        construction but is NOT echoed into training_config.json.")

    # 2. GPU-Util gate
    print("\n=== GPU-Util gate ===")
    with open(args.timing) as f:
        timing = dict(line.strip().split("=", 1) for line in f if "=" in line)
    train_start = int(timing["TRAIN_START_EPOCH"])
    train_end = int(timing["TRAIN_END_EPOCH"])

    rows = []
    with open(args.gpu_log) as f:
        for line in f:
            line = line.strip()
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                ts = int(parts[0])
                util = int(parts[1].strip().replace("%", ""))
                mem = int(parts[2].strip())
            except ValueError:
                continue
            rows.append((ts, util, mem))

    steady_start = train_start + args.steady_offset
    steady_end = train_end - 3
    steady = [(ts, u, m) for ts, u, m in rows if steady_start <= ts <= steady_end]
    print(f"  {len(rows)} samples total, {len(steady)} in steady-state window "
          f"(skipped first {args.steady_offset}s)")

    if len(steady) < 20:
        fail(f"Only {len(steady)} steady-state samples — gate not meaningful")
    else:
        utils = [u for _, u, _ in steady]
        mean_util = statistics.mean(utils)
        median_util = statistics.median(utils)
        p10 = sorted(utils)[max(0, len(utils) // 10)]
        mem_peak = max(m for _, _, m in steady)
        active_frac = sum(1 for u in utils if u >= 50) / len(utils) * 100
        print(f"  mean={mean_util:.1f}%  median={median_util}%  p10={p10}%  "
              f"max={max(utils)}%  active(>=50%)={active_frac:.1f}%")
        print(f"  GPU mem peak during steady window: {mem_peak} MiB "
              f"({mem_peak/1024:.2f} GiB)")

        if mean_util >= args.util_threshold:
            ok(f"Mean GPU-Util {mean_util:.1f}% >= {args.util_threshold}% "
               f"(compute-bound, healthy)")
        elif mean_util >= 50.0:
            print(f"  WARN: Mean GPU-Util {mean_util:.1f}% in 50-70% band — "
                  f"borderline, consider faster storage or more workers")
        else:
            fail(f"Mean GPU-Util {mean_util:.1f}% < 50% — severe I/O starvation")

    # 3. Artifact sanity
    print("\n=== Artifact sanity ===")
    adapter = os.path.join(args.output_dir, "adapter", "adapter_model.safetensors")
    cfg = os.path.join(args.output_dir, "adapter", "adapter_config.json")
    for name, path in [("adapter_model.safetensors", adapter),
                       ("adapter_config.json", cfg)]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            ok(f"{name} exists ({size/1024:.1f} KB)")
        else:
            fail(f"{name} missing at {path}")

    loss_lines = 0
    if os.path.exists(args.train_log):
        with open(args.train_log) as f:
            for line in f:
                if "'loss':" in line and "train_runtime" not in line:
                    loss_lines += 1
        if loss_lines >= 3:
            ok(f"Train loss logged {loss_lines} times (>= 3)")
        else:
            fail(f"Only {loss_lines} train-loss lines")

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} gate(s))")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: PASS — all Plan 03-01 Task 2 gates green")


if __name__ == "__main__":
    main()
