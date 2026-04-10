---
fileClass: Log
name: phase3-vram-tune-infrastructure
description: Added Plan 03-02 VRAM tuning infrastructure (tune_vram.py grid runner + 4 new CLI flags on train_standard_lora.py)
created: 2026-04-10
updated: 2026-04-10
tags: [phase3, vram, tuning, lora, infrastructure]
aliases: []
status: complete
subtype: tuning
---

# Phase 3 VRAM Tuning Infrastructure (Plan 03-02 Task 1)

## Summary

Built the tooling Plan 03-02 Task 2 needs to run a 6-cell VRAM/throughput
tuning grid on the A6000 host. No training was launched in this entry —
only code + a `--dry_run` smoke test.

## Changes

### `scripts/training/train_standard_lora.py`

Four new CLI flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--vram_config PATH` | None | Load launch flags from a JSON file. Applied to args for flags NOT explicitly passed on the CLI. Lets Plan 03-03 launch with one arg. |
| `--no_grad_checkpoint` | False | Disable gradient checkpointing. Activation memory grows 2-4x but per-step time drops ~25-35%. |
| `--optim {adamw_torch,adamw_torch_fused}` | `adamw_torch_fused` | Optimizer implementation. Safe fallback to `adamw_torch` on ValueError/ImportError/RuntimeError. |
| `--eval_batch_size` | 8 | `per_device_eval_batch_size` (was hardcoded 1, which made eval O(N) slow). |

`apply_lora()` now takes a `gradient_checkpointing=True` kwarg; when False
it skips `model.gradient_checkpointing_enable()` and sets `use_cache=False`
directly on both the outer and thinker configs.

`training_config.json` gained three observability fields:

- `vram_config_source` — absolute path of the loaded JSON, or None
- `gradient_checkpointing_enabled` — actual runtime value
- `optim` — the optim that was actually used at runtime (captures the fused-to-torch fallback if it happened)

### `scripts/training/tune_vram.py` (new, ~530 lines)

- 6-cell default grid (A baseline fixed-4x4 GC-on torch | B fused | C bs=8
  grad_accum=2 | D dynamic 180/64 | E dynamic 300/96 | F dynamic 180/64 GC-off).
- Per-cell subprocess isolation (`subprocess.run` per cell) with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
  `CUDA_VISIBLE_DEVICES=0` in the env.
- Background `nvidia-smi --query-gpu=utilization.gpu,memory.used` polling via
  a bash loop so the log format matches Plan 01's gpu.log (`epoch_ts,util,mem`).
- Imports `parse_gpu_log`, `parse_timing`, `compute_gpu_util_stats`,
  `extract_step_times` from `validate_dryrun_gates.py` (the module-level
  helpers below).
- Writes `{cell_dir}/{metrics.json, gpu.log, timing.txt, train.log}` per cell.
- Pre-flight refuses to start if GPU0 memory.used > 1 GiB. Between cells,
  waits up to 30s for VRAM to drain below 1 GiB.
- Winner selection: sort passing cells by
  `(-mean_gpu_util, -tokens_per_sec, +median_step_time_s, +complexity_score)`
  with an Occam tiebreaker — if a simpler-complexity cell is within 2pp on
  mean_gpu_util, prefer it.
- Writes `grid_results.csv`, `grid_results.md`, and `vram_config.json`
  (with source/target_host/flags/measured/verdict/verdict_reason) at the
  end. `verdict` ∈ {go (>= 70%), go_with_warning (55-70%), blocked (< 55%)}.
- `--dry_run` flag prints per-cell commands without running them (used for
  smoke-testing the grid generator).
- `--only_cell ID` restricts the run to a single cell (reproducibility
  re-runs). In that mode it writes `repro_metrics.json` instead of
  overwriting the main grid artifacts.

### `scripts/training/validate_dryrun_gates.py`

Factored out reusable module-level helpers that `tune_vram.py` imports:

- `parse_gpu_log(path) -> list[(epoch_ts, util, mem_mib)]`
- `parse_timing(path) -> dict`
- `compute_gpu_util_stats(rows, train_start, train_end, steady_offset=30)`
- `extract_step_times(train_log_path) -> list[float]` — parses `s/it` or
  `it/s` from HF trainer tqdm progress lines.

`main()` behavior unchanged — same CLI, same gate output, same exit codes.

## Rule 3 deviation: subset_size=5000 instead of --full_dataset

**Plan 03-02 text:** "All cells share: `--full_dataset ...`"

**Blocking issue:** when a dynamic-batch cell (D/E/F) hits
`FrameBudgetBatchSampler(...)` it calls `train_dataset.get_durations()`,
which for a fresh dataset without a cache file runs `soundfile.info()` on
every row. On the 1.04M-sample train split on overlayfs with a 4-CPU
cgroup, that easily exceeds the plan's "20-25 min" budget for the whole
grid BY ITSELF on the first dynamic cell alone — and each cell has its
own output_dir, so the cache is not shared.

**Fix:** `SHARED_BASE_FLAGS["subset_size"] = 5000`. Per-step
VRAM/step-time/GPU-util is determined by batch × seq_len × activations,
not by dataset rows, so measurement is unaffected. 5000 samples comfortably
cover even cell E's worst case (50 × 96 = 4800 samples consumed).

**Scope:** tuning-only. Plan 03-03's real launch uses `--full_dataset` per
the locked flags, unchanged by this deviation.

## Rule 3 deviation: flag-count verify command

**Plan 03-02 Task 1 verify:** `python train_standard_lora.py --help | grep
-E "--vram_config|--no_grad_checkpoint|--optim|--eval_batch_size" | wc -l |
grep -q "^4$"`

**Issue:** argparse prints each flag twice in `--help` output (once in the
wrapped usage header, once in the options section). The plan's assumed
count of exactly 4 is impossible for ≥1 flags on a standard terminal width.
Observed count: 8 (or 9 when the help body referenced a flag by name).

**Fix:** equivalent strict check restricted to the options section:
```bash
python scripts/training/train_standard_lora.py --help 2>&1 \
  | sed -n '/^options:/,$p' \
  | grep -cE '^\s*(--vram_config|--no_grad_checkpoint|--optim|--eval_batch_size)\b'
```
This returns exactly 4. Documented in the commit body and the SUMMARY.

## Verify

```bash
python -c "import ast; ast.parse(open('scripts/training/train_standard_lora.py').read()); ast.parse(open('scripts/training/tune_vram.py').read()); print('SYNTAX OK')"
# SYNTAX OK

python scripts/training/train_standard_lora.py --help 2>&1 \
  | sed -n '/^options:/,$p' \
  | grep -cE '^\s*(--vram_config|--no_grad_checkpoint|--optim|--eval_batch_size)\b'
# 4

python scripts/training/tune_vram.py --dry_run 2>&1 | grep -c "^cell-"
# 6
```

## Commit

- `b2fc8b3` — feat(03-02): add tune_vram.py grid runner + --vram_config / --no_grad_checkpoint / --optim / --eval_batch_size flags

## Next

Task 2: run the 6-cell grid on this A6000 host (expected ~20-25 min wall-clock
with subset_size=5000), pick a winner by the deterministic selection rule,
freeze `outputs/full-lora-vram-tune/vram_config.json`, and re-run the
winner for reproducibility.
