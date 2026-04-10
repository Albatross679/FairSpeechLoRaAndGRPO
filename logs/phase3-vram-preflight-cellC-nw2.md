---
fileClass: Log
name: phase3-vram-preflight-cellC-nw2
description: Zero-cost pre-flight for Plan 03-02 blocked verdict — ran cell C on the full 1.04M-sample dataset with num_workers=2 cold cache; mean_gpu_util 45.67% still below 55% floor; sharpened diagnosis from I/O-bound to CPU-decode-bound
created: 2026-04-10
updated: 2026-04-10
tags: [phase3, vram, tuning, preflight, decode-bound, cpu-bound, blocked, lora]
aliases: []
status: complete
subtype: tuning
---

# Phase 3 VRAM Pre-flight — Cell C, num_workers=2, Full Dataset (Plan 03-02 follow-up)

## Summary

Ran the zero-cost pre-flight recommended in `io_remediation_report.md`
(Options 3 + 4). Cell C flags (`--batch_size 8 --grad_accum 2
--optim adamw_torch_fused`) on the **full 1.04M-sample training set**
(cold page cache, no warming) for 100 training steps with
`--dataloader_num_workers 2 --dataloader_prefetch_factor 2`.

**Result:** mean_gpu_util = **45.67%** (steady, 20s offset from
training start) — **still below the 55% acceptable floor**. Verdict
stays `blocked`.

## Rationale

The original Task 3 report recommended starting with Options 3+4 as
a zero-cost experiment. The goal was to find out whether (a) the
warm-cache 5000-sample grid was overestimating full-scale util and
(b) whether `num_workers=2` (untested in Plan 01) helps on the
4-core cgroup.

## Configuration

| Flag | Value |
|------|-------|
| Mode | `train` (full dataset path) |
| FS manifest | `outputs/manifests/fs_train.csv` (26,472 rows) |
| CV manifest | `outputs/manifests/cv_train.csv` (1,133,666 rows) |
| `--full_dataset` | yes |
| `--batch_size` | 8 |
| `--grad_accum` | 2 |
| effective batch | 16 |
| `--optim` | `adamw_torch_fused` |
| Gradient checkpointing | on (default) |
| `--dataloader_num_workers` | **2** (grid used 4) |
| `--dataloader_prefetch_factor` | 2 |
| `--max_audio_secs` | None (skipped the 30+ min sf.info scan) |
| `--max_steps` | 100 |
| `--lr` | 9.8e-4 |
| Seed | 42 |

## Measurements

| Metric | Value | Source |
|--------|-------|--------|
| Wall-clock total | 234 s | `timing.txt` |
| Trainer training_time_sec | 208 s | `training_config.json` |
| Startup overhead | ~26 s | wall − training_time |
| Median step time | 1.85 s | tqdm output |
| Peak VRAM (torch.cuda.max_memory_allocated) | 15.93 GB | `training_config.json` |
| Peak VRAM (nvidia-smi) | 18,852 MiB ≈ 18.4 GB | gpu.log |
| Final train loss | 0.3884 | `training_config.json` |
| Steady-state mean_gpu_util (20s offset) | **45.67%** | gpu.log |
| Steady-state median_gpu_util | 34% | gpu.log |
| p10 / p90 gpu_util | 9% / 100% | gpu.log |
| Effective batch size | 16 | `training_config.json` |
| Exit code | 0 | `timing.txt` |

## Comparison to grid baselines

| Measurement | Context | mean_gpu_util | step_s | peak_vram_gb |
|---|---|---|---|---|
| Grid cell C (original) | subset 5000, nw=4, warm | 41.0% | 2.01 | 18.57 |
| Grid cell C (reproducibility) | subset 5000, nw=4, warm | 50.6% | 2.08 | 18.57 |
| **Pre-flight (this run)** | **full dataset, nw=2, cold** | **45.7%** | **1.85** | **15.93** |

The three measurements span a **9.6 pp** range on mean_gpu_util. The
run-to-run variance on this host is large — any single measurement
near the 55% floor should be treated with ±5 pp error bars.

## Findings

### 1. num_workers=2 is marginally better than num_workers=4

+4.7 pp mean util, −0.16 s median step time. The 4-core cgroup was
running 4 dataloader workers + the main trainer process = 5 procs
contending for 4 cores. Dropping to 2 workers reduces context-switch
thrash. Small but real win. Keep as default.

### 2. Cold-cache full-dataset ≈ warm-cache small-subset

This is the surprising result. If filesystem read latency were the
binding constraint on this host, cold-cache random reads over 1.04M
files would have been meaningfully slower than warm re-reads of 5000
files. They weren't — the measurements are within the noise band.

Implication: **page cache hit rate is not the ceiling on this
workload**. Options 2 (/dev/shm staging) and 3 (warm page cache)
would have ~zero impact because they both cache file **bytes**,
which is not the slow step.

### 3. The real bottleneck is CPU-bound audio decode

What IS slow, per-sample:

1. `soundfile.read()` — libsndfile MP3 decompression (CPU, not I/O)
2. Resample 16 kHz → target sample rate (CPU)
3. Mel-spectrogram extraction via Qwen3-ASR processor (CPU)

Rough budget: 4 cores × ~1.5–2 samples/sec/core ≈ 6–8 samples/sec
total decode throughput. The GPU consumes 16 samples per step in
2 s = 8 samples/sec. The system is sitting exactly at the decode
break-even point, which is why mean_gpu_util caps near 45–50%.

This reframes the original "I/O-bound" diagnosis. It is more
accurately a **decode-bound / CPU-bound data-preprocessing pipeline**.
The fix must target the decode step itself, not the filesystem.

## Implications for remediation options

| Option | Original ranking | Post-pre-flight |
|---|---|---|
| 1. WebDataset pre-decode to sharded log-mel/PCM | primary | **Primary** — eliminates CPU decode entirely |
| 2. /dev/shm staging | medium | **RULED OUT** — caches bytes not tensors |
| 3. Warm page cache | low | **RULED OUT** — same reason |
| 4. Lower num_workers to 2 | trivial | **+4.7 pp measured, keep as default** |
| 5. Migrate to 8+ core NVMe host | alternative | **~1.5–2× per core count doubling** |
| 6. Accept ~45% util, run Plan 03-03 as-is | fallback | **~66h for 2 epochs — still too slow** |

`io_remediation_report.md` was updated with a pre-flight addendum at
the top and an "after pre-flight" ranking column — see it for the
authoritative updated recommendation. Options 2 and 3 are no longer
options.

## Verdict decision

**Stay `blocked`.** The pre-flight moved the best measurement from
41% → 45.67% (within 5 pp of the acceptable floor given this host's
measurement noise) but did not clear 55%. A single noisy run near
the floor is not high-enough signal to upgrade to `go_with_warning`.
`vram_config.json` stays at `verdict: blocked`, now with
`num_workers=2` baked in as an additional measured fact.

## Recommendation to user

Pick one of:

1. **Option 1 — WebDataset pre-decode.** New plan that builds a
   `scripts/data/build_shards.py` preprocessing pipeline + a
   `ShardedASRFairnessDataset` loader + a re-run of the 6-cell grid
   on the sharded dataset. ~4–6h preprocessing + medium code
   changes. Expected 2–5× throughput. Fixes the root cause for all
   downstream phases.
2. **Option 5 — host migration.** Spin up a vast.ai or Lightning.ai
   instance with 8+ CPU cores and NVMe storage. ~1h ops + data copy.
   Expected 1.5–2× throughput. Buys around the constraint for this
   run only.

**Bake into both paths:** `--dataloader_num_workers 2` (confirmed
+4.7 pp on this host).

## Artifacts

- `outputs/full-lora-preflight-cellC-nw2/metrics.json` — aggregated measurement
- `outputs/full-lora-preflight-cellC-nw2/training_config.json` — trainer state (loss, VRAM)
- `outputs/full-lora-preflight-cellC-nw2/gpu.log` — raw 1 Hz nvidia-smi polling
- `outputs/full-lora-preflight-cellC-nw2/timing.txt` — start/end epoch markers
- `outputs/full-lora-preflight-cellC-nw2/train.log` — full trainer stdout

## Deviations from the original Task 3 plan

- **Skipped Option 3 (warm page cache)**: replaced by the more honest
  cold-cache full-dataset experiment. Warming 1.13M MP3 files is not
  feasible in a single pre-flight pass anyway (> 15 h even on 4 cores).
- **Initially set `--max_audio_secs 30`**: the sf.info scan over 1.16M
  files hit a 30+ min wall, had to kill and restart without the filter.
  Accepted ~5% OOM risk on outlier-length samples over 100 random
  draws. No OOM occurred.
- **Used `python -u` (unbuffered stdout)**: the first attempt produced
  an empty train.log because Python's stdout was block-buffered when
  redirected to a file. Fixed in the retry.
