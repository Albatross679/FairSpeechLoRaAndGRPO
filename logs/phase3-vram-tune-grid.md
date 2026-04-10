---
fileClass: Log
name: phase3-vram-tune-grid
description: Ran Plan 03-02 Task 2 6-cell VRAM tuning grid + Task 3 I/O remediation report; verdict blocked, winner cell C (fixed 8x2), Plan 03-03 launch gated on I/O mitigation
created: 2026-04-10
updated: 2026-04-10
tags: [phase3, vram, tuning, lora, grid, blocked, io]
aliases: []
status: complete
subtype: tuning
---

# Phase 3 VRAM Tuning Grid Run + Remediation Report (Plan 03-02 Tasks 2 + 3)

## Summary

Ran the 6-cell VRAM tuning grid from `scripts/training/tune_vram.py` on
the A6000 host. All six cells passed sanity gates (no OOM, finite loss
< 3.0, step time < 30 s). **No cell cleared the 55% `mean_gpu_util`
floor**; best was 42.6% (cells D, E). Verdict = `blocked` committed
to `outputs/full-lora-vram-tune/vram_config.json`. Task 3 fired and
produced `outputs/full-lora-vram-tune/io_remediation_report.md` with
6 ranked mitigation options — Plan 03-03 launch is gated on user
choice.

Also fixed a Rule 1 bug in `tune_vram.py:select_winner()` that caused
cell E to be (incorrectly) selected instead of cell C. See the
`fix(03-02): correct Occam tiebreak` commit for details.

## Grid

| Cell | Config | peak_vram_gb | mean_gpu_util | step_s | final_loss |
|------|--------|--------------|---------------|--------|------------|
| A | fixed 4×4, adamw_torch, GC on | 11.23 | 35.0% | 2.49 | 0.824 |
| B | fixed 4×4, adamw_torch_fused, GC on | 11.23 | 35.7% | 2.50 | 0.828 |
| **C (WINNER)** | **fixed 8×2, adamw_torch_fused, GC on** | **18.57** | **41.0%** | **2.01** | **0.413** |
| D | dynamic fb=180 max=64, GC on | 11.12 | 42.6% | 8.72 | 0.742 |
| E | dynamic fb=300 max=96, GC on | 16.10 | 42.6% | 16.08 | 0.679 |
| F | dynamic fb=180 max=64, GC off | 24.94 | 35.9% | 7.77 | 0.743 |

Winner: **cell C** per the plan's selection rule (Occam tiebreak —
all passing cells within 2 pp of the util leader, prefer lowest
complexity_score; C is the only fixed-batching cell in the 2 pp
tie band). Reproducibility re-run of cell C: peak_vram delta 0.00 GB,
mean_gpu_util delta +9.57 pp (41.0 → 50.57, within ±10 pp tolerance
but on the edge — measurement is noisier than expected).

## Diagnosis (Task 3)

The grid itself is decisive evidence that I/O is the binding
constraint on this host. Key signals:

1. Util capped near 42% across an 8× range of per-step compute
   (cell C 2.01 s vs cell E 16.08 s).
2. Turning off gradient checkpointing *lowered* util (cell F 35.9 vs
   cell D 42.6) — faster step time = less prefetch window = more
   GPU wait on data = textbook I/O-bound signature.
3. Fused vs non-fused adam is a no-op for util (A→B +0.7 pp) —
   optimizer is not on the critical path.
4. Doubling the microbatch (A→C, +6 pp) is the only meaningful
   util gain — amortizing data transfer over a bigger batch.

Host constraints: 4-core cgroup on AMD EPYC 9554, overlayfs storage,
1.13M small MP3 files. `num_workers > 4` gave no speedup per Plan 01.

## Mitigation options (ranked, see io_remediation_report.md)

| # | Option | Impact | Cost | Notes |
|---|--------|--------|------|-------|
| 1 | WebDataset pre-decode shards | 3–5× | ~50 GB disk, ~4–6h preprocessing, medium code | Root-cause fix |
| 2 | Stage rotating subset into /dev/shm | 1.5–2.5× | ~15 GB RAM, small code | Needs combining with Option 3 |
| 3 | Warm page cache before training | +5–10 pp | 0 code, ~20 GB RAM | Zero-cost pre-flight |
| 4 | Lower num_workers to 2 | ±5 pp | 0 code | Zero-cost pre-flight |
| 5 | Migrate to 8+ core NVMe host | 1.7–3× | $, ~1h ops, data copy | Buys around the constraint |
| 6 | Accept 42% util, run Plan 03-03 as-is | — | ~73h GPU time | Fallback only |

Recommendation: Options 4 + 3 as a zero-cost pre-flight; if mean
util still < 55%, pick Option 1 (primary) or Option 5 (alternative).
Do NOT default to Option 6.

## Commits

- `e537ab2` — `fix(03-02): correct Occam tiebreak in tune_vram.select_winner`
- `d99045c` — `chore(03-02): VRAM tuning grid results + frozen vram_config.json (winner: cell C)`
- `6e21867` — `docs(03-02): Task 3 I/O remediation report for blocked verdict`
- (this summary/metadata commit) — `docs(03-02): complete VRAM tuning + frozen config`

## Plan 03-03 status

**BLOCKED.** Waiting on user decision about which I/O mitigation
option to pursue. See
`outputs/full-lora-vram-tune/io_remediation_report.md` and
`.planning/phases/03-full-sft-lora-training/03-02-SUMMARY.md`
§"Next Phase Readiness" for the decision context.
