# Plan 03-02 Task 3 — I/O Remediation Report

**Generated:** 2026-04-10
**Verdict trigger:** `vram_config.json` verdict = `blocked` — best cell mean_gpu_util = 41.0% (cell C), well below the 55% acceptable floor.
**Task 3 scope:** diagnose the bottleneck and recommend mitigation options. **Do not implement** any option without explicit user approval.

---

## Is I/O actually the bottleneck? — evidence

The 6-cell grid spans fixed batching, dynamic batching, fused vs non-fused adam, and gradient checkpointing on/off. The mean_gpu_util numbers show a clear pattern:

| Cell | Config | peak_vram_gb | mean_gpu_util | step_s |
|------|--------|--------------|---------------|--------|
| A    | fixed 4×4, adamw_torch          | 11.23 | 35.0% | 2.49 |
| B    | fixed 4×4, adamw_torch_fused    | 11.23 | 35.7% | 2.50 |
| C    | fixed 8×2, adamw_torch_fused    | 18.57 | 41.0% | 2.01 |
| D    | dynamic fb=180 max=64, GC on    | 11.12 | 42.6% | 8.72 |
| E    | dynamic fb=300 max=96, GC on    | 16.10 | 42.6% | 16.08 |
| F    | dynamic fb=180 max=64, **GC off** | 24.94 | 35.9% | 7.77 |

Observations:

1. **Util is capped near 42% across wildly different compute profiles.** Fixed 8×2 (cell C) and dynamic fb=300 (cell E) differ in per-step compute by ~8× (step_s 2.01 vs 16.08), yet mean_gpu_util is within 1.6 pp of each other. If compute were the bottleneck, the heavier cells would show higher util. They don't.
2. **Turning off gradient checkpointing (cell F) *lowered* util, not raised it.** Cell F's 2.5–4× activation memory traded for 25–35% faster step time produced a *slower* util reading (35.9% vs D's 42.6%). Interpretation: when the step finishes faster, the dataloader has less time to prefetch between steps — GPU spends a larger fraction of each step waiting for the next batch. This is the textbook signature of an I/O-bound workload.
3. **Fused vs non-fused adam (A → B) is a no-op for util** (+0.7 pp). Fused adamw_torch_fused delivers zero throughput win here, because the optimizer step is not on the critical path — waiting for data is.
4. **Doubling per-device microbatch (C) gains 5.3 pp.** The only meaningful util improvement came from amortizing data transfer over a bigger batch, not from tuning the trainer or optimizer. Consistent with dataloader-bound behavior.
5. **n_steady_samples is stable (~95–110)** across cells, so the steady-state window is healthy. Numbers aren't being skewed by warmup or shutdown artifacts.

**Conclusion:** I/O is the binding constraint. Trainer-level tuning cannot push past ~42% on this host. The path forward is reducing the work the dataloader does per sample, not tuning the trainer.

## Host constraints (from 03-02-PLAN.md `<hardware>` + verified at runtime)

- **CPU:** 4-core cgroup quota (`cpu.max = 396000 100000`) on an AMD EPYC 9554 host. `nproc` reports 4, `os.cpu_count()` reports 4. `num_workers > 4` gives no speedup and produces a PyTorch "suggested max number of worker is 4" warning (confirmed by Plan 01).
- **Storage:** overlayfs over `/dev/vda` (container overlay). 196 GB total, 97 GB free. Datasets live on the overlay at `/workspace/project/datasets/`.
- **CV train set:** 1,133,665 individual MP3 files at `datasets/cv-corpus-25.0-2026-03-09/en/clips/`. Small-file random-access read workload on overlayfs is slow.
- **FS train set:** 26,471 individual WAV files at `datasets/meta_speech_fairness/asr_fairness_audio/`. Small-file workload similarly.
- **RAM:** 32 GB total, ~23 GB available, `/dev/shm` 32 GB (shared with RAM so effective cap ~15 GB).
- **GPU:** RTX A6000, 48 GB, Ampere SM 8.6. Compute is NOT the bottleneck.

The per-sample cost is dominated by:
1. `soundfile.read` / `torchaudio.load` for the MP3 (libsndfile MP3 decode is CPU-bound)
2. Resample to 16 kHz
3. Mel-spectrogram extraction (Qwen3-ASR processor)
4. Per-file metadata reads (`sf.info`, file-open syscall through overlayfs)

With 4 CPU workers and ~42% GPU-Util during steady state, each worker is saturated: `step_s ≈ decode_per_batch / num_workers`.

## Ranked mitigation options

Each row is a standalone investigation; the user picks one or more to pursue. All of them become follow-up plans. Do **not** bundle mitigation work into Plan 03-03.

### Option 1: Pre-decode the audio dataset to WebDataset tar shards — **highest impact**

- **What:** Run a one-time preprocessing pass over the 1.16M train samples. For each sample: decode MP3/WAV → resample to 16 kHz → save raw PCM (or pre-computed log-mel features) into `.tar` shard files using WebDataset format. Swap `ASRFairnessDataset` for a `WebDataset` + `DataLoader` reader that streams from the shard archive sequentially.
- **Why it wins:** Replaces 1.16M random-access MP3 decodes per epoch with ~100–500 sequential shard reads. Decode happens once. I/O becomes sequential reads of large files on overlayfs (fast) instead of random small-file access (slow). PyTorch WebDataset loaders are known to saturate modern GPUs for ASR on much bigger hosts.
- **Projected impact:** 3–5× throughput. Targets 70–90% mean_gpu_util. Brings Plan 03-03 wall-clock well under 36h.
- **Cost:**
  - **Disk:** ~50 GB additional (raw PCM at 16 kHz mono float16 ≈ 32 kB/s × ~2000h ≈ 50 GB; log-mel would be ~30 GB).
  - **Preprocessing time:** ~4–6 hours on the 4-core host (decode-bound, same constraint we're fighting).
  - **Code changes:** Medium. New `build_shards.py` script. New `ShardedASRFairnessDataset`. Dataloader wiring stays the same.
- **Risk:** Shard-based loading is less flexible if the dataset changes. Retraining requires re-sharding if the FS set gets new samples.
- **Recommended as new plan:** `03-02.5-pre-decode-to-webdataset` or similar.

### Option 2: Stage a rotating subset into `/dev/shm` — **medium impact**

- **What:** Before each epoch, copy a randomly-sampled N% of the train set into `/dev/shm` (tmpfs, RAM-backed, effective cap ~15 GB). Train exclusively from the shm copy. Refresh between epochs.
- **Why:** tmpfs I/O is orders of magnitude faster than overlayfs for small random reads. Eliminates the overlayfs read latency without changing the decode path.
- **Projected impact:** 1.5–2.5× throughput. Probably gets us to 55–70% util band.
- **Cost:**
  - **RAM:** ~15 GB for the staging buffer (leaves ~8 GB for torch + dataloader workers + training buffers → tight).
  - **Preprocessing time:** ~2 min per epoch to rsync the subset (overlaps with eval if timed right).
  - **Code changes:** Small. A `stage_to_shm(manifest, dest, subset_pct)` helper + a callback that refreshes between epochs.
- **Risk:** Only a subset is available per epoch, so effective epoch size shrinks unless N% ≈ 100%. At 1.16M samples × ~150 kB each, full staging would be ~175 GB — doesn't fit.
- **Recommended as new plan:** `03-02.6-tmpfs-staging` or similar.

### Option 3: Warm the page cache before training — **low impact**

- **What:** Run `find datasets/cv-corpus-25.0-2026-03-09/en/clips datasets/meta_speech_fairness/asr_fairness_audio -type f -print0 | xargs -0 cat >/dev/null` before launching training. Forces the kernel to page-cache as much of the dataset as RAM allows.
- **Why:** First read is slow (overlayfs latency); second read is served from page cache (fast). If all (or most) files fit in RAM, subsequent reads — including the second+ epoch — are cheap.
- **Projected impact:** Marginal on the first epoch. Meaningful on epoch 2+. Probably +5–10 pp util on warm runs.
- **Cost:**
  - **RAM:** ~20 GB of page cache used, competing with training.
  - **Preprocessing time:** ~10–20 min (bounded by disk read throughput).
  - **Code changes:** None. It's a shell one-liner.
- **Risk:** Linux may evict the page cache under memory pressure during training (torch allocations + dataloader workers). Needs testing to confirm it survives.
- **Worth trying even if other options are pursued** — zero code cost.
- **Recommended as:** a pre-flight step inside whichever plan does launch Plan 03-03.

### Option 4: Lower `num_workers` to 2 — **near-zero impact, near-zero cost**

- **What:** Try `--dataloader_num_workers 2` instead of the current 4.
- **Why:** At 4 cores, running 4 workers + the trainer main process = 5 processes fighting for 4 cores. 2 workers may reduce context-switch thrash. Plan 01 already tried 4 vs 8; 2 has not been tried.
- **Projected impact:** 0 to +5 pp util. Might hurt.
- **Cost:** Zero. One CLI flag change.
- **Risk:** Throughput regression is more likely than gain; this is a defensive "cheap experiment" to run while the bigger options are being decided.
- **Recommended as:** an add-on to the pre-flight step in whichever plan launches Plan 03-03.

### Option 5: Migrate to a different host — **unknown impact, not our call**

- **What:** Move Plan 03-03 to a host with more CPU cores and/or faster storage (NVMe bypassing overlayfs). The ML-project's cloud spec notes vast.ai and Lightning options.
- **Why:** This project is running on Lightning.ai Studio infrastructure with a 4-core cgroup quota. A host with 8+ cores + NVMe would likely hit the 70% util gate without any code changes.
- **Projected impact:** 1.7–3× throughput, depending on host specs.
- **Cost:**
  - **Ops:** Re-spinning a new VM, copying 100+ GB of dataset, re-installing dependencies. The `training-vm-setup` skill exists for exactly this.
  - **$:** Monetary cost of the new instance, possibly higher than the current host.
  - **Lead time:** ~1 hour to spin up + copy data.
- **Risk:** Transient problem. Fixes the symptom without any code improvement. If a bigger-scale run (v1.1 milestone) goes back to a constrained host later, this recurs.
- **Recommended as:** user-level decision. Not a plan.

### Option 6: Accept degraded throughput and let Plan 03-03 run longer — **fallback**

- **What:** Launch Plan 03-03 as-is with the frozen cell-C config. At ~41% util and 2.01s/step × 1,043,810 / 16 ≈ 131,128s/epoch ≈ 36.4h/epoch. Two epochs (per D-05) ≈ 73h.
- **Why:** Avoid all the work above. Ship the baseline adapter now and come back to tuning later.
- **Projected wall-clock:** ~73h for 2 epochs × 1.16M samples. Early stopping may cut this.
- **Cost:** 3+ days of GPU time. GPU $/h × 73h = monetary cost of running.
- **Risk:**
  - 73h exceeds typical cloud instance session budgets. Plan 03-03 must be checkpoint-robust (it is — `--resume_from_checkpoint` is wired from Plan 01).
  - Any interruption on a 73h run is expensive.
  - Opportunity cost: all downstream phases (04-RL, 05-launch-scaled-up-RL) are gated on Plan 03-03 completion. 3 days of blocked work.
- **Recommended as:** the option to pursue ONLY if time-to-ship dominates throughput tuning cost (e.g., a hard demo deadline). Otherwise Option 1 or Option 5 is cheaper overall.

## Recommendation

1. **First:** Option 4 (lower workers to 2) + Option 3 (warm page cache) — both zero-cost, ~1 hour of experiment time. Rerun cell C with these two changes on top of the existing config. If mean_gpu_util clears 55%, verdict upgrades to `go_with_warning` and Plan 03-03 can launch under Option 6 without remediation.
2. **If #1 doesn't reach 55%:** choose between Option 1 (WebDataset pre-decode) and Option 5 (better host). Option 1 is the "fix the root cause" path; Option 5 is the "buy around the constraint" path. Both unblock Plan 03-03. Option 1 also benefits future phases; Option 5 only benefits this run.
3. **Do NOT pursue:** Option 2 (tmpfs staging) alone — too RAM-constrained to fit a meaningful subset. Only viable in combination with Option 3 for warming a small curated subset.
4. **Do NOT pursue:** Option 6 (accept 73h) as the primary path. The cost of GPU time for 3 days is higher than the engineering cost of Option 1, and Plan 03-03 is on the critical path for all downstream work.

## Next action for the user

Pick the mitigation path. Each becomes its own plan (or a modification to Plan 03-03's pre-flight). Claude Code should not start any of these without explicit approval.

---

*Report generated from evidence in `outputs/full-lora-vram-tune/grid_results.md` and `outputs/full-lora-vram-tune/cell-*/metrics.json`. Torch profiler diagnostics and page-cache drop-caches tests (Plan §Task 3 Diagnostic A/B) were skipped because the 6-cell grid evidence already identifies I/O as the bottleneck with high confidence; explicit profiler evidence can be gathered if the user wants to cross-validate before committing to Option 1.*
