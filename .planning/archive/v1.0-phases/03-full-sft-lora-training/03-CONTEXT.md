# Phase 3: Full SFT LoRA Training - Context

**Gathered:** 2026-04-08 (updated from 2026-04-07 based on Phase 2.1 results)
**Status:** Ready for planning

<domain>
## Phase Boundary

Scale up SFT LoRA training from Phase 2's 2K-sample prototype to the full dataset (~1.16M samples: 26,471 Fair-Speech + 1,133,318 Common Voice including unlabeled accent samples). Uses the validated locked config from Phase 2.1 with adjusted batch size and learning rate for the larger dataset. Trains until convergence with early stopping. Produces the production-quality baseline adapter. Evaluates fairness only on the accented subset (FS + accented CV). LibriSpeech test-clean kept as eval-only benchmark.

</domain>

<decisions>
## Implementation Decisions

### Dataset scope
- **D-01:** Use ALL available training data — all 26,471 Fair-Speech + all 1,133,318 Common Voice utterances (~1.16M total). Includes CV samples without accent labels (they improve general ASR quality even though they can't be evaluated for accent fairness).
- **D-02:** Drop the equal FS/CV split constraint from Phase 2 (D-02 was for prototyping only). Full-scale training uses all data from both sources. CV:FS ratio is ~43:1.
- **D-03:** Speaker-disjoint train/eval split (90/10). CV has speaker_id (client_id); FS lacks it (assigned pseudo-IDs).
- **D-15:** LibriSpeech test-clean (2,620 samples) kept as eval-only benchmark — not included in training.

### Training configuration (updated from Phase 2.1)
- **D-04:** Use locked config from Phase 2.1 Stage 2 as base: rank=4, alpha_ratio=1 (alpha=4), dropout=0.05, target_mlp=true, weight_decay=1.742e-4, use_rslora=false. LR and batch size adjusted for 25x larger dataset (see D-06, D-16).
- **D-05:** Train until convergence with early stopping (patience=3, metric=eval_loss). Max epochs set to 2 as a safety cap (1 epoch expected to suffice with 1.16M samples).
- **D-06:** Effective batch size = 16 (batch_size=4, gradient_accumulation=4). Increased from sweep config (eff=2) because 1.16M samples at eff=2 would take 290h/epoch. At eff=16: ~36h/epoch on RTX 3090.
- **D-07:** Checkpoint every eval step (save_steps=eval_steps), keep last 3 checkpoints (save_total_limit=3). Required for load_best_model_at_end with EarlyStoppingCallback.
- **D-08:** Evaluate every 5000 steps (~13 evals per epoch at 65K steps/epoch)
- **D-09:** Support resume from checkpoint (--resume_from_checkpoint) for interruption recovery
- **D-10:** Run explicit VRAM profiling step before launching full training to confirm peak VRAM within RTX 3090 24 GB budget
- **D-11:** Keep LR scheduler as constant (no warmup, no decay) — matching what the sweep optimized
- **D-16:** LR scaled from 3.465e-4 to ~9.8e-4 using sqrt rule (sqrt(16/2) = 2.83x) to account for 8x larger effective batch. Dry-run validates this doesn't cause instability.

### Evaluation strategy
- **D-12:** Evaluate on training eval split (~116K samples) for overall WER. For fairness metrics, filter to accented subset only (FS ethnicity + CV accented).
- **D-13:** Also evaluate on cv_dev.csv as independent held-out set (filter to accented samples for fairness)
- **D-14:** With ~116K eval samples, all demographic groups will exceed MIN_GROUP_SIZE=50
- **D-17:** LibriSpeech test-clean (2,620 samples) as third eval benchmark — overall WER only (no demographic axis)

### Claude's Discretion
- Max epoch cap (2 vs 3) based on expected convergence time
- W&B run organization and naming
- Whether to load best checkpoint or last checkpoint for final evaluation

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 2.1 outputs (prerequisites — updated)
- `outputs/standard-lora/locked_config.json` — **Updated** validated HP config from Phase 2.1 Stage 2 (rank=4, alpha_ratio=1, lr=3.47e-4, target_mlp=true, dropout=0.05, wd=1.74e-4, lr_scheduler=constant, use_rslora=false)
- `outputs/packed-sweep/config_comparison.json` — Phase 2 vs Phase 2.1 comparison (shows Stage 2 winner beat Phase 2 sweep by 10.5%)
- `outputs/packed-sweep/vram_profile.json` — VRAM profiles by rank/target_mlp (rank=4+mlp=true: ~328 MB per adapter)

### Phase 2 outputs (still relevant)
- `outputs/standard-lora/training_config.json` — Phase 2 training details (500 steps, 2K subset, peak 9.97 GB VRAM with target_mlp=false)
- `outputs/standard-lora/eval/analysis_standard-lora.json` — Phase 2 eval baseline (WER ~2-3%)
- `scripts/training/train_standard_lora.py` — Training script to modify for full-scale
- `scripts/training/evaluate_adapter.py` — Evaluation bridge (reuse as-is)

### Existing infrastructure
- `scripts/training/data_loader.py` — ASRFairnessDataset, DemographicStratifiedSampler
- `scripts/training/data_collator.py` — DataCollatorForQwen3ASR
- `outputs/manifests/fs_train.csv` — Fair-Speech manifest (26,471 rows)
- `outputs/manifests/cv_train.csv` — Common Voice train manifest (1,133,318 rows, regenerated 2026-04-10 from full CV 25 corpus for Phase 3 scale-up per D-01; was 18,876 in Phase 2 era)
- `outputs/manifests/cv_dev.csv` — Common Voice dev manifest (14,133 rows; was 1,854 in Phase 2 era after accent filtering)

### Research and architecture
- `.planning/research/PITFALLS.md` — Pitfall #9 (speaker-disjoint splits), Pitfall #3 (premature scaling — Phase 3 only runs after Phase 2.1 validation)

### Requirements
- `.planning/REQUIREMENTS.md` — BASE-01, EVAL-01

</canonical_refs>

<code_context>
## Existing Code Insights

### Phase 2.1 Training Results (updated baseline)
- Locked config: rank=4, alpha_ratio=1, dropout=0.05, lr=3.465e-4, target_mlp=true, wd=1.742e-4, lr_scheduler=constant, warmup_ratio=0.0, grad_accum_steps=1
- Phase 2.1 Stage 2 best eval_loss: 0.1424 (100-step sweep trial, 500 samples)
- Phase 2.1 Stage 2 peak VRAM: 12,153 MB (packed multi-adapter — single adapter will be much lower)
- VRAM profile rank=4+target_mlp=true: ~328 MB adapter overhead

### Scale-Up Estimates (updated 2026-04-10 for actual A6000 host)
- Full dataset: ~1,159,789 samples -> 90/10 split -> ~1,043,810 train + ~115,979 eval
- Steps per epoch: 1,043,810 / 16 (effective batch=16) = ~65,238
- **Wall-clock: unknown until Plan 03-02 measures it** on the actual host. The original "~2 sec/step → 36h/epoch" estimate assumed an RTX 3090 with fast I/O — neither is true here. Plan 03-02's VRAM tuning grid projects wall-clock from `winner.median_step_time_s × 65,238 steps/epoch`.
- Early stopping (patience=3, eval every 5000 steps) will terminate after 15000 steps of no improvement
- **Peak VRAM budget:** < 44 GB (4 GB safety margin on the A6000's 48 GB). Plan 03-01 dry-run measured 7.87 GB peak at the conservative bs=4/grad_accum=4/GC-on config; Plan 03-02 explores larger configs within the 44 GB ceiling.

### Hardware (actual, 2026-04-10)
- **GPU:** NVIDIA RTX A6000, 48 GB GDDR6 ECC, SM 8.6 (Ampere), boost 2100 MHz
- **Torch:** 2.10.0+cu128, cudnn 9.10, driver 580.95.05
- **CPU:** AMD EPYC 9554 host (124 cores) but cgroup-limited to **3.96 cores** (`cpu.max = 396000 100000`) — dataloader worker parallelism is capped at ~4
- **RAM:** 32 GB total, ~23 GB available; /dev/shm tmpfs 32 GB (shares RAM)
- **Storage:** overlayfs over `/dev/vda`, 196 GB / 97 GB free. Datasets on the same overlay — no separate NVMe mount. This is the dominant I/O constraint surfaced in Plan 03-01.
- **Dataset locations:** `/workspace/project/datasets/cv-corpus-25.0-2026-03-09/en/clips/` (1.13M MP3s) and `/workspace/project/datasets/meta_speech_fairness/asr_fairness_audio/` (26K WAVs)

**A6000 vs RTX 3090 perf note:** A6000 has 2× the VRAM of a 3090 but ~20% less memory bandwidth (768 vs 936 GB/s). At matched batch size, compute throughput is roughly parity. The A6000's advantage is *capacity* (run larger effective batches), not *raw speed*.

### Script Modifications (Plan 03-01 — done; Plan 03-02 — planned)
**Plan 03-01 added:**
- `train_standard_lora.py`: `load_full_dataset()` loads ALL data without subsetting
- `train_standard_lora.py`: `--full_dataset`, `--num_epochs`, `--locked_config_path`, `--save_total_limit`, `--resume_from_checkpoint`, `--lr`, `--dataloader_num_workers`, `--dataloader_prefetch_factor`
- `train_standard_lora.py`: `EarlyStoppingCallback` (patience=3, metric=eval_loss)
- `train_standard_lora.py`: Dynamic `save_steps`/`eval_steps` based on dataset size
- `train_standard_lora.py`: `load_model_and_processor()` uses `attn_implementation="sdpa"` by default
- `train_standard_lora.py`: `TrainingArguments` uses `dataloader_pin_memory=True`, `persistent_workers=True`, `prefetch_factor`
- `scripts/training/validate_dryrun_gates.py`: reusable pre-flight VRAM/GPU-Util gates script

**Plan 03-02 will add:**
- `train_standard_lora.py`: `--vram_config PATH` to consume a frozen JSON launch config
- `train_standard_lora.py`: `--no_grad_checkpoint` to disable gradient checkpointing (default: on, backwards-compat with Plan 03-01 dry-run)
- `train_standard_lora.py`: `--optim {adamw_torch,adamw_torch_fused}` with safe fallback (default: fused)
- `train_standard_lora.py`: `--eval_batch_size` (default: 8, was hardcoded 1)
- `train_standard_lora.py`: `training_config.json` observability fields: `vram_config_source`, `gradient_checkpointing_enabled`, `optim`
- `scripts/training/tune_vram.py`: 6-cell grid runner that picks a Pareto-optimal config and freezes it as `outputs/full-lora-vram-tune/vram_config.json`

### Key Config Delta (Phase 2 -> Phase 3)
- target_mlp: false -> true (adds LoRA to MLP gate/up/down_proj)
- dropout: 0.0 -> 0.05 (regularization)
- lr: 8.2e-4 -> 9.8e-4 (sqrt-scaled for batch=16)
- weight_decay: 1.33e-5 -> 1.74e-4 (13x higher regularization)
- grad_accum_steps: 2 -> 4 (effective batch 4 -> 16 with batch_size=4)
- lr_scheduler: cosine -> constant (no decay)
- dataset: 2K subset -> 1.16M full

</code_context>

<specifics>
## Specific Ideas

- Add `--full_dataset` flag rather than making subset_size optional, for clarity
- Early stopping removes the need for a fixed epoch target — just set max_epochs=3 as a safety cap
- Save adapter to a different output directory (e.g., `outputs/full-lora/adapter`) to preserve Phase 2's prototype adapter
- VRAM profiling step: run a short forward+backward pass (10 steps) with full config before committing to full training

</specifics>

<deferred>
## Deferred Ideas

None — scope is clear from roadmap definition

</deferred>

---

*Phase: 03-full-sft-lora-training*
*Context gathered: 2026-04-08*
