# Phase 3: Full SFT LoRA Training - Context

**Gathered:** 2026-04-08 (updated from 2026-04-07 based on Phase 2.1 results)
**Status:** Ready for planning

<domain>
## Phase Boundary

Scale up SFT LoRA training from Phase 2's 2K-sample prototype to the full dataset (~45.3K samples: 26,471 Fair-Speech + 18,876 Common Voice). Uses the validated locked config from Phase 2.1 (rank=4, alpha=4, lr=3.47e-4, target_mlp=true, dropout=0.05). Trains until convergence with early stopping. Produces the production-quality baseline adapter with enough evaluation samples per demographic group for meaningful fairness metrics.

</domain>

<decisions>
## Implementation Decisions

### Dataset scope
- **D-01:** Use ALL available training data — all 26,471 Fair-Speech + all 18,876 Common Voice utterances (~45.3K total). No subsetting.
- **D-02:** Drop the equal FS/CV split constraint from Phase 2 (D-02 was for prototyping only). Full-scale training uses all data from both sources.
- **D-03:** Speaker-disjoint train/eval split (90/10). Note: manifests lack speaker_id columns, so this falls back to random stratified split (same limitation as Phase 2).

### Training configuration (updated from Phase 2.1)
- **D-04:** Use locked config from Phase 2.1 Stage 2: rank=4, alpha_ratio=1 (alpha=4), dropout=0.05, lr=3.465e-4, target_mlp=true, weight_decay=1.742e-4, lr_scheduler=constant, warmup_ratio=0.0, grad_accum_steps=1, use_rslora=false
- **D-05:** Train until convergence with early stopping (patience=3, metric=eval_loss). Max epochs set to 3 as a safety cap. No fixed epoch target.
- **D-06:** Effective batch size = 2 (batch_size=2, gradient_accumulation=1). Matches sweep-validated config exactly.
- **D-07:** Checkpoint every 4000 steps, keep last 3 checkpoints (save_total_limit=3)
- **D-08:** Evaluate every 2000 steps
- **D-09:** Support resume from checkpoint (--resume_from_checkpoint) for interruption recovery
- **D-10:** Run explicit VRAM profiling step before launching full training to confirm peak VRAM estimate (~10-10.5 GB) is within T4 15 GB budget
- **D-11:** Keep LR scheduler as constant (no warmup, no decay) — matching what the sweep optimized

### Evaluation strategy
- **D-12:** Evaluate on training eval split (~4.5K samples) for per-group fairness metrics
- **D-13:** Also evaluate on cv_dev.csv (1,854 samples) as independent held-out set
- **D-14:** With ~4.5K eval samples, most groups should exceed MIN_GROUP_SIZE=50 for meaningful fairness metrics

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
- `outputs/manifests/cv_train.csv` — Common Voice train manifest (18,876 rows)
- `outputs/manifests/cv_dev.csv` — Common Voice dev manifest (1,854 rows)

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

### Scale-Up Estimates (updated for new config)
- Full dataset: ~45,347 samples -> 90/10 split -> ~40,812 train + ~4,535 eval
- Steps per epoch: 40,812 / 2 (effective batch=2) = ~20,406
- At ~3.5 sec/step: ~19.8 hours per epoch
- Early stopping (patience=3, eval every 2000 steps) will terminate after 6000 steps of no improvement
- Estimated peak VRAM: ~10-10.5 GB (rank=4, target_mlp=true, single adapter)

### Script Modifications Needed
- `train_standard_lora.py`: `create_stratified_subset()` needs `--full_dataset` mode to skip subsetting and use all data
- `train_standard_lora.py`: Update HP values to match new locked_config (target_mlp=true, lr, dropout, wd, etc.)
- `train_standard_lora.py`: Add EarlyStoppingCallback (patience=3, metric=eval_loss)
- `train_standard_lora.py`: save_steps=200 and eval_steps=100 too frequent for 20K steps — change to 4000/2000
- `train_standard_lora.py`: Add VRAM profiling step before training launch
- `train_standard_lora.py`: grad_accum_steps=1 (was 2 in Phase 2)

### Key Config Delta (Phase 2 -> Phase 2.1)
- target_mlp: false -> true (adds LoRA to MLP gate/up/down_proj)
- dropout: 0.0 -> 0.05 (regularization)
- lr: 8.2e-4 -> 3.47e-4 (lower, more stable)
- weight_decay: 1.33e-5 -> 1.74e-4 (13x higher regularization)
- grad_accum_steps: 2 -> 1 (effective batch 4 -> 2)
- lr_scheduler: (default) -> constant (no decay)

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
