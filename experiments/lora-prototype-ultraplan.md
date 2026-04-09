---
fileClass: Experiment
name: LoRA Prototype Ultraplan
description: Systematic 4-phase LoRA fine-tuning validation for Qwen3-ASR-1.7B before scaling to full GRPO training
status: running
algorithm: LoRA (PEFT)
task: ASR fairness fine-tuning
hardware: Tesla T4 15GB
created: 2026-04-06
updated: 2026-04-06
tags: [lora, prototype, hyperparameter-tuning, qwen3-asr, fairness]
aliases: [ultraplan, lora-prototype]
---

# LoRA Prototype Ultraplan

**Principle:** Build from simple to complex. Each phase has a concrete gate. If a gate fails, fix before advancing. Scaling a broken pipeline multiplies waste.

## Environment

| Resource | Value |
|----------|-------|
| GPU | Tesla T4, 15GB VRAM |
| Model | Qwen3-ASR-1.7B (encoder 300M frozen, decoder ~1.4B LoRA-trainable) |
| Training data | Fair-Speech (26,471) + Common Voice (6,006) = 32,477 total |
| Eval data | Common Voice dev (1,515) |
| Mixed precision | bfloat16 |
| Gradient checkpointing | Enabled (use_reentrant=False) |

## Data Summary

| Dataset | Samples | Demographic Axis | Groups |
|---------|---------|------------------|--------|
| Fair-Speech | 26,471 | ethnicity | 7 (Black/AA, White, Native American, Asian, Hispanic, Pacific Islander, Middle Eastern) |
| Common Voice (extracted) | 6,006 | accent | 14 (us, england, indian, canada, etc.) + 3,357 unlabeled |
| Common Voice dev | 1,515 | accent | 11 + 1,362 unlabeled |

## VRAM Budget (Tesla T4 15GB)

| Component | Estimated VRAM |
|-----------|---------------|
| Qwen3-ASR-1.7B base (bf16) | ~3.4 GB |
| LoRA adapters (rank 16) | ~14 MB |
| Optimizer states (AdamW) | ~28 MB |
| Gradient checkpointing overhead | ~0.5 GB |
| Activations (batch_size=2) | ~2-4 GB |
| **Total** | **~6-8 GB** |
| Headroom | ~7-9 GB |

---

## Phase 1: Pipeline Sanity

**Goal:** Verify end-to-end training works before any real training.
**Time budget:** < 30 minutes.
**Script:** `scripts/training/phase1_sanity.py`

### Checks

| # | Check | Method | Pass Criterion | Failure Means |
|---|-------|--------|----------------|---------------|
| 1 | Data loading | Load 1 batch via DataCollatorForQwen3ASR | Shapes: input_ids (B, T), input_features (B, C, F), labels (B, T) with -100 masking | Preprocessing or collation bug |
| 2 | Loss at init | Forward pass on 1 batch, check loss | Loss ≈ 10-12 (≈ -ln(1/V) for V≈150K vocab) | Wrong loss fn, bad init, or label masking error |
| 3 | Single-batch overfit | Train 1 batch for 200 steps | Loss → < 0.1 | Broken backprop, wrong optimizer, or frozen layers |
| 4 | Gradient flow | Check grad norms per LoRA layer after 1 step | All LoRA params have nonzero gradients | Detached graph or accidental freezing |
| 5 | Eval loop | Generate transcription from 1 audio sample | Output is non-empty text, no errors | Decode bug, generate() incompatibility with PEFT |
| 6 | Checkpoint round-trip | Save adapter → reload → compare outputs | Identical output tokens (torch.equal) | Serialization bug |

### Commands

```bash
# Run all Phase 1 checks
python scripts/training/phase1_sanity.py \
    --fs_manifest outputs/manifests/fs_train.csv \
    --cv_manifest outputs/manifests/cv_train.csv

# Expected output: 6/6 PASS
```

### Gate
All 6 checks pass. If any fail, debug and fix before Phase 2.

### Actual Results (2026-04-06)

**GATE PASSED** in 186 seconds. Key numbers:
- Initial loss: 4.43 (model already knows English ASR well)
- Overfit: 4.43 -> 0.0000 in 200 steps (converged by step 50)
- VRAM peak: 7.74GB (comfortable margin on 15GB T4)
- Trainable params: 6.4M (0.31% of 2.05B)
- Training speed: ~0.8s/step (batch_size=2)

### Contingencies

| Failure | Root Cause | Fix |
|---------|-----------|-----|
| Loss at init is NaN | Label masking wrong, -100 not ignored | Check collator prefix masking logic |
| Single-batch won't overfit | LoRA not actually training | Verify target_modules regex matches actual layer names |
| Gradient all zeros | Encoder not frozen properly, graph detached | Check requires_grad flags, use_reentrant setting |
| Generate fails with PEFT | PeftModel.generate() delegation broken | Use model.base_model.generate() or merge adapter first |
| Checkpoint differs | bf16 rounding during save/load | Compare with atol=1e-3 instead of exact |

---

## Phase 2: Small-Scale HP Exploration

**Goal:** Find promising hyperparameter regions using 500-sample subset.
**Time budget:** 2-4 hours.
**Script:** `scripts/training/phase2_hp_sweep.py`

### Data Subset

- **Size:** 500 samples (350 Fair-Speech + 150 Common Voice, proportional to full data)
- **Stratification:** By ethnicity (FS) and accent (CV), ensuring all groups represented
- **Fixed seed:** 42 for reproducible subsets

### HP Search Space

| Parameter | Range | Type | Priority |
|-----------|-------|------|----------|
| Learning rate | [1e-5, 1e-3] | log-uniform | 1 (highest) |
| LoRA rank | {4, 8, 16, 32} | categorical | 2 |
| LoRA alpha | {1x rank, 2x rank} | categorical | 3 |
| Target modules | {attn-only, attn+MLP} | categorical | 4 |
| Weight decay | [1e-6, 1e-2] | log-uniform | 5 |

### Search Method

- **Optuna TPE** (Tree-structured Parzen Estimator)
- **Trials:** 20
- **Steps per trial:** 100 (≈ 10% of expected full training)
- **Metric:** Validation loss on held-out 50 samples from subset
- **Pruning:** Optuna MedianPruner (stop bad trials early)
- **Tracking:** W&B project `lora-prototype-sweep`

### Expected LoRA Config Ranges (from knowledge/lora-hyperparameter-sensitivity.md)

| Config | Conservative | Moderate | Aggressive |
|--------|-------------|----------|------------|
| Rank | 4 | 16 | 32 |
| Alpha | 8 | 32 | 64 |
| LR | 5e-5 | 2e-4 | 5e-4 |
| Dropout | 0.1 | 0.05 | 0.0 |
| Targets | attn-only | attn-only | attn+MLP |

### Commands

```bash
# Run HP sweep
python scripts/training/phase2_hp_sweep.py \
    --fs_manifest outputs/manifests/fs_train.csv \
    --cv_manifest outputs/manifests/cv_train.csv \
    --n_trials 20 \
    --steps_per_trial 100 \
    --subset_size 500 \
    --output_dir outputs/hp-sweep

# Expected output: Best params, top-3 configs with scores
```

### Gate
Top 2-3 configurations show consistent relative ranking. If rankings are noisy, increase subset size to 1000 or steps to 200.

### What Transfers to Full Scale

| Transfers Well | Does NOT Transfer |
|---------------|-------------------|
| Relative ranking of rank values | Exact optimal LR |
| attn-only vs attn+MLP winner | Dropout rate |
| LR schedule shape | Weight decay strength |
| Alpha/rank ratio | Exact number of steps |

---

## Phase 3: Medium-Scale Validation

**Goal:** Confirm Phase 2 findings at 5x data scale.
**Time budget:** 4-8 hours.
**Script:** `scripts/training/phase3_validation.py`

### Setup

| Parameter | Value |
|-----------|-------|
| Data fraction | 5,000 samples (3,500 FS + 1,500 CV) |
| Configs to test | Top 3 from Phase 2 |
| Steps per config | 500 |
| Eval frequency | Every 50 steps |
| Seeds per config | 2 (for variance) |

### LR Scaling Rule

When batch size scales by k from Phase 2:
- AdamW: Multiply LR by sqrt(k)
- If batch_size stays the same: no LR adjustment needed

### Checks

| Check | How | Pass Criterion | Failure Action |
|-------|-----|----------------|----------------|
| Ranking preserved | Compare val loss of top 3 | Same relative order as Phase 2 | Subset unrepresentative — resample |
| Loss curve healthy | Plot train/val loss | Smooth descent, no divergence | LR too high or data issue |
| No overfitting | Train-val gap | Val loss not rising while train falls | Reduce dropout or add weight decay |
| Metric trajectory | Plot WER over steps | Still improving at step 500 | May need more steps at full scale |
| VRAM stable | Monitor peak memory | No growth over training | Memory leak in data pipeline |
| Demographic balance | Per-group loss variance | No group with 3x average loss | Adjust sampling or add fairness loss |

### Commands

```bash
# Run medium-scale validation
python scripts/training/phase3_validation.py \
    --fs_manifest outputs/manifests/fs_train.csv \
    --cv_manifest outputs/manifests/cv_train.csv \
    --hp_results outputs/hp-sweep/best_params.json \
    --subset_size 5000 \
    --steps 500 \
    --output_dir outputs/medium-validation

# Expected output: Ranking comparison table, loss curves, per-group analysis
```

### Gate
Top config holds lead at medium scale. Loss curves healthy. No overfitting. Per-group loss reasonably balanced.

---

## Phase 4: Full-Scale Standard LoRA Baseline

**Goal:** Train the final standard LoRA baseline on full data.
**Time budget:** 8-16 hours.
**Script:** `scripts/training/lora_baseline.py` (evolved from prototype)

### Pre-Flight Checklist

- [ ] Best HP config from Phase 3 locked — no more changes
- [ ] Full dataset loaded and verified (32,477 samples)
- [ ] Checkpoint saving every 500 steps
- [ ] Evaluation every 250 steps on CV dev set (1,515 samples)
- [ ] W&B logging enabled (`lora-baseline` project)
- [ ] GPU memory verified with 1 full batch
- [ ] Estimated wall-clock: calculate from Phase 3 throughput
- [ ] Early stopping: patience = 3 eval cycles

### Training Configuration (Template — fill from Phase 3)

```python
# Locked from Phase 3 validation
LORA_RANK = ???           # From sweep
LORA_ALPHA = ???          # From sweep
LORA_DROPOUT = ???        # From sweep
TARGET_MODULES = ???      # From sweep (attn-only or attn+MLP)
LEARNING_RATE = ???       # From sweep, possibly scaled
BATCH_SIZE = ???          # From sweep
WEIGHT_DECAY = ???        # From sweep

# Full-scale parameters
MAX_STEPS = ???           # Extrapolated from Phase 3 convergence
WARMUP_STEPS = ???        # 5-10% of MAX_STEPS
EVAL_STEPS = 250
SAVE_STEPS = 500
GRADIENT_ACCUMULATION = ???  # If batch_size needs effective increase
```

### Monitoring Checklist (in order)

1. **Gradient norms** — Stable? (exponential growth = explosion)
2. **LR schedule** — Warmup executing correctly?
3. **Train loss** — Decreasing smoothly? (spikes OK if they recover)
4. **Val loss** — Tracking train loss? (gap growing = overfitting)
5. **WER** — Trend improving?
6. **Per-group WER** — Any group diverging?

### Stop Conditions

| Condition | Action |
|-----------|--------|
| Loss NaN/Inf | Kill. Check gradient clipping, LR, corrupted samples. |
| Val loss rising > 3 eval cycles | Overfitting. Stop, use best checkpoint. |
| WER plateaued > 5 eval cycles | Diminishing returns. Stop, use best checkpoint. |
| GPU memory growing | Memory leak. Kill, fix, restart from checkpoint. |

### Gate
- Standard LoRA baseline WER established on full eval set
- Per-group WER gap documented (this becomes the "before GRPO" reference)
- Adapter saved and round-trip verified
- Results logged to W&B

---

## Compute Budget

| Phase | % of Total | Estimated Time | GPU Hours |
|-------|-----------|----------------|-----------|
| Phase 1 (sanity) | < 1% | < 30 min | 0.5 |
| Phase 2 (HP sweep) | 15-25% | 2-4 hours | 4 |
| Phase 3 (validation) | 10-20% | 4-8 hours | 8 |
| Phase 4 (full run) | 45-55% | 8-16 hours | 16 |
| Buffer / reruns | 10-15% | - | 3 |
| **Total** | 100% | ~16-30 hours | ~31.5 |

---

## Decision Log

| ID | Decision | Rationale |
|----|----------|-----------|
| P-01 | Use 500 samples for Phase 2 (not 200) | 200 is the Phase 1.1 prototype size; Phase 2 needs more for stable HP ranking |
| P-02 | Sweep rank {4, 8, 16, 32}, not just current r=16 | knowledge/lora-hp-sensitivity.md says r=32 may be optimal for ASR |
| P-03 | Include attn+MLP in sweep | D-04 deferred MLP to later phase, but HP sweep is the right place to test |
| P-04 | Use Optuna TPE over grid search | Grid wastes trials; TPE is more sample-efficient with 20 trials |
| P-05 | 2 seeds per config in Phase 3 | Catches noise vs real ranking differences |
| P-06 | Tesla T4 15GB (not RTX A4000 16GB) | Actual hardware; VRAM budget still comfortable at ~6-8GB |
