---
fileClass: Log
name: Phase 4 GRPO Infrastructure
description: Created custom GRPO training loop and reward computation for fairness-aware ASR fine-tuning
created: 2026-04-08
updated: 2026-04-08
tags:
  - grpo
  - phase-4
  - infrastructure
  - rl
aliases:
  - GRPO infrastructure
status: complete
subtype: feature
---

## What Changed

Created the core infrastructure for Phase 4 (RL Prototyping Before Scaling Up RL):

### New Files

1. **`scripts/training/reward.py`** — `RewardComputer` class implementing composite reward:
   - Accuracy: `R_acc = (1-lambda)(1-WER)` using jiwer + whisper_normalizer
   - Fairness: `R_fair = lambda(-|WER_g - WER_mean|)` with rolling window accumulation
   - WER floor check to prevent reward hacking (Pitfall #3)
   - Per-group WER diagnostics for monitoring
   - `frac_zero_std` metric for training signal health (Pitfall #1)

2. **`scripts/training/train_grpo.py`** — Custom GRPO training loop (~500 lines):
   - Two modes: `prototype` (single lambda) and `sweep` (3 lambdas: 0, 0.3, 0.7)
   - Dr. GRPO variant by default (no std normalization)
   - Per-token log-prob computation via teacher-forced forward pass (Pitfall #4)
   - Reference policy via `disable_adapter_layers()` (no second model copy)
   - Sequential candidate generation with VRAM cleanup
   - VRAMMonitor with leak detection
   - Clipped surrogate loss with optional KL penalty
   - W&B integration for experiment tracking

### Updated Files

3. **`.planning/phases/04-rl-prototyping-before-scaling-up-rl/04-CONTEXT.md`** — Full context document with 22 implementation decisions, canonical references, and deferred items.

4. **`.planning/ROADMAP.md`** — Updated Phase 4 goal, requirements mapping (GRPO-01 through GRPO-09, EXPT-01/04), and added Phase Details section.

## Design Decisions

- **TRL rejected:** GRPOTrainer lacks audio data_collator and can't handle Qwen3-ASR encoder-decoder architecture (confirmed via TRL docs + GitHub #2807).
- **G=2 start:** T4 15GB VRAM is tight; generate candidates sequentially with cache cleanup.
- **Dr. GRPO default:** Removes difficulty bias (std normalization) per arxiv:2503.20783.
- **Rolling fairness window (10 batches):** With batch_size=1, single-batch fairness signal is trivially zero.
- **Reuses existing infrastructure:** Model loading, LoRA injection, data loading, stratified sampling all imported from prior phases.
