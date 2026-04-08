---
fileClass: Experiment
name: PLoRA Stage 1 Packed Sweep
description: Stage 1 validation of PLoRA packed LoRA adapter training with 20 trials on T4 GPU
status: complete
algorithm: PLoRA (Packed LoRA) with Optuna TPE
task: Hyperparameter sweep for ASR LoRA fine-tuning
hardware: Tesla T4 (15 GB VRAM)
created: 2026-04-07
updated: 2026-04-07
tags: [plora, lora, hp-sweep, optuna, phase-2.1]
aliases: [stage1-sweep]
---

# PLoRA Stage 1 Packed Sweep

## Setup

- **Model:** Qwen3-ASR-1.7B (bf16)
- **Dataset:** 500-sample stratified subset (FairSpeech + CommonVoice)
- **Trials:** 20, 100 steps each, eval every 20 steps
- **HP space:** 6 dimensions (lr, rank, alpha_ratio, dropout, target_mlp, weight_decay)
- **Packing:** All 20 adapters packed on single frozen base model
- **Peak VRAM:** 9,744 MB (budget: 13,000 MB)
- **Wall-clock:** 89.1 min (Phase 2 sequential: 92 min)

## Pre-validation

| Test | Result |
|------|--------|
| Gradient isolation | PASSED (max_diff=0.0, 224 params checked) |
| VRAM profiling | 8 configs: 313-724 MB per adapter |
| Overfit smoke test | PASSED (3 trials, loss decreased) |

## Results

| Rank | Trial | eval_loss | rank | mlp | lr | alpha_ratio | dropout | weight_decay |
|------|-------|-----------|------|-----|------|-------------|---------|-------------|
| 1 | 17 | 0.1495 | 32 | T | 1.29e-4 | 1 | 0.05 | 3.88e-3 |
| 2 | 16 | 0.1578 | 8 | T | 2.41e-4 | 2 | 0.15 | 1.86e-3 |
| 3 | 18 | 0.1677 | 4 | T | 5.95e-4 | 1 | 0.00 | 1.05e-6 |
| 4 | 13 | 0.1700 | 16 | F | 4.00e-5 | 2 | 0.15 | 9.07e-5 |
| 5 | 2 | 0.1707 | 16 | F | 3.84e-5 | 2 | 0.00 | 1.82e-6 |

## Gate validation (vs Phase 2)

- **Shared configs:** 1/2 required → SOFT FAIL
- **Phase 2 top-3:** {(4,True), (4,False)} — all rank=4
- **Stage 1 top-3:** {(32,True), (8,True), (4,True)} — mixed ranks
- **Analysis:** Stage 1 discovered higher-rank configs outperform rank=4. Phase 2's (4,False) winner at Stage 1 rank #7 (loss=0.177 vs 0.174). Loss distributions overlap.

## Key findings

1. **Higher ranks competitive:** rank=32+MLP achieves best loss (0.150), beating Phase 2's rank=4 (0.159)
2. **All top-3 use MLP targets:** target_mlp=True dominates
3. **Packing works:** 20 adapters on one model, gradient isolation verified, VRAM within budget
4. **Speedup marginal at 1.03x:** Round-robin overhead cancels model-load savings for same-size batch
