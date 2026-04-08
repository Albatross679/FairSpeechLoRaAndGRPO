---
fileClass: Experiment
name: Phase 2.1 Stage 2 - PLoRA + ASHA 100-Trial Sweep
description: 100-trial packed LoRA HP sweep with HyperbandPruner ASHA pruning on Qwen3-ASR-1.7B
created: 2026-04-08
updated: 2026-04-09
tags: [lora, hp-sweep, asha, plora, pruning, qwen3-asr]
aliases: [stage2-sweep]
status: complete
algorithm: TPE + HyperbandPruner
task: LoRA HP optimization
hardware: T4 15GB
---

# Phase 2.1 Stage 2: PLoRA + ASHA 100-Trial Sweep

## Setup

- Model: Qwen3-ASR-1.7B (bf16)
- Technique: PLoRA packed adapters + ASHA pruning via HyperbandPruner
- HP space: 6 dimensions (lr, rank, alpha_ratio, dropout, target_mlp, weight_decay)
- Trials: 100 (asked in batches of 18-29 based on VRAM)
- Steps/trial: 100
- Rung schedule: {25, 50, 75} with reduction_factor=3
- Subset: 500 samples (468 train, 49 eval)
- VRAM budget: 13 GB (peak observed: 12,153 MB)

## Results

| Metric | Value |
|--------|-------|
| Total trials | 100 |
| Wall-clock time | 358.0 min (5.97 hours) |
| Best eval_loss | 0.1424 (trial 77) |
| Best rank | 4 |
| Best target_mlp | True |
| Best lr | 3.47e-4 |
| Gate vs Stage 1 | PASSED (2 shared) |

## Top-3 Configs

| Rank | Trial | eval_loss | LoRA rank | target_mlp | lr |
|------|-------|-----------|-----------|------------|------|
| 1 | 77 | 0.1424 | 4 | True | 3.47e-4 |
| 2 | 16 | 0.1441 | 8 | True | 2.41e-4 |
| 3 | 85 | 0.1442 | 4 | True | 3.23e-4 |

## Pruning Behavior

- ASHA pruned poorly-performing trials at each rung step
- Batch 1 (25 adapters): 9 pruned at step 25, 9 at step 75
- Batch 4 (29 adapters): 6 pruned at step 25, 6 at step 75
- High-loss configs (lr too low, wrong rank) killed early

## Comparison vs Phase 2

- Phase 2 sweep best: 0.159 (rank=4, alpha_ratio=2, target_mlp=true)
- Stage 2 best: 0.1424 (10.5% improvement)
- Speedup: 1.28x for 100 trials vs 460 min sequential equivalent
