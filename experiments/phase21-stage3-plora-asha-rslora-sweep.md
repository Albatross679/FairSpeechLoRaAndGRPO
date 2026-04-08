---
fileClass: Experiment
name: Phase 2.1 Stage 3 - PLoRA + ASHA + RsLoRA 100-Trial Sweep
description: 100-trial packed LoRA HP sweep with HyperbandPruner and RsLoRA rank-stable scaling on Qwen3-ASR-1.7B
created: 2026-04-08
updated: 2026-04-09
tags: [lora, rslora, hp-sweep, asha, plora, pruning, qwen3-asr]
aliases: [stage3-sweep]
status: complete
algorithm: TPE + HyperbandPruner + RsLoRA
task: LoRA HP optimization with rank-stable scaling
hardware: T4 15GB
---

# Phase 2.1 Stage 3: PLoRA + ASHA + RsLoRA 100-Trial Sweep

## Setup

- Model: Qwen3-ASR-1.7B (bf16)
- Technique: PLoRA packed adapters + ASHA pruning + RsLoRA (alpha/sqrt(r) scaling)
- HP space: 9 dimensions (6 original + lr_scheduler, warmup_ratio, grad_accum_steps)
- Trials: 100 (asked in batches of 19-29)
- Steps/trial: 100
- Rung schedule: {25, 50, 75} with reduction_factor=3
- Subset: 500 samples (468 train, 49 eval)
- VRAM budget: 13 GB (peak observed: 10,293 MB)

## Results

| Metric | Value |
|--------|-------|
| Total trials | 100 |
| Wall-clock time | 386.6 min (6.44 hours) |
| Best eval_loss | 0.1451 (trial 66) |
| Best rank | 8 |
| Best target_mlp | True |
| Best lr | 2.91e-4 |
| Best lr_scheduler | cosine |
| Best warmup_ratio | 0.1 |
| Gate vs Stage 2 | PASSED (2 shared) |

## Top-3 Configs

| Rank | Trial | eval_loss | LoRA rank | target_mlp | lr | scheduler |
|------|-------|-----------|-----------|------------|------|-----------|
| 1 | 66 | 0.1451 | 8 | True | 2.91e-4 | cosine |
| 2 | 16 | 0.1471 | 4 | True | 2.83e-4 | linear |
| 3 | 13 | 0.1488 | 4 | True | 8.58e-4 | linear |

## RsLoRA Impact on Optimal Rank

| Distribution | rank=4 | rank=8 | rank=16 | rank=32 |
|-------------|--------|--------|---------|---------|
| Stage 2 top-10 | 8 | 2 | 0 | 0 |
| Stage 3 top-10 | 2 | 7 | 0 | 1 |

- Optimal rank shifted from 4 (Stage 2) to 8 (Stage 3)
- RsLoRA's alpha/sqrt(r) scaling stabilizes higher-rank adapters
- TPE converged on rank=32 for later batches (cosine sched, accum=4)
- Stage 3 prefers cosine scheduler with warmup (0.1 ratio)

## Key Finding

RsLoRA changes the rank landscape but did not improve absolute eval_loss on this setup. Stage 2 (rank=4, no RsLoRA, loss=0.1424) slightly outperformed Stage 3 (rank=8, RsLoRA, loss=0.1451). The rank shift is genuine and theoretically expected, but the benefit may require more training steps to manifest.
