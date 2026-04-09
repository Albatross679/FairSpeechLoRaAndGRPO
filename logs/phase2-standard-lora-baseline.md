---
fileClass: Log
name: Phase 2 Standard LoRA Baseline Training + Evaluation Bridge
description: Completed HP sweep, validation, final training, and evaluation bridge for standard LoRA SFT adapter
created: 2026-04-07
updated: 2026-04-07
tags: [training, lora, evaluation, fairness, phase-2]
aliases: []
status: complete
subtype: training
---

## Summary

Executed Phase 2: trained a standard LoRA SFT adapter for Qwen3-ASR-1.7B and built the evaluation bridge that connects adapters to fairness metrics.

## Training Pipeline

1. **HP Sweep:** 20 Optuna trials on 500-sample subset, 100 steps/trial. Top-3 spread 9.1%. All top configs: rank=4, lr~1e-3, alpha_ratio 1-2x.
2. **Validation:** Top 3 configs on 1000 samples, 300 steps. Ranking flipped — attention-only (no MLP) LoRA won at medium scale.
3. **Final Training:** 500 steps on 2000 samples. Loss 2.31→0.27. VRAM 9.97/14 GB.

## Locked Config

rank=4, alpha=4, dropout=0, no MLP LoRA, lr=8.19e-4, wd=1.33e-5

## Evaluation Results

- Fair-Speech WER: 2.11% (199 utterances)
- Common Voice WER: 2.90% (500 utterances)
- Per-group fairness metrics need full-scale run (prototype groups < 50 samples)

## Key Finding

LoRA-adapted model needs `language English<asr_text>` prefix in generation prompt. Without it, model immediately outputs `<|im_end|>` due to prefix masking during training.

## Files Changed

- Created: `scripts/training/train_standard_lora.py` (validate + train modes)
- Created: `scripts/training/evaluate_adapter.py` (evaluation bridge)
- Created: `outputs/standard-lora/adapter/` (adapter weights)
- Created: `outputs/hp-sweep/` (sweep results)
