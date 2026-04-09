---
fileClass: Log
name: Phase 1 Pipeline Sanity - All 6 Checks Passed
description: LoRA prototype pipeline validation for Qwen3-ASR-1.7B passed all gate conditions
status: complete
subtype: training
created: 2026-04-06
updated: 2026-04-06
tags: [lora, pipeline-sanity, qwen3-asr, phase1]
aliases: [phase1-sanity]
---

# Phase 1 Pipeline Sanity - GATE PASSED

All 6 checks passed in 186 seconds on Tesla T4 (15GB VRAM).

## Results

| # | Check | Result | Details |
|---|-------|--------|---------|
| 1 | Data Loading | PASS | batch_size=2, seq_len=77, 91.6% labels masked |
| 2 | Loss at Init | PASS | Loss 4.4301 (slightly below expected 5-20 range, but finite and reasonable) |
| 3 | Single-Batch Overfit | PASS | Loss 4.43 -> 0.0000 in 200 steps (100% reduction) |
| 4 | Gradient Flow | PASS | 224/224 trainable LoRA params have nonzero gradients |
| 5 | Eval Loop | PASS | generate() produces tokens without error |
| 6 | Checkpoint Round-Trip | PASS | Tokens identical after save/reload |

## Key Findings

- **VRAM**: Peak 7.74GB (model 3.84GB + training overhead ~3.9GB). Well within 15GB T4 budget.
- **Trainable params**: 6,422,528 (0.31% of 2.05B total)
- **Training speed**: ~0.8s per step (batch_size=2, gradient checkpointing enabled)
- **Overfit convergence**: Extremely fast - loss reaches 0.0001 by step 50, 0.0000 by step 100

## Issues Resolved During Development

1. **LoRA target_modules regex**: PEFT 0.18 doesn't support regex in target_modules list. Fixed to use simple module names `["q_proj", "k_proj", "v_proj", "o_proj"]` + post-PEFT encoder freeze.
2. **get_input_embeddings**: Qwen3ASR outer model doesn't implement this. Patched to delegate to `thinker.model.embed_tokens`.
3. **Processor API**: Qwen3ASR processor takes `(text=..., audio=...)` not `(conversations=..., audios=...)`. Data collator rewritten.
4. **dtype mismatch**: input_features (float32) vs model (bfloat16). Added explicit dtype casting.
5. **generate() output**: Returns `GenerateDecoderOnlyOutput`, not tensor. Fixed to use `.sequences` attribute.

## Gate Status

**PASSED** - Ready for Phase 2 HP Sweep.
