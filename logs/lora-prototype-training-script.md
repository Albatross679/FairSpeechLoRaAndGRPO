---
fileClass: Log
name: LoRA Prototype Training Script
description: Created data collator and LoRA prototype training script for Qwen3-ASR-1.7B fine-tuning
created: 2026-04-05
updated: 2026-04-05
tags:
  - lora
  - training
  - qwen3-asr
  - phase-1.1
aliases: []
status: complete
subtype: feature
---

## Summary

Created two new modules for the LoRA prototype training pipeline:

1. **`scripts/training/data_collator.py`** - `DataCollatorForQwen3ASR` class that converts raw audio + transcript pairs into Qwen3-ASR chat-format training batches with prefix masking (labels=-100 for non-transcript tokens).

2. **`scripts/training/lora_prototype.py`** - End-to-end LoRA prototype training script that loads Qwen3-ASR-1.7B via qwen-asr wrapper, patches forward for Trainer compatibility, freezes the audio encoder, enables gradient checkpointing (use_reentrant=False), applies LoRA r=16/alpha=32 to decoder self-attention only, creates a 200-sample stratified subset, and trains via HuggingFace Trainer.

## Key Implementation Details

- Model loading uses `Qwen3ASRModel.from_pretrained()` then extracts `.model` (Pattern 1)
- Forward patch delegates to `thinker.forward()` for loss computation (Pattern 2)
- LoRA targets decoder attention only via regex: `thinker\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)`
- MLP modules deliberately excluded per D-01 attention-only decision
- VRAM profiled at 5 checkpoints: model load, PEFT wrap, first forward, first backward, end of training
- Subset creation: 100 Fair-Speech (ethnicity-stratified) + 100 Common Voice (accent-stratified)

## Files Changed

- `scripts/training/data_collator.py` (new, 268 lines)
- `scripts/training/lora_prototype.py` (new, 553 lines)
- `scripts/training/__init__.py` (updated exports)
