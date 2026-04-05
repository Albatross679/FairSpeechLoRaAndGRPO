---
fileClass: Log
name: LoRA Prototype Validation Script
description: Created validate_lora_prototype.py with 4 validation checks for LoRA prototype training results
created: 2026-04-05
updated: 2026-04-05
tags: [lora, validation, peft, qwen3-asr, vram, wer]
aliases: []
status: complete
subtype: feature
---

# LoRA Prototype Validation Script

Created `scripts/training/validate_lora_prototype.py` (773 lines) with 4 end-to-end validation checks for the LoRA prototype training results.

## Checks Implemented

1. **VRAM Usage** - Loads adapter model, runs forward+backward pass, measures peak VRAM via `torch.cuda.max_memory_allocated()`. Passes if peak < 14GB.
2. **Transcription Quality** - Transcribes eval samples, checks for empty hypotheses, chatty patterns (11 patterns), and overall WER < 80% using jiwer.
3. **Adapter Round-Trip** - Saves adapter to temp dir, reloads, generates on same input with `do_sample=False`, compares output tokens with `torch.equal()`.
4. **Loss Trend** - Reads `trainer_state.json`, compares first-quarter vs last-quarter mean loss. Passes if loss is decreasing.

## Design Decisions

- Follows `validate_test_run.py` pattern: issue collection list, summary reporting, `sys.exit(1 if issues else 0)`
- VRAM check loads model fresh and runs a real forward+backward pass rather than parsing log output (more reliable)
- Round-trip test uses deterministic generation (`do_sample=False`, `torch.manual_seed(42)`) for exact token comparison
- Model loaded once for transcription quality + round-trip checks to minimize GPU memory overhead
- Adapter file size sanity check (> 1KB, < 100MB) included in round-trip test
