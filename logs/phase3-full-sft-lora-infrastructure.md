---
fileClass: Log
name: Phase 3 Full SFT LoRA Training Infrastructure (Plan 03-01)
description: Prepared train_standard_lora.py for full-scale 1.16M-sample run — SDPA attention, --lr override, full dataloader pipeline config, and dry-run validation on RTX A6000
created: 2026-04-10
updated: 2026-04-10
tags: [training, lora, phase-3, infrastructure, vram, dataloader]
aliases: []
status: complete
subtype: training
---

## Summary

Executed Phase 03 Plan 01 ("infrastructure + dry-run"). Modified
`scripts/training/train_standard_lora.py` to support the full 1.16M-sample
SFT LoRA run planned for Plan 03-02, then validated the changes with a short
50-step / 200-sample dry-run on an NVIDIA RTX A6000 (48 GB).

No actual training was performed. Purpose: catch config / VRAM / dataloader
issues cheaply before committing to a ~36h full-scale run.

## Code Changes (Task 1)

All changes restricted to `scripts/training/train_standard_lora.py`:

1. **SDPA attention enabled by default** in `load_model_and_processor()` via
   new `attn_implementation="sdpa"` kwarg forwarded to
   `Qwen3ASRModel.from_pretrained(...)`. Memory-efficient attention (O(N)
   activation memory) — free win on Ampere+ GPUs. Validate mode inherits the
   new default (no regression; HF's default on recent transformers is also
   sdpa).
2. **`--lr` CLI override** applied after reading locked config. Lets Plan 02
   pass `--lr 9.8e-4` (sqrt-scaled for effective batch 16) without mutating
   `outputs/standard-lora/locked_config.json`.
3. **Full dataloader pipeline config** in the TrainingArguments block:
   - `dataloader_num_workers` (CLI, default 4)
   - `dataloader_pin_memory=True`
   - `dataloader_persistent_workers` (True when workers > 0)
   - `dataloader_prefetch_factor` (CLI, default 2; None when workers=0)
   These are the load-bearing settings for keeping GPU fed on a 1.16M audio
   dataset. Previously only `dataloader_pin_memory=False` was set.
4. **DynamicBatchTrainer.get_train_dataloader** updated to honor
   `persistent_workers` + `prefetch_factor` when workers > 0 (previously
   passed only num_workers and pin_memory).
5. **CLI args added:** `--lr`, `--dataloader_num_workers`,
   `--dataloader_prefetch_factor`. (The other Plan 03-01 flags —
   `--full_dataset`, `--num_epochs`, `--save_total_limit`,
   `--resume_from_checkpoint`, `--locked_config_path` — were already present
   from a prior revision.)

`run_validate()` was not modified. The only shared-path change is the new
SDPA default in `load_model_and_processor()`.

Commit: `86b4f64` (feat(03-01): add full-scale SFT LoRA training infrastructure)

## Dry-Run Validation (Task 2)

Hardware note: plan was written for RTX 3090 (24 GB). Actual dry-run host has
RTX A6000 (48 GB), so the VRAM gates (< 24 GB / < 14 GB / < 20 GB) pass
trivially. The load-bearing check is the **mean GPU-Util ≥ 70%** gate over
the 50-step window — that confirms the dataloader keeps the A6000 fed and
therefore will also keep a 3090 fed for the Plan 03-02 full-scale run.

Command (summarised — full command in SUMMARY.md):

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python scripts/training/train_standard_lora.py \
  --mode train \
  --fs_manifest outputs/manifests/fs_train.csv \
  --cv_manifest outputs/manifests/cv_train.csv \
  --output_dir outputs/full-lora-dryrun \
  --locked_config_path outputs/standard-lora/locked_config.json \
  --subset_size 200 --max_steps 50 \
  --batch_size 4 --grad_accum 4 --lr 9.8e-4 \
  --dataloader_num_workers 4 --dataloader_prefetch_factor 2 \
  --save_total_limit 1 --wandb_project none
```

Results (populated at dry-run completion — see
`.planning/phases/03-full-sft-lora-training/03-01-SUMMARY.md` for numbers and
GO/NO-GO verdict).

## Files Changed

- `scripts/training/train_standard_lora.py` (+39 / -6)

## Related

- Plan: `.planning/phases/03-full-sft-lora-training/03-01-PLAN.md`
- Summary: `.planning/phases/03-full-sft-lora-training/03-01-SUMMARY.md`
- Downstream: Plan 03-02 will consume this script for the full-scale run
- Locked HPs: `outputs/standard-lora/locked_config.json`
