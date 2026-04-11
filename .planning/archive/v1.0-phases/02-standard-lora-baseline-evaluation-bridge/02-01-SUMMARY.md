---
phase: 02-standard-lora-baseline-evaluation-bridge
plan: 01
status: complete
started: 2026-04-06
completed: 2026-04-07
---

# Plan 02-01 Summary: Standard LoRA Training Pipeline

## What was delivered

Trained a standard LoRA SFT adapter (lambda=0, no fairness reward) for Qwen3-ASR-1.7B following the ML prototyping 4-phase process.

## Key results

- **HP Sweep (Phase 2):** 20 Optuna trials on 500-sample subset. Top-3 loss spread 9.1% (gate passed). All top configs converged on rank=4, lr~1e-3.
- **Validation (Phase 3):** Top 3 configs validated on 1000-sample subset, 300 steps each. Ranking flipped — config with attention-only LoRA (no MLP) won at medium scale.
- **Final Training (Phase 4):** 500 steps on 2000-sample subset. Loss: 2.31 → 0.27. Peak VRAM: 9.97 GB (within 14 GB budget).

## Locked HP configuration

| Param | Value |
|-------|-------|
| rank | 4 |
| alpha | 4 (1x ratio) |
| dropout | 0.0 |
| target_mlp | false |
| lr | 8.19e-4 |
| weight_decay | 1.33e-5 |

## Artifacts

| Path | Description |
|------|-------------|
| `outputs/hp-sweep/best_params.json` | Best HP config from sweep |
| `outputs/hp-sweep/top3_configs.json` | Top 3 configs for validation |
| `outputs/hp-sweep/all_trials.csv` | All 20 trial results |
| `outputs/standard-lora/locked_config.json` | Locked config after validation |
| `outputs/standard-lora/adapter/adapter_model.safetensors` | Trained adapter (8.6 MB) |
| `outputs/standard-lora/adapter/adapter_config.json` | Adapter config |
| `outputs/standard-lora/training_config.json` | Full training config + metrics |
| `outputs/standard-lora/train_subset.csv` | Training data subset |
| `outputs/standard-lora/eval_subset.csv` | Evaluation data subset |
| `scripts/training/train_standard_lora.py` | Training script (validate + train modes) |

## Deviations

- **Ranking flip:** Sweep top-1 (trial 15, MLP LoRA) was beaten by trial 12 (attention-only) at validation scale. Used validation winner per plan.
- **No speaker-disjoint split:** Manifests lack `speaker_id` column. Random split used with warning.
- **500 steps instead of 1344:** Reduced from 3-epoch auto-compute to 1-epoch for prototyping speed. Loss was already at 0.11 by step 250.

## Requirements satisfied

- **BASE-01:** Standard LoRA adapter trained and saved ✓
