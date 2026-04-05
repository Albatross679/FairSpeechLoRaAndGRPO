---
fileClass: Log
name: Demographic Stratified Sampler and DataLoader Factory
description: Created DemographicStratifiedSampler and create_dataloader factory for GRPO fairness-aware training
created: 2026-04-05
updated: 2026-04-05
tags: [pytorch, sampler, dataloader, oversampling, demographics, fairness, grpo]
aliases: []
status: complete
subtype: data-prep
---

# Demographic Stratified Sampler and DataLoader Factory

Extended `scripts/training/data_loader.py` with `DemographicStratifiedSampler` and `create_dataloader` factory.

## Changes

- Added `DemographicStratifiedSampler(Sampler)`:
  - Oversamples small demographic groups to MIN_GROUP_SIZE=50 floor (D-09)
  - Tracks unlabeled indices separately (contribute to R_acc per D-08)
  - Deterministic via `np.random.RandomState(seed)` for reproducibility
  - Effective epoch length configurable via `num_batches` parameter (Pitfall 6)
- Added `create_dataloader()` factory function:
  - Single entry point for both training (stratified=True) and evaluation (stratified=False)
  - Supports Fair-Speech (ethnicity) and Common Voice (accent) via `demographic_axis` parameter
  - Configurable batch_size, num_workers, pin_memory
- Added `__main__` CLI smoke test block for quick verification with real data

## Key Design Decisions

- Oversampling uses integer repetition then truncation (not WeightedRandomSampler) for simplicity
- Groups below MIN_GROUP_SIZE get repeated to reach exactly the floor count
- Unlabeled samples are appended to the pool (not stratified) per D-08
- Epoch length defaults to oversampled pool size but can be fixed via num_batches
