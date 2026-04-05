---
fileClass: Log
name: DemographicStratifiedSampler and create_dataloader factory
description: Added demographic-stratified batch sampler with oversampling and DataLoader factory function
created: 2026-04-05
updated: 2026-04-05
tags: [data-loader, pytorch, fairness, sampling, oversampling]
aliases: []
status: complete
subtype: feature
---

# DemographicStratifiedSampler and create_dataloader factory

Added to `scripts/training/data_loader.py`:

- **DemographicStratifiedSampler(Sampler)**: Oversamples small demographic groups to MIN_GROUP_SIZE=50 floor (D-09). Groups with missing labels are tracked as unlabeled_indices and included in the pool (contributing to R_acc per D-08) but not used for stratification. Uses numpy RandomState with fixed seed for reproducibility. Supports optional num_batches to control effective epoch length (per Pitfall 6 from RESEARCH.md).

- **create_dataloader factory**: One-call setup for both training (stratified=True with DemographicStratifiedSampler) and evaluation (stratified=False, sequential). Accepts manifest_csv path and demographic_axis ("ethnicity" for Fair-Speech, "accent" for Common Voice).

- **__main__ smoke test**: Quick CLI test to verify loading from a manifest CSV with a single batch.

File is now 290 lines with all four public exports: ASRFairnessDataset, DemographicStratifiedSampler, collate_fn, create_dataloader.
