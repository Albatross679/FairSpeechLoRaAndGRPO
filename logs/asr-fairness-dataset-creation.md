---
fileClass: Log
name: ASR Fairness Dataset and Collate Function
description: Created ASRFairnessDataset PyTorch Dataset class and collate_fn for demographic-aware ASR data loading
created: 2026-04-05
updated: 2026-04-05
tags: [pytorch, dataset, dataloader, fairness, demographics, soundfile]
aliases: []
status: complete
subtype: data-prep
---

# ASR Fairness Dataset and Collate Function

Created `scripts/training/data_loader.py` with `ASRFairnessDataset` class and `collate_fn` function.

## Changes

- Created `scripts/training/__init__.py` as package marker with public imports
- Created `scripts/training/data_loader.py` with:
  - `ASRFairnessDataset(Dataset)`: loads audio via `soundfile.read()`, returns dicts with audio tensor, transcript, demographic group, and utterance ID
  - `collate_fn(batch)`: pads variable-length audio to max-in-batch length, returns batched tensors with lengths
- Audio loading uses `soundfile.read()` (not `torchaudio.load()` which is broken in this environment)
- Missing demographic labels handled as empty strings per D-08
- Stereo-to-mono conversion handled for robustness
- Resampling via `torchaudio.functional.resample()` when source sample rate differs from target

## Key Design Decisions

- `demographic_axis` parameter supports both "ethnicity" (Fair-Speech) and "accent" (Common Voice)
- Padding is per-batch (not global max) to mitigate OOM risk (T-01.0-05)
- Column validation at init time to fail fast on malformed manifests
