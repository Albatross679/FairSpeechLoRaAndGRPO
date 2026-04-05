---
fileClass: Log
name: ASRFairnessDataset and collate_fn creation
description: Created PyTorch Dataset class for ASR fairness training with soundfile-based audio loading and variable-length collation
created: 2026-04-05
updated: 2026-04-05
tags: [data-loader, pytorch, fairness, audio, soundfile]
aliases: []
status: complete
subtype: feature
---

# ASRFairnessDataset and collate_fn creation

Created `scripts/training/data_loader.py` with:

- **ASRFairnessDataset(Dataset)**: Loads audio via `soundfile.read()` (not torchaudio.load which is broken), returns dicts with audio tensor, transcript, demographic_group, and utterance_id. Supports both Fair-Speech (ethnicity axis) and Common Voice (accent axis). Missing demographic labels handled with `fillna("")` per D-08.

- **collate_fn**: Pads variable-length audio to max-in-batch length with zeros. Returns batched tensor with audio_lengths for downstream processing.

- **scripts/training/__init__.py**: Package marker with imports of all public symbols.

Key design decisions:
- Audio loaded as float32 numpy array then converted to torch tensor
- Resampling via `torchaudio.functional.resample()` when source SR differs from target 16kHz
- Column validation at init time with clear error messages
