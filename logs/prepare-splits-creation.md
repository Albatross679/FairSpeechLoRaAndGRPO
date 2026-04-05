---
fileClass: Log
name: prepare-splits-creation
description: Created prepare_splits.py for Fair-Speech speaker-level stratified train/eval splitting
created: 2026-04-05
updated: 2026-04-05
tags: [data-prep, fair-speech, splitting]
aliases: []
status: complete
subtype: data-prep
---

# Created prepare_splits.py

Created `scripts/prepare_splits.py` implementing speaker-level stratified train/eval splitting for the Fair-Speech dataset.

## Key Features

- **Speaker ID discovery**: Reads `fairspeech_metadata.tsv` at runtime to discover the speaker/participant column (~593 unique values)
- **StratifiedGroupKFold**: Uses sklearn's `StratifiedGroupKFold(n_splits=5)` to split at speaker level while preserving ethnicity distribution
- **Zero-leakage assertion**: Asserts no speaker overlap between train and eval splits
- **Distribution stats**: Prints per-axis (ethnicity, age, first_language, gender) train vs eval proportions
- **CV manifest check**: Verifies `cv_test_manifest.csv` exists and prints accent distribution
- **Argparse CLI**: Supports `--fs-manifest`, `--cv-manifest`, `--output-dir`, `--seed` arguments
- **Deterministic**: Fixed `SEED=42` for reproducibility (D-03)

## Constants

- `FS_DIR`, `OUTPUT_DIR`, `CV_DIR` matching existing pipeline paths
- `SEED = 42`, `N_SPLITS = 5`, `MIN_GROUP_SIZE = 50`

## Outputs

- `fs_train.csv` (~80% of Fair-Speech utterances)
- `fs_eval.csv` (~20% of Fair-Speech utterances)
- Both include all demographic columns plus `speaker_id`
