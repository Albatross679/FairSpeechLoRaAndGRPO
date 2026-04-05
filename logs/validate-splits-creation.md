---
fileClass: Log
name: validate-splits-creation
description: Created validate_splits.py for Fair-Speech train/eval split validation
created: 2026-04-05
updated: 2026-04-05
tags: [data-prep, validation, fair-speech]
aliases: []
status: complete
subtype: data-prep
---

# Created validate_splits.py

Created `scripts/validate_splits.py` implementing comprehensive validation for Fair-Speech train/eval splits.

## Seven Validation Checks

1. **Demographic label completeness** -- flags >5% missing ethnicity labels as failures
2. **MIN_GROUP_SIZE** -- checks all demographic groups meet threshold (50) in eval split
3. **Speaker/utterance leakage** -- verifies zero overlap in speaker_id and utterance_id
4. **Audio file existence** -- samples 100 paths per split, checks os.path.isfile
5. **Text normalization consistency** -- detects empty normalized text where raw exists, validates character set
6. **CV manifest existence** -- verifies cv_test_manifest.csv is present
7. **Distribution drift** -- flags ethnicity groups differing >5pp between train and eval

## Features

- Issue collection pattern matching existing validate_*.py scripts
- Separate issues (critical, cause exit 1) and warnings (informational)
- `--plots` flag for matplotlib diagnostic charts (ethnicity dist, age dist, group sizes)
- Argparse CLI with `--train`, `--eval`, `--cv-manifest`, `--plots`, `--output-dir`
- Summary reporting with train/eval stats and per-axis group counts
