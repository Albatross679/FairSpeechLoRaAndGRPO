---
fileClass: Log
name: Phase 2 Validation Gap Bridging
description: Bridged 4 validation gaps from Phase 2 Nyquist audit — fairness metrics, bootstrap CIs, LibriSpeech eval, speaker-disjoint splits
created: 2026-04-07
updated: 2026-04-07
status: complete
subtype: evaluation
tags: [phase-2, validation, fairness, evaluation, gap-bridging]
aliases: []
---

# Phase 2 Validation Gap Bridging

## What changed

Bridged 4 of 5 validation gaps identified by Nyquist audit of Phase 2:

1. **Fairness metrics at scale**: Reran evaluation on 799 FS + 632 CV utterances (all groups >= 50). Fairness metrics now fully computed — FS max-min ratio 3.43, CV max-min ratio 3.69.

2. **Bootstrap 95% CIs**: Enabled for all 14 demographic groups across FS and CV. Previously skipped with `--skip_bootstrap`.

3. **LibriSpeech evaluation**: Generated `ls_test_clean.csv` manifest (2,620 utterances, 40 speakers) from LibriSpeech test-clean. Overall WER: 1.77%.

4. **Speaker-disjoint splits (CV)**: Added `speaker_id` column (from `client_id`) to Common Voice manifests. Updated `train_standard_lora.py` to use `GroupShuffleSplit` with pseudo-IDs for FS rows.

## Remaining limitation

Fair-Speech has no speaker identifier — `hash_name` is 1:1 per utterance. This is a dataset limitation, not a code gap.

## Files modified

- `scripts/training/generate_manifests.py` — CV speaker_id, LS manifest generation
- `scripts/training/train_standard_lora.py` — Mixed speaker_id handling in disjoint split
- `tests/test_phase02_validation.py` — 79 passing, 1 xfail (was 64 passing, 4 xfail)
- `.planning/phases/02-.../02-VERIFICATION.md` — Updated with bridged results

## Test results

```
79 passed, 1 xfailed in 1.12s
```
