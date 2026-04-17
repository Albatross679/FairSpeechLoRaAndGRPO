---
fileClass: Log
name: Stage F energy-based VAD (T8)
description: Implemented energy-based VAD preprocessing in scripts/head_surgery/energy_vad.py and added two unit tests to tests/test_head_surgery.py.
created: 2026-04-17
updated: 2026-04-17
tags: [head-surgery, vad, stage-f, T8]
aliases: []
status: complete
subtype: feature
---

## What changed

- `scripts/head_surgery/energy_vad.py`: replaced docstring-only stub with full implementation.
  - `_frame()`: sliding-window framing (20 ms frame / 10 ms hop).
  - `filter_silence()`: RMS-based silence detection; drops runs of below-threshold frames ≥ `min_silence_ms`.
  - `run_vad_arm()`: sweeps three `db_floor` values × severity levels from a silence-injection manifest, re-infers with Whisper, writes `outputs/head_surgery/vad_scores.csv`.
  - `_cli()`: argparse entry point for `--silence-manifest`, `--batch-size`, `--device`.
- `tests/test_head_surgery.py`: added `test_filter_silence_drops_zero_region` and `test_filter_silence_preserves_all_speech` using synthetic numpy arrays (no GPU, no real audio).

## Test results

`pytest tests/test_head_surgery.py -v`: **15 passed, 1 skipped** (skip is pre-existing Stage A artifact gate).

## Commit

`3fd1f1a` — feat(head_surgery): Stage F energy-based VAD (T8, code)

## Notes

`run_vad_arm` / `_cli` are out of scope for this dispatch — they require a silence-injection perturbation manifest that does not yet exist. They are included for completeness but not exercised by CI.
