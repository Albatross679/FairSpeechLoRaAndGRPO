---
fileClass: Log
name: head-surgery-stage-d-scoring
description: Implemented Stage D per-head scoring, paired bootstrap p-value, and regression guard in score_heads.py; added two bootstrap unit tests.
created: 2026-04-17
updated: 2026-04-17
tags: [head-surgery, stage-d, bootstrap, regression-guard, scoring]
aliases: []
status: complete
subtype: feature
---

## Summary

Replaced the docstring-only stub `scripts/head_surgery/score_heads.py` with the full Stage D implementation:

- `paired_bootstrap_delta_p`: one-sided bootstrap test for H0: masked insertion rate >= baseline. Returns fraction of bootstrap resamples where Delta <= 0.
- `compute_head_scores`: aggregates per-(L, h) insertion metrics (total, repetition, syntactic, content) from `sweep.csv` vs `baseline_predictions.csv`, then calls bootstrap for each head.
- `compute_regression_guard`: runs Whisper-large-v3 with each top-K head masked on a non-Indian manifest CSV; enforces Gate G5 cross-check against `t7_non_indian_baseline_wer.json` (prints SKIP when missing).
- `write_top_k`: filters to regression-safe heads and writes `top_k_heads.csv`.

Two new unit tests in `tests/test_head_surgery.py`:
- `test_paired_bootstrap_reports_significance_when_effect_large`: n=200 Poisson-generated counts with strong reduction; expects p < 0.05.
- `test_paired_bootstrap_reports_null_when_effect_zero`: identical base/masked counts; expects p > 0.5.

## Test result

13 passed, 1 skipped (Stage-A artifact guard from Task 10). Commit: e389d16.
