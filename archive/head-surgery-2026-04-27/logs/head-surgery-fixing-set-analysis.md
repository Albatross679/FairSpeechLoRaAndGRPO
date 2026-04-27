---
fileClass: Log
name: head-surgery-fixing-set-analysis
description: Post-hoc fixing-set analysis on sweep.csv — which (L, h) pairs cover which of the ~45 affected utterances, and what is the minimum surgical set?
status: complete
subtype: evaluation
created: 2026-04-18
updated: 2026-04-18
tags: [head-surgery, fixing-set, post-hoc, no-gpu]
aliases: []
---

# Fixing-set analysis

## Inputs
- `outputs/head_surgery/sweep.csv` (309,760 rows)
- `outputs/head_surgery/baseline_predictions.csv` (484 rows)
- `outputs/head_surgery/head_scores.csv` (640 rows)

## Method
Three-filter coverage matrix (helps, no global harm, regression_ok) +
greedy and ILP min-set-cover. Implementation:
`scripts/head_surgery/fixing_set_analysis.py`.

## Results
- Affected utterances: **45**
- Valid heads (after all three filters): **115**
- Greedy cover size: **8**
- ILP optimum size: **8**
- Unhelpable utterances: **30** — `["common_voice_en_17417536", "common_voice_en_17758246", "common_voice_en_18545942", "common_voice_en_18976651", "common_voice_en_18976653", (... and 25 more — see minimum_surgical_set.json)]`
- Runtime: **66.22 s**

## Artifacts
- `outputs/head_surgery/fixing_set_per_utterance.csv`
- `outputs/head_surgery/coverage_matrix.npz`
- `outputs/head_surgery/minimum_surgical_set.json`

## Sanity checks
- `n_affected = 45`: matches §8's quote of 45/484 utterances with ≥1 hallucination. PASS.
- `n_valid_heads = 115`: within the ≤640 hard bound; the "likely ≤50" heuristic was conservative — more heads pass all three filters than anticipated. PASS.
- `len(ilp) = 8 ≤ len(greedy) = 8`: ILP and greedy converged to the same cover size; ILP is optimal. PASS.
- L=0 h=5 (catastrophic keystone, 101% insertion rate) absent from both greedy and ILP: filter (ii) correctly eliminates it. PASS.
- L=20 h=11 (bootstrap-significant head from §7.4) appears in both greedy and ILP: it passes all three filters and contributes 3 newly covered utterances. Expected.

## Interpretation note
The min-cover is a *hypothesis* about multi-head masking — it assumes the
effects of masking multiple heads simultaneously are at least as good as
the union of single-head effects. That assumption is unverified; the
sweep is single-head. Validating requires a separate GPU run that masks
the candidate set together.
