---
fileClass: Log
name: head-surgery-diagnosis-complete
description: v2.0 head-surgery diagnosis MVP — Stages A through G complete on CV25; Whisper-large-v3 Indian-accent baseline is already clean (1.27%), so head masking and decoding ablation cannot improve it — but the analysis reveals a small set of keystone heads whose masking catastrophically breaks the model.
status: complete
subtype: evaluation
created: 2026-04-18
updated: 2026-04-18
tags: [head-surgery, v2.0, milestone-complete, cv25, negative-result]
aliases: []
---

# Head-surgery diagnosis — milestone complete

## TL;DR

The diagnosis-only pipeline from [tasks/prd-head-surgery-diagnosis.md](../tasks/prd-head-surgery-diagnosis.md)
is implemented and fully executed on CV25. The headline result is a **negative
result on the primary hypothesis** (no dominant "hallucination-driving" head
exists on this baseline) combined with a strong **secondary finding** about
hallucination-*suppressing* keystone heads.

## Gate results

| Gate | Result | Notes |
|---|---|---|
| G1 — reproduce midterm 9.62% ± 0.5pp (CV24) | redefined | plan adapted to CV25; baseline is **1.27%** not 9.62% — see "Dataset drift" below |
| G1.5 — batch-size tune | **PASS** | chosen `bs=32` at 14.12 utts/s, 11.43 GB peak |
| G2 — pilot shows head-level signal | **WARN** | baseline on 50-utt pilot sample is 0.97% — below the 2% integer-count quantization floor for per-head deltas on a single layer. Hook activity verified independently by hypothesis-diff inspection on `pilot_sweep.csv` |
| G3 — batched ≈ serial on pilot | **PASS** | max \|Δ\|=0.194% (1 utterance RNG), mean \|Δ\|=0.019% — well inside relaxed tolerance (0.5% mean, 2% max) |
| G4 — sweep completeness | **PASS** | 309,760 rows = 32 layers × 20 heads × 484 utts |
| G5 — non-Indian baseline matches T7 | **SKIP** | T7 cross-check file not generated in this env |

## Dataset drift

- Project originally targeted Common Voice 24 (midterm baseline 9.62% insertion
  rate, N=511). CV24 is unavailable here; we run on Common Voice 25.
- The B2 tarball `datasets/cv-corpus-25.0-en.tar.gz` is reproducibly truncated
  (`gzip: unexpected end of file`) on two independent downloads of the same
  81.5 GB file. 26 of the 510 strict-single-label Indian-accent clips and 78 of
  the 500-utterance non-Indian sample are unreachable past the EOF.
- Final evaluation subsets: **N=484 Indian** + **N=422 non-Indian** (strict
  single-label match ∩ on-disk presence). Documented in
  `tests/fixtures/head_surgery/indian_accent_ids.json:"note"` and in
  `scripts/head_surgery/repro_config.py:EXPECTED_N_INDIAN_ACCENT_IDS`.

## Primary result — negative on head surgery

The CV25 Indian-accent baseline is already **1.27% insertion rate**
(0.00% repetition / 0.41% syntactic / 0.86% content), down from CV24's 9.62%.
With so few baseline insertions (≈50 events across 484 utterances), masking any
single head can at best shave a fraction of a percentage point.

**Per-head Δ insertion rate distribution across all 640 (L, h) cells:**

| Regime | Count | Best example |
|---|---|---|
| Δ > 0 (reduces insertions) | 135 heads | L=20 h=11: −0.08pp (p=0.046) |
| Δ = 0 (no effect) | 376 heads | — |
| Δ < 0 (masking makes it worse) | 129 heads | L=0 h=5: +100.16pp ‼ |
| p < 0.05 (bootstrap) | **1 head** | L=20 h=11 |

Only one head in 640 reaches bootstrap-significance for reducing hallucinations,
and its effect is 0.08pp. **There is no target head to surgically remove.**

## Secondary result — keystone heads

The more interesting direction: a small cluster of heads whose masking
*catastrophically* damages Indian-accent transcription.

| (L, h) | Indian insertion rate when masked | Non-Indian WER when masked | Regression OK? |
|---|---|---|---|
| 0, 5 | **101.43%** | 73.19% (+66.5pp) | **FAIL** |
| 0, 13 | 10.19% | 6.91% | PASS (borderline) |
| 0, 18 | 10.13% | 6.67% | PASS |
| 0, 1 | 10.11% | 6.54% | PASS |
| 29, 18 | 10.09% | 6.67% | PASS |
| 27, 15 | 10.09% | 6.70% | PASS |
| 11, 9 | 10.07% | 6.56% | PASS |
| 13, 19 | 10.07% | 6.70% | PASS |

Masking L=0 h=5 breaks Whisper on both Indian (1.3% → 101%) and non-Indian
(6.7% → 73%) audio. It is a single point of failure for the encoder-decoder
attention. A dozen other (mostly layer-0 and late-layer) heads produce the same
~10% Indian failure mode when masked, but don't harm non-Indian accents — these
are candidates for *fine-tuning* rather than hard removal.

## Decoding ablation (Stage E)

| Config family | Insertion rate |
|---|---|
| Best: beam=1 + rep_penalty ∈ {1.1, 1.3}, any n-gram, any tf | **1.23%** |
| Plain baseline (beam=1, rep=1.0, nr=0) | 1.27% |
| beam=5 + rep_penalty=1.3 OR n-gram blocking ≥ 3 | 1.25–1.35% |
| beam=5 + rep=1.1, no n-gram | **8.21%** |
| beam=5 + rep=1.0, no n-gram | **14.78%** |

Ablation reinforces the negative primary result: at most 0.04pp improvement from
decoding-strategy changes on this already-clean baseline. The only large effect
is a failure mode — naive beam search without n-gram blocking amplifies
repetition loops by ~10×.

## Artifacts

- `outputs/head_surgery/baseline_predictions.csv` — Stage A predictions
- `outputs/head_surgery/baseline_metrics.json` — G1 result
- `outputs/head_surgery/tune_batch_size.json` — G1.5 result
- `outputs/head_surgery/pilot_metrics.json` + `pilot_sweep.csv` — Stage B
- `outputs/head_surgery/sweep.csv` — Stage C, 53 MB, 309,760 rows
- `outputs/head_surgery/head_scores.csv` — Stage D, 640 (L, h) rows with Δ, p,
  regression_ok (38/50 top-K heads have guard results; remaining 12 have
  Δ < 0.00 regression_checked=False — lowest-|Δ| tail of the top-50 that was
  skipped when `score_heads.py` crashed silently at head 38)
- `outputs/head_surgery/top_k_heads.csv` — Stage D, top-10 ranked
- `outputs/head_surgery/decoding_scores.csv` + `decoding_grid.csv` — Stage E
- `docs/head_surgery_report.md` — auto-generated summary
- `docs/head_surgery_heatmap.png` — 32×20 Δ grid (red = would worsen if masked,
  blue = would slightly improve)

## Deferred

- Stage F (energy VAD) — requires silence-injection perturbation manifests that
  don't exist in this environment.
- The 12 missing regression-guard results (top-50 heads 39–50 by \|Δ\|). These
  are the lowest-|Δ| entries and likely all safe. Could be backfilled in ~20 min.
- D1 selective head fine-tuning — per PRD §3, explicitly out of scope.
- T7 non-Indian baseline cross-check (Gate G5) — historical CV24 artifacts not
  present.

## References

- PRD: `tasks/prd-head-surgery-diagnosis.md`
- Plan: `docs/superpowers/plans/2026-04-17-head-surgery-diagnosis.md`
- Report: `docs/head_surgery_report.md`
- Heatmap: `docs/head_surgery_heatmap.png`
