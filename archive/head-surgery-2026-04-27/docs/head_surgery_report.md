# Head-Surgery Diagnosis — Results

Target model: Whisper-large-v3. Evaluation subset: Indian-accent CV25 test utterances (per `scripts/head_surgery/repro_config.py`).

## 0. Summary

### Heads scored

| Category | Count |
|---|---:|
| Total (L, h) cells scored | **640** (32 layers × 20 heads) |
| Top-K reported (§2) | 10 |
| Δ > 0 (masking reduces insertions) | 135 |
| Δ = 0 (no effect) | 376 |
| Δ < 0 (masking worsens insertions) | 129 |
| Bootstrap-significant (p<0.05) | **1** (best: L=20, h=11, −0.08pp) |
| Catastrophic keystone heads (masking breaks model) | ~8 (L=0 h=5: +100pp) |

### Experiments executed (Stages A–G)

| Stage | Gate | Experiment | Artifact |
|---|---|---|---|
| A | G1 | Baseline insertion on CV25 Indian N=484 | `baseline_metrics.json` |
| A.5 | G1.5 | Batch-size tuning (chose bs=32) | `tune_batch_size.json` |
| B | G2 | 50-utt pilot head-mask sweep | `pilot_sweep.csv` |
| C | G3+G4 | Full sweep — 309,760 rows (640 heads × 484 utts) | `sweep.csv` (53 MB) |
| D | — | Scoring + bootstrap + regression guard | `head_scores.csv`, `top_k_heads.csv` |
| E | — | Decoding ablation — 36 configs | `decoding_scores.csv` |
| F | — | Energy VAD under silence injection | `vad_scores.csv` |
| G | — | Aggregate report + heatmap | this file + `head_surgery_heatmap.png` |

### Duration

| Milestone | Date |
|---|---|
| Pivot to head-surgery + domain research | 2026-04-11 |
| PRD written | 2026-04-17 13:46 |
| First code commit (scaffolding) | 2026-04-17 15:11 |
| Milestone complete (Stage E+G log) | 2026-04-18 15:11 |

- Calendar span: **~7 days**
- Active implementation + execution: **~24 h**

## 1. Dataset

### What we have

| Source | Version | Tarball | Status |
|---|---|---|---|
| Common Voice | **v25 (en)** | `datasets/cv-corpus-25.0-en.tar.gz` (81.5 GB) | **Truncated** — `gzip: unexpected end of file` on two independent B2 downloads |

### What we used

| Subset | Filter | Expected | Actual | Missing |
|---|---|---:|---:|---:|
| Indian-accent test | strict single-label `accents == "India and South Asia"` | 510 | **484** | 26 past EOF |
| Non-Indian test | strict single-label, sampled | 500 | **422** | 78 past EOF |

Reproducibility config: `scripts/head_surgery/repro_config.py:EXPECTED_N_INDIAN_ACCENT_IDS`. ID manifest: `tests/fixtures/head_surgery/indian_accent_ids.json`.

### Expected vs actual

| | Expected (pre-download) | Actual |
|---|---|---|
| CV version | CV24 (for midterm parity) | **CV25** — CV24 tarball unavailable in this env |
| Indian N | 511 (Srishti) / 510 (ours pre-truncation) | **484** |
| Baseline insertion rate | ~9.62% (midterm) | **1.27%** |

### Srishti's project vs ours

| | Srishti (midterm) | This milestone |
|---|---|---|
| Dataset | **Common Voice v24** | **Common Voice v25** |
| Indian-accent N | 511 | 484 |
| Baseline insertion rate | **9.62%** | **1.27%** |
| Breakdown (rep / syn / con) | reported ~non-zero repetition | **0.00% / 0.41% / 0.86%** |

Consequence: the ~8× drop between CV24 and CV25 on this subgroup means the head-surgery hypothesis (that a dominant hallucination-driving head could be masked to close the Indian-accent gap) is redefined — the baseline on CV25 is already near floor, so head masking has ≤0.08pp room to improve it. See milestone log [logs/head-surgery-diagnosis-complete.md](../logs/head-surgery-diagnosis-complete.md) §"Dataset drift".

## 2. Top-K hallucination-driving heads

|   layer |   head |   delta_insertion_rate |   delta_repetition |   delta_syntactic |   delta_content |   p_value_delta |   regression_ok |   non_indian_wer_masked |
|--------:|-------:|-----------------------:|-------------------:|------------------:|----------------:|----------------:|----------------:|------------------------:|
|      20 |     11 |                  0.001 |              0.000 |             0.001 |          -0.000 |           0.046 |           1.000 |                   0.066 |
|       0 |     15 |                  0.001 |              0.000 |             0.001 |           0.000 |           0.051 |           1.000 |                   0.066 |
|      22 |     19 |                  0.001 |              0.000 |             0.001 |           0.000 |           0.051 |           1.000 |                   0.067 |
|       7 |      6 |                  0.001 |              0.000 |             0.001 |          -0.000 |           0.128 |           1.000 |                   0.067 |
|      10 |      8 |                  0.001 |              0.000 |             0.001 |          -0.000 |           0.374 |           1.000 |                   0.067 |
|      25 |      5 |                  0.001 |              0.000 |             0.001 |          -0.000 |           0.051 |           1.000 |                   0.066 |
|       0 |     14 |                  0.001 |              0.000 |             0.001 |          -0.000 |           0.239 |           1.000 |                   0.068 |
|      13 |     17 |                  0.000 |              0.000 |             0.001 |          -0.000 |           0.132 |         nan     |                 nan     |
|      12 |      8 |                  0.000 |              0.000 |             0.001 |          -0.000 |           0.138 |         nan     |                 nan     |
|      11 |     11 |                  0.000 |              0.000 |             0.001 |          -0.000 |           0.133 |         nan     |                 nan     |


## 3. Decoding-strategy ablation (36 configs)

|   beam |   rep_penalty |   no_repeat_ngram | temp_fallback   |   total |   repetition |   syntactic |   content |
|-------:|--------------:|------------------:|:----------------|--------:|-------------:|------------:|----------:|
|      1 |         1.100 |                 0 | False           |   0.012 |        0.000 |       0.003 |     0.009 |
|      1 |         1.100 |                 0 | True            |   0.012 |        0.000 |       0.003 |     0.009 |
|      1 |         1.300 |                 3 | True            |   0.012 |        0.000 |       0.003 |     0.009 |
|      1 |         1.300 |                 3 | False           |   0.012 |        0.000 |       0.003 |     0.009 |
|      1 |         1.300 |                 0 | False           |   0.012 |        0.000 |       0.003 |     0.009 |
|      1 |         1.300 |                 0 | True            |   0.012 |        0.000 |       0.003 |     0.009 |
|      1 |         1.100 |                 5 | False           |   0.012 |        0.000 |       0.003 |     0.009 |
|      1 |         1.100 |                 5 | True            |   0.012 |        0.000 |       0.003 |     0.009 |
|      1 |         1.300 |                 5 | True            |   0.012 |        0.000 |       0.003 |     0.009 |
|      1 |         1.300 |                 5 | False           |   0.012 |        0.000 |       0.003 |     0.009 |

Top 10 configs by lowest insertion rate.

## 4. Energy-VAD under silence injection

| severity      |   db_floor |   total |   repetition |   syntactic |   content |   total_ref_words |
|:--------------|-----------:|--------:|-------------:|------------:|----------:|------------------:|
| silence_25pct |    -40.000 |   0.018 |        0.000 |       0.008 |     0.010 |              4885 |
| silence_50pct |    -40.000 |   0.018 |        0.000 |       0.009 |     0.009 |              4885 |
| silence_75pct |    -40.000 |   0.019 |        0.000 |       0.009 |     0.010 |              4885 |
| silence_25pct |    -35.000 |   0.019 |        0.000 |       0.008 |     0.011 |              4885 |
| silence_50pct |    -35.000 |   0.018 |        0.000 |       0.008 |     0.010 |              4885 |
| silence_75pct |    -35.000 |   0.021 |        0.000 |       0.011 |     0.010 |              4885 |
| silence_25pct |    -30.000 |   0.017 |        0.000 |       0.008 |     0.008 |              4885 |
| silence_50pct |    -30.000 |   0.020 |        0.001 |       0.009 |     0.010 |              4885 |
| silence_75pct |    -30.000 |   0.017 |        0.000 |       0.008 |     0.008 |              4885 |


## 5. All heads — ranked (top 50)

|   layer |   head |   delta_insertion_rate |   p_value_delta |   regression_ok |
|--------:|-------:|-----------------------:|----------------:|----------------:|
|      20 |     11 |                  0.001 |           0.046 |           1.000 |
|      10 |      8 |                  0.001 |           0.374 |           1.000 |
|       0 |     14 |                  0.001 |           0.239 |           1.000 |
|       7 |      6 |                  0.001 |           0.128 |           1.000 |
|      25 |      5 |                  0.001 |           0.051 |           1.000 |
|      22 |     19 |                  0.001 |           0.051 |           1.000 |
|       0 |     15 |                  0.001 |           0.051 |           1.000 |
|       3 |      3 |                  0.000 |           0.140 |         nan     |
|      15 |      6 |                  0.000 |           0.138 |         nan     |
|      12 |      8 |                  0.000 |           0.138 |         nan     |
|      15 |     18 |                  0.000 |           0.364 |         nan     |
|      17 |      8 |                  0.000 |           0.139 |         nan     |
|      23 |     16 |                  0.000 |           0.138 |         nan     |
|      24 |     18 |                  0.000 |           0.136 |         nan     |
|      23 |     11 |                  0.000 |           0.136 |         nan     |
|      19 |     17 |                  0.000 |           0.134 |         nan     |
|      22 |      1 |                  0.000 |           0.138 |         nan     |
|      22 |      2 |                  0.000 |           0.136 |         nan     |
|      21 |     15 |                  0.000 |           0.136 |         nan     |
|      25 |     18 |                  0.000 |           0.138 |         nan     |
|      24 |      6 |                  0.000 |           0.141 |         nan     |
|      31 |      9 |                  0.000 |           0.138 |         nan     |
|      20 |     18 |                  0.000 |           0.138 |         nan     |
|      18 |     17 |                  0.000 |           0.228 |         nan     |
|      16 |      2 |                  0.000 |           0.136 |         nan     |
|      20 |      6 |                  0.000 |           0.136 |         nan     |
|      20 |      5 |                  0.000 |           0.292 |         nan     |
|      16 |      1 |                  0.000 |           0.364 |         nan     |
|      30 |     16 |                  0.000 |           0.130 |         nan     |
|      29 |      6 |                  0.000 |           0.136 |         nan     |
|      11 |      2 |                  0.000 |           0.138 |         nan     |
|      16 |     13 |                  0.000 |           0.135 |         nan     |
|      13 |      0 |                  0.000 |           0.133 |         nan     |
|      11 |     11 |                  0.000 |           0.133 |         nan     |
|      13 |     17 |                  0.000 |           0.132 |         nan     |
|      22 |      5 |                  0.000 |           0.138 |         nan     |
|      25 |     16 |                  0.000 |           0.130 |         nan     |
|      25 |      0 |                  0.000 |           0.138 |         nan     |
|       4 |     11 |                  0.000 |           0.374 |         nan     |
|      21 |      2 |                  0.000 |           0.366 |         nan     |
|       0 |     12 |                  0.000 |           0.441 |         nan     |
|       4 |      8 |                  0.000 |           0.374 |         nan     |
|      30 |      0 |                  0.000 |           0.366 |         nan     |
|      21 |     10 |                  0.000 |           0.402 |         nan     |
|      19 |     12 |                  0.000 |           0.375 |         nan     |
|      23 |      1 |                  0.000 |           0.366 |         nan     |
|      22 |      9 |                  0.000 |           0.366 |         nan     |
|      31 |      8 |                  0.000 |           0.366 |         nan     |
|      21 |     18 |                  0.000 |           0.375 |         nan     |
|      26 |      1 |                  0.000 |           0.375 |         nan     |

*Full table in `outputs/head_surgery/head_scores.csv`.*
