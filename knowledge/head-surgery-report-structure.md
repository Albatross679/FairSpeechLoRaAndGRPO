---
fileClass: Knowledge
name: Head-Surgery Report Structure
description: Section-by-section outline of the v2.0 head-surgery milestone writeup, reframed around the audio-pipeline correction that eliminated the midterm's 9.62% Indian-accent hallucination gap.
created: 2026-04-17
updated: 2026-04-17
tags:
  - head-surgery
  - v2.0
  - report-structure
  - negative-result
  - reproducibility-audit
aliases:
  - head-surgery report outline
  - v2.0 writeup structure
---

# Head-Surgery Milestone — Report Structure

This document lays out the planned structure of [`docs/head_surgery_report.md`](../docs/head_surgery_report.md) (produced by [`scripts/head_surgery/aggregate_report.py`](../scripts/head_surgery/aggregate_report.py)), and what each section is expected to contain given the milestone's pivot to a negative-result framing.

## Headline framing (decided 2026-04-17)

The original PRD ([tasks/prd-head-surgery-diagnosis.md](../tasks/prd-head-surgery-diagnosis.md)) framed this milestone as **"identify the decoder self-attention heads that drive Whisper-large-v3's 9.62% Indian-accent insertion rate."** The Stage A baseline rerun produced **1.27% on n=484 Indian-accent CV25 utterances**, attributable to two audio-pipeline fixes committed during the infra build:

- `77ca299` — resample audio to 16 kHz before Whisper inference.
- `3cecd6f` — switch to librosa for MP3 decode + resample.

The writeup is therefore reframed as a **reproducibility audit of the midterm's §3.2 finding**, with the head-surgery sweep results reported as a null-result appendix and the pipeline infrastructure as a methodological contribution.

## Section outline

### 1. Executive summary

One paragraph. Lead with: *Correcting the audio input pipeline (16 kHz resampling + librosa MP3 decode) reduces Whisper-large-v3's Indian-accent insertion rate from 9.62% (midterm §3.2) to 1.27% (this work), without any model changes. The proposed §4.2 head-surgery intervention is therefore unnecessary on correctly-processed CV25 audio. We additionally release the per-head attention-masking infrastructure (serial + batched-condition hooks, 640-cell sweep driver, paired-bootstrap scorer) for future work on harder hallucination regimes.*

### 2. Reproducibility audit

**Pulls from:** [`outputs/head_surgery/baseline_metrics.json`](../outputs/head_surgery/baseline_metrics.json), [`outputs/head_surgery/baseline_predictions.csv`](../outputs/head_surgery/baseline_predictions.csv).

| Sub-section | Content |
|---|---|
| 2.1 Setup | Whisper-large-v3 @ HF revision `06f233fe…`, seed pinned, generate() config frozen in `repro_config.py`, 484 pure-India-accent CV25 test utterances (vs midterm's ~511). |
| 2.2 Observed insertion rate | 1.27% total; 0% repetition, 0.41% syntactic, 0.86% content. Compare side-by-side with the midterm's 9.62% total / 43% rep / 48% syn / 9% content split. |
| 2.3 Attribution | Diff of audio pipeline between midterm and this work. Two fix commits with before/after insertion-rate cells if a rerun with the old pipeline is feasible (stretch). |
| 2.4 Subset-size caveat | Tarball truncation reduced the subset from ≥510 planned to 484 on disk. Size difference is too small to account for the 7.6pp shift. |

### 3. Methodological contribution — masking infrastructure

**Pulls from:** [`scripts/head_surgery/head_mask_hook.py`](../scripts/head_surgery/head_mask_hook.py), [`scripts/head_surgery/run_diagnosis_sweep.py`](../scripts/head_surgery/run_diagnosis_sweep.py), [`scripts/head_surgery/tune_batch_size.py`](../scripts/head_surgery/tune_batch_size.py).

| Sub-section | Content |
|---|---|
| 3.1 Per-head forward hook | Pre-hook on `decoder.layers[L].self_attn.out_proj`; serial + batched-condition variants; bytes-equivalent (Gate G3) within 1e-4 WER. |
| 3.2 Batched-condition inference | Per-sample head masks, safe under autoregressive `generate()` with KV cache (attention output is downstream of the cache). ~6× speedup vs serial at bs=32. |
| 3.3 Inference-specific batch-size tuner | Geometric sweep {1,2,4,8,16,32}; picks largest bs where VRAM < 90% and throughput ≥ 95% of best. Chosen bs=32 (22.5% VRAM, 14.1 utts/sec). |
| 3.4 Paired-bootstrap scorer | 10,000-iter resample over utterances; reports one-sided p-value for Δ insertion rate. |

### 4. Null-result appendix — head-masking sweep

**Pulls from:** [`outputs/head_surgery/sweep.csv`](../outputs/head_surgery/sweep.csv), [`outputs/head_surgery/head_scores.csv`](../outputs/head_surgery/head_scores.csv), [`outputs/head_surgery/top_k_heads.csv`](../outputs/head_surgery/top_k_heads.csv).

| Sub-section | Content |
|---|---|
| 4.1 Sweep geometry | 32 decoder layers × 20 self-attention heads × 484 utterances = 309,760 condition-utterances. Gates G3 + G4 passed. |
| 4.2 Top-10 head table | Ranked by Δ insertion rate. Expected: deltas near zero, p-values high — reported as-is for completeness. |
| 4.3 Per-head heatmap | 32×20 Δ heatmap from [`scripts/plots/head_surgery_heatmap.py`](../scripts/plots/head_surgery_heatmap.py). With the baseline near floor this is visually uninformative; included to demonstrate the tooling. |
| 4.4 Interpretation | Near-floor baseline (1.27%) means there is little hallucination signal for masking to eliminate. This is a consequence of the §2 audit finding, not a method failure. |

### 5. Decoding-strategy ablation (T5)

**Pulls from:** [`outputs/head_surgery/decoding_scores.csv`](../outputs/head_surgery/decoding_scores.csv).

36 configs: `beam ∈ {1, 5} × rep_penalty ∈ {1.0, 1.1, 1.3} × no_repeat_ngram ∈ {0, 3, 5} × temp_fallback ∈ {on, off}`. Top-10 by lowest insertion rate. Scientifically still interesting — explores whether specific decoding settings *widen* the tiny remaining insertion gap or *close* it. Recommend reporting the min/max range across configs alongside baseline 1.27%.

### 6. Energy-based VAD arm (T8)

**Pulls from:** [`outputs/head_surgery/vad_scores.csv`](../outputs/head_surgery/vad_scores.csv) (once Stage F runs). Depends on silence-injection perturbation manifests being either present or regenerable via [`scripts/data/generate_perturbations.py`](../scripts/data/generate_perturbations.py).

3 silence severities × 3 dB floors. This arm remains scientifically meaningful because silence-injected audio is a regime where hallucination *still* occurs (midterm §3.3) — the audio-pipeline fix does not suppress hallucination under acoustic degradation.

### 7. Discussion

| Sub-section | Content |
|---|---|
| 7.1 What the midterm's 9.62% likely measured | Hypothesis: input-sample-rate mismatch leading to Whisper receiving mis-scaled log-mel features, triggering pathological decoding (repetition loops). Supported by the 43/48/9 repetition/syntactic/content split reported by the midterm. |
| 7.2 When §4.2 head surgery *would* be the right tool | Regimes where Whisper-large-v3 still hallucinates after audio correction: silence-injected audio, very-low-SNR additive noise, chunk-masked audio (midterm §3.3). |
| 7.3 Limitations | n=484 vs planned 510–511 (tarball truncation); non-Indian regression guard at n≈422 vs planned 2,584 — smaller than midterm but adequate for Δ measurements. |
| 7.4 Future work | Repeating the sweep on a silence-25% or noise-10dB perturbation condition would answer the original §4.2 question in the regime where it matters. |

### 8. Artifact index

A flat table of every file produced, with path, provenance (which stage), and one-line description. Generated automatically by Stage G.

| Artifact | Stage | Description |
|---|---|---|
| `outputs/head_surgery/baseline_metrics.json` | A | Gate G1 baseline (1.27% observed) |
| `outputs/head_surgery/tune_batch_size.json` | A.5 | Gate G1.5, chosen bs=32 |
| `outputs/head_surgery/pilot_metrics.json` | B | Gate G2 (relaxed for near-floor baseline) |
| `outputs/head_surgery/sweep.csv` | C | Gates G3, G4 — 309,760-row sweep |
| `outputs/head_surgery/head_scores.csv` | D | Per-(L, h) Δ + bootstrap p-values + regression flags |
| `outputs/head_surgery/top_k_heads.csv` | D | Top-10 ranked by Δ |
| `outputs/head_surgery/decoding_scores.csv` | E | 36-config decoding ablation |
| `outputs/head_surgery/vad_scores.csv` | F | 3 × 3 energy-VAD grid |
| `docs/head_surgery_heatmap.png` | (stretch) | 32×20 Δ heatmap |
| `docs/head_surgery_report.md` | G | This report |

### 9. Reproducibility

**Pulls from:** [`scripts/head_surgery/repro_config.py`](../scripts/head_surgery/repro_config.py), [`tests/fixtures/head_surgery/indian_accent_ids.json`](../tests/fixtures/head_surgery/indian_accent_ids.json).

One subsection each: model revision hash, frozen `generate()` config, 484-ID snapshot, library versions pinned in [`pyproject.toml`](../pyproject.toml).

## Writing conventions

- **Data tables first, prose second.** Every claim anchors to a cell in an artifact.
- **Paired comparisons use the same column order.** `midterm | this work | Δ` always, never `this work | midterm`.
- **Near-zero numbers get explicit precision.** Report insertion rates to 2 decimal places (0.01 pp); bootstrap p-values to 3 significant figures.
- **Negative-result framing is stated upfront**, not buried. The executive summary carries the pivot.

## Cross-references

- Upstream spec: [tasks/prd-head-surgery-diagnosis.md](../tasks/prd-head-surgery-diagnosis.md)
- Implementation plan: [docs/superpowers/plans/2026-04-17-head-surgery-diagnosis.md](../docs/superpowers/plans/2026-04-17-head-surgery-diagnosis.md)
- Progress snapshot: [docs/head-surgery-progress.html](../docs/head-surgery-progress.html)
- Pivot-decision memory: `memory/project_head_surgery_v2_negative_result_pivot.md` (private)
