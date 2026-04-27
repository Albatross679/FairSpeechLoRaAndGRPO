---
fileClass: Knowledge
name: ASR Fairness Metrics From Existing Inference Research
description: Feasibility research on fairness metrics that can be computed from the project's existing ASR inference outputs.
created: 2026-04-27
updated: 2026-04-27
tags:
  - asr
  - fairness
  - metrics
  - common-voice
  - inference
aliases:
  - ASR fairness metric add-ons
---

# ASR Fairness Metrics From Existing Inference: Research Report

**Researched:** 2026-04-27  
**Domain:** ASR fairness  
**Confidence:** HIGH for local data inventory and no-GPU metric feasibility; MEDIUM for literature-derived prioritization because fairness metrics are context-dependent.  
**Mode:** feasibility

---

## Summary

The current local data is enough to add several fairness analyses without rerunning ASR inference. The strongest additions are: **catastrophic/high-error-rate parity**, **speaker-aware or mixed-effects statistical testing**, **intersectional fairness**, **robustness disparity curves/AUC under perturbations**, and **inequality-index summaries** such as Gini or Theil over group WERs. These additions fit the teammate recommendation to focus on compression and perturbation robustness, because they can show whether compressed or LLM-ASR models fail only in average WER or also in tail-risk and robustness disparity.

The current teammate metrics already cover the basic story: per-group WER, MMR, relative/absolute gaps, bootstrap CIs, Mann-Whitney tests, error-type decomposition, WERD, and perturbation fairness amplification. The main gap is not missing WER tables. The gap is **tail risk, uncertainty structure, and whether disparities persist after controlling for speaker/repeated-utterance structure**.

## Current local inference/data state

| Area | Local state | What it enables |
|---|---:|---|
| Common Voice audio | 16,398 local MP3s in `datasets/common_voice_24_clips/` | Re-read audio if needed, but most proposed metrics do not need audio |
| Common Voice predictions | 117 CSVs = 9 models × 13 conditions | Clean and perturbed fairness metrics across accent, gender, age |
| Common Voice prediction schema | `utterance_id`, `reference`, `hypothesis`, `wer`, demographics, model metadata; most perturbed files also include `num_hyp_words`, `num_ref_words`, `perturbation` | WER, per-utterance tail metrics, edit decomposition, degradation curves |
| Common Voice demographics | Accent/gender/age in prediction files; `client_id` available by joining `datasets/data/cv_test_manifest.csv` on `utterance_id` | Speaker-aware bootstrap, mixed-effects models, intersectional cells |
| LibriSpeech predictions | 9 complete clean CSVs, plus one partial checkpoint file | Accuracy baseline only; no demographic fairness axis locally |
| FairSpeech raw prediction tables | Not locally present | Cannot extend FairSpeech metrics locally until teammate shares raw CSV/JSON |

## Metrics already used by teammates

| Metric family | Evidence in repo | Status |
|---|---|---|
| Per-group WER by accent/gender/age | `datasets/results/commonvoice/analysis/full_analysis.json`; `datasets/results/tables/table_wer_by_perturbation_and_*_cv.csv` | Already used |
| Max-min WER ratio / MMR | `compute_fairness_metrics.py`, `compute_perturbation_metrics.py` | Already used |
| Relative gap % and absolute WER gap | `summary.csv`, `table_wer_by_perturbation_and_*_cv.csv` | Already used |
| WER standard deviation/range | Present in `full_analysis.json`, less visible in summaries | Computed but underused |
| Bootstrap CIs and Mann-Whitney U | `compute_fairness_metrics.py` | Already used for clean CV |
| Substitution / insertion / deletion rates | `error_decomposition.json`; `table_insertion_rate_by_perturbation_and_*_cv.csv` | Already used partly; perturbation tables emphasize insertion more than sub/del |
| WERD and fairness amplification | `table_fairness_gap_amplification_*_cv.csv`; figures CSVs | Already used |
| Hallucination insertion classification | Implemented in perturbation code, but local hallucination-analysis directory is absent | Mostly not part of current local deliverable |

## Recommended additions that need no new inference

### 1. Catastrophic / high-error-rate parity

**Definition:** For each group, compute the fraction of utterances whose per-utterance WER exceeds a threshold, e.g. `WER >= 0.5`.

**Why add it:** Average WER hides whether one group has many unusable transcripts. Stanford's ASR disparity audit also reported a threshold where at least half the words were botched, not only average WER.

**Local example, clean Common Voice accent:**

| Model | Worst group by high-error rate | Best group | Gap |
|---|---:|---:|---:|
| wav2vec2-large | Indian 20.4% | Australia 4.1% | 16.3 pp |
| whisper-large-v3 | African 5.9% | Canada 0.0% | 5.9 pp |
| qwen3-asr-1.7b | African 3.9% | Canada 1.0% | 2.9 pp |
| granite-speech-3.3-8b | African 13.7% | Australia 1.0% | 12.7 pp |

**Recommendation:** Add this. It is simple, interpretable, and materially different from MMR.

### 2. Speaker-aware uncertainty and mixed-effects disparity

**Definition:** Join predictions to `client_id`, then estimate disparity with speaker-aware bootstrap or a Poisson/negative-binomial model of word errors with reference-word count as exposure.

**Why add it:** Current CIs treat utterances as independent. Common Voice has repeated contributors, so speaker-level resampling is a cleaner uncertainty story. Mixed-effects Poisson regression is specifically proposed in ASR fairness literature to control nuisance factors and unobserved speaker heterogeneity.

**Feasibility:** Directly computable from predictions plus manifest. Need word-level error counts from `jiwer.process_words(reference, hypothesis)` and `client_id` from the manifest.

**Recommendation:** Add if you want one statistically mature contribution. It strengthens any compression/fairness claim without changing the model runs.

### 3. Robustness disparity AUC / slope under perturbation severity

**Definition:** For each model and group, summarize WER degradation across severity levels as a slope or area under curve. Then compute gap/ratio of those slopes across groups.

**Why add it:** Teammates suggested focusing on compression and perturbations. Current tables show MMR amplification at each perturbation, but a slope/AUC condenses the robustness story: “which group degrades fastest as noise/reverb/masking increases?”

**Local example, accent WERD range from clean to SNR 0 dB:**

| Model | Least-degraded accent | Most-degraded accent | WERD spread |
|---|---|---|---:|
| wav2vec2-large | African +153% | Canada +275% | 122 pp |
| whisper-large-v3 | Indian +30% | Canada +241% | 212 pp |
| qwen3-asr-1.7b | Australia +48% | Indian +129% | 81 pp |
| granite-speech-3.3-8b | African +12% | Canada +154% | 142 pp |

**Recommendation:** Add this. It directly supports a compression/robustness chapter.

### 4. Intersectional fairness

**Definition:** Compute WER/high-error metrics for intersections such as `accent × gender`, `accent × age`, and `gender × age` when cell size is large enough.

**Local cell availability with `n >= 50`:**

| Intersection | Valid cells | Best use |
|---|---:|---|
| accent × gender | 4 | Conservative appendix only |
| accent × age | 6 | Useful exploratory analysis |
| gender × age | 8 | Strong enough for a compact table |
| accent × gender × age | 6 | Too sparse for primary claims |

**Recommendation:** Add a small “intersectional stress test” table, but keep it secondary because Common Voice labels are sparse.

### 5. Inequality-index summaries over group losses

**Definition:** Compute a one-number inequality index over group WERs or per-utterance WERs, e.g. Gini, Theil / generalized entropy, coefficient of variation.

**Why add it:** MMR only uses the best and worst group. Gini/Theil use all groups, so they are more stable when the best/worst group changes across perturbations.

**Local example, clean accent group mean WER:** Whisper-large-v3 has high accent inequality by Gini/Theil despite low average WER, while Qwen3-ASR-1.7B has lower inequality.

**Recommendation:** Add one inequality metric, probably **Theil** or **Gini**, but do not replace per-group WER. Use it as a compact ranking/appendix metric.

### 6. Error-type disparity beyond insertion

**Definition:** For each axis and perturbation, compute max-min ratio and gap for substitution rate, deletion rate, and insertion rate separately.

**Why add it:** The current perturbation tables emphasize insertion rate. Compression and perturbation may instead show deletion-heavy behavior, especially under masking/silence.

**Recommendation:** Add if time permits. It complements the teammate's advice while avoiding a hallucination-centered framing.

## Metrics not recommended from current files

| Metric | Why not now |
|---|---|
| Calibration / confidence fairness | Prediction CSVs do not contain token probabilities, logprobs, entropy, or confidence scores |
| Equalized odds / TPR/FPR in the standard classification sense | ASR is sequence prediction, not binary classification; can only approximate via exact-match/high-error thresholds |
| Phonetic error rate by accent | Needs pronunciation/phonetic annotation or forced alignment; not directly in existing CSVs |
| FairSpeech ethnicity extensions | Local raw FairSpeech prediction CSVs are absent; only figure artifacts are present |
| Audio-quality fairness by duration/SNR | Duration can be recomputed from local CV audio, but actual perturbation audio files are not locally stored as separate artifacts |

## Concrete next step

Create one new no-GPU analysis script that writes:

- `datasets/results/commonvoice/analysis/additional_fairness_metrics.csv`
- `datasets/results/commonvoice/analysis/high_error_rate_by_group.csv`
- `datasets/results/commonvoice/analysis/robustness_disparity_auc.csv`
- optionally `datasets/results/commonvoice/analysis/speaker_clustered_bootstrap.json`

Minimal first version:

1. Load all `datasets/results/commonvoice/predictions_*.csv`.
2. Join `client_id` from `datasets/data/cv_test_manifest.csv`.
3. For each model × perturbation × axis × group, compute mean utterance WER, corpus WER, high-error rate, p90 WER, sub/ins/del rates.
4. For each model × perturbation × axis, compute MMR, absolute gap, relative gap, Gini, Theil, high-error-rate gap.
5. For each model × axis × perturbation type, compute severity slope/AUC of group WERD.

## Sources

| Source | Type | Date | Key claim supported |
|---|---|---:|---|
| https://news.stanford.edu/stories/2020/03/automated-speech-recognition-less-accurate-blacks | university report | 2020-03-23 | ASR audits should report group WER disparities and high-error threshold rates, not only aggregate WER |
| https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html | documentation | current | Fairness assessment commonly converts ordinary error metrics into group metrics and reports differences/ratios |
| https://fairlearn.org/v0.8/user_guide/mitigation.html | documentation | current | Bounded group loss and error-rate parity provide analogies for ASR loss parity |
| https://arxiv.org/abs/2109.09061 | paper | 2021-09-20 | Mixed-effects Poisson regression is proposed for ASR fairness to control nuisance factors and unobserved heterogeneity |
| https://www.isca-archive.org/interspeech_2025/rai25_interspeech.html | paper page | 2025 | ASR-FAIRBENCH integrates WER and fairness scoring using mixed-effects modeling |
| https://www.isca-archive.org/interspeech_2024/veliche24_interspeech.pdf | paper | 2024 | Fair-Speech is designed for ASR evaluation across demographic labels; ASR fairness needs demographic breakdowns beyond aggregate WER |
| https://aclanthology.org/2025.findings-emnlp.1044.pdf | paper | 2025 | Fairness conclusions change depending on metric; gap-only metrics can hide degraded disadvantaged-group performance |
| https://arxiv.org/abs/1807.00787 | paper | 2018 | Inequality indices can quantify individual/group unfairness and decompose between/within group effects |
| https://link.springer.com/article/10.1007/s41060-024-00541-w | review | 2024 | Generalized entropy/Theil are documented fairness/inequality metrics for algorithmic outcomes |
| https://arxiv.org/abs/2510.18374 | paper | 2026 revision | Recent ASR fairness work uses macro-averaged WER and fairness objectives across accent groups |
