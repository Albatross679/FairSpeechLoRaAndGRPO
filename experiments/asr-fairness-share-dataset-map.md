---
fileClass: Experiment
name: ASR Fairness Share Dataset Map
description: Quick inventory of the unzipped asr_fairness_share archive under datasets/.
status: complete
created: 2026-04-27
updated: 2026-04-27
tags:
  - dataset-map
  - commonvoice
  - librispeech
  - asr-fairness
aliases:
  - asr_fairness_share map
---

# ASR Fairness Share Dataset Map

## Location

- Archive: `datasets/asr_fairness_share.zip`
- Unzipped contents: `datasets/data/` and `datasets/results/`
- Total unzipped dataset area: about 602 MB including the 106 MB zip archive.

## Top-level map

| Path | Role | Notes |
|---|---|---|
| `datasets/data/cv_test_manifest.csv` | Common Voice test manifest | 16,398 rows; columns: `utterance_id`, `audio_path`, `sentence`, `gender`, `accent`, `accent_raw`, `age`, `client_id` |
| `datasets/data/ls_manifest.csv` | LibriSpeech manifest | 2,620 rows; columns: `utterance_id`, `audio_path`, `sentence`, `sentence_raw` |
| `datasets/results/commonvoice/` | Common Voice prediction and metadata CSV/JSON files | 117 prediction CSVs + 117 metadata JSONs, all with 16,398 rows |
| `datasets/results/commonvoice/analysis/` | Common Voice analysis outputs | JSON summaries, CSV summaries, figures, LaTeX tables, hypothesis markdown |
| `datasets/results/librispeech/` | LibriSpeech prediction and metadata files | 9 complete clean-model prediction CSVs with 2,620 rows, plus one 2,400-row checkpoint CSV |
| `datasets/results/figures/` | Paper/result figure data | PNG/PDF figures and source CSVs for perturbation/fairness plots |
| `datasets/results/tables/` | Paper/result tables | CSV tables for WER, insertion rate, hallucination classification, and fairness amplification |
| `datasets/results/hallucination_analysis/` | Hallucination analysis outputs | JSON/PNG/PDF artifacts |

## File inventory

- 148 CSV files
- 130 JSON files
- 24 PNG files
- 23 PDF files
- 4 TeX files
- 2 Markdown files
- 1 ZIP archive

## Common Voice coverage

- Models: `wav2vec2_large`, `whisper_small`, `whisper_medium`, `whisper_large_v3`, `qwen3_asr_0.6b`, `qwen3_asr_1.7b`, `granite_speech_3.3_2b`, `granite_speech_3.3_8b`, `canary_qwen_2.5b`.
- Conditions: clean, `snr_20db`, `snr_10db`, `snr_0db`, `silence_25pct`, `silence_50pct`, `silence_75pct`, `mask_10pct`, `mask_20pct`, `mask_30pct`, `reverb_0_3s`, `reverb_0_6s`, `reverb_1_0s`.
- Matrix shape: 9 models × 13 conditions = 117 prediction files.

## Manifest demographic availability

Common Voice manifest has substantial missing demographic labels:

- `gender`: 14,111 missing; 1,753 male; 534 female.
- `age`: 13,808 missing; largest labeled group is twenties with 1,305 rows.
- `accent`: 13,253 missing; largest labeled groups include US (1,193), Indian (511), England (381), Canada (99), Australia (98), African (51).

LibriSpeech manifest has no demographic columns in the shared manifest.

## Important caveat

The archive does **not** contain raw audio files. Manifests point to absolute external paths such as `/users/PAS2030/.../*.mp3` and `/users/PAS2030/.../*.wav`, so local reruns will need the original audio mounted or paths rewritten.
