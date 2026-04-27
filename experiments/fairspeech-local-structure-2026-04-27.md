---
fileClass: Experiment
name: FairSpeech Local Structure Audit
description: Inspected the downloaded FairSpeech metadata and WAV directory to verify local structure, row/file matching, schema, demographic coverage, and audio header properties.
created: 2026-04-27
updated: 2026-04-27
tags:
  - fairspeech
  - dataset-map
  - audio-format
  - demographics
aliases:
  - FairSpeech local structure
status: complete
---

# FairSpeech Local Structure Audit

Audited `/Users/qifanwen/Desktop/Vault1/projects/FairSpeechLoRaAndGRPO/datasets/meta_speech_fairness/` after downloading from Backblaze B2 and deleting the local zip archive.

## Local structure

- `downloaded_file.tsv`: 26,471 rows × 7 columns; 3.79 MiB.
- `asr_fairness_audio/`: 26,471 `.wav` files; 6.013 GiB apparent WAV bytes.
- Join key: `hash_name` maps directly to `asr_fairness_audio/{hash_name}.wav`.
- Completeness: 26,471 / 26,471 metadata rows have matching WAV files; 0 WAVs lack metadata.
- Local zip archive: deleted after extraction; remote B2 copy remains.

## Metadata schema

Columns: `hash_name`, `transcription`, `age`, `gender`, `first_language`, `socioeconomic_bkgd`, `ethnicity`.

- Unique `hash_name`: 26,471.
- Unique transcriptions: 23,046.
- No blank values on the five demographic axes checked.

## Demographic coverage

- Ethnicity: Black or African American 7,807; White 5,646; Native American / American Indian / Alaska Native 4,632; Asian / South Asian / Asian American 3,854; Hispanic / Latino / Spanish 2,814; Native Hawaiian / Other Pacific Islander 969; Middle Eastern / North African 749.
- Gender: female 14,422; male 12,049.
- Age: 31–45 12,770; 46–65 5,687; 23–30 4,168; 18–22 3,846.
- First language: 30 groups. Largest are English 21,528; Spanish 1,862; Mandarin 422; Hindi 233; Cantonese 191; Tagalog 190; French 174; Portuguese 172.
- Socioeconomic background: Low 14,780; Medium 9,824; Affluent 1,867.

## Audio header probe

- Local cache: `datasets/data/fairspeech_audio_format_probe.csv`.
- Sample rate: 16,000 Hz for all 26,471 WAVs.
- Channels: mono for all 26,471 WAVs.
- Sample width: 2 bytes / 16-bit PCM for all 26,471 WAVs.
- Effective PCM bitrate: 256 kbps for all 26,471 WAVs (`16,000 Hz × 16 bits × 1 channel`).
- Total duration: 55.55 hours.
- Mean / median / min / max duration: 7.55 s / 6.20 s / 1.36 s / 65.62 s.
- Percentiles: p95 15.32 s; p99 32.36 s.

## Duration histogram

| Duration bin | Count | Share |
|---:|---:|---:|
| 0–3 s | 753 | 2.8% |
| 3–5 s | 6,762 | 25.5% |
| 5–7 s | 8,882 | 33.6% |
| 7–10 s | 6,050 | 22.9% |
| 10–15 s | 2,629 | 9.9% |
| 15–20 s | 667 | 2.5% |
| 20–30 s | 415 | 1.6% |
| 30–45 s | 159 | 0.6% |
| 45–66 s | 154 | 0.6% |

The long tail motivates per-model batch-size profiling or duration-bucketed inference before full VM runs.

## Source-rate control decision

FairSpeech does not need the Common Voice-style source-rate filter. All 26,471 metadata rows have matching 16 kHz mono 16-bit PCM WAV files, and all rows have ethnicity labels. Use all FairSpeech rows for the secondary ethnicity compression study after ASR predictions/analysis tables are available or rerun locally.

## Project-state update

Updated `docs/project-state.html` with a new FairSpeech local data-structure subsection, metadata schema, demographic coverage, corrected local archive status, and a Chapter 2 source-rate-control table.
