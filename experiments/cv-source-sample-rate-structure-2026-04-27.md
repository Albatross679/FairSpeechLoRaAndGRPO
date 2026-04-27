---
fileClass: Experiment
name: Common Voice Source Sample-Rate Structure
description: Probed Common Voice 24 audio headers and joined source sample rate/bitrate back to the accent manifest to decide whether the compression experiment should use a single original sampling rate.
created: 2026-04-27
updated: 2026-04-27
tags:
  - common-voice
  - audio-format
  - sample-rate
  - accent-fairness
aliases:
  - CV source-rate structure
status: complete
---

# Common Voice Source Sample-Rate Structure

## Method

- Probed all 16,398 local Common Voice MP3 files with `ffprobe`.
- Wrote local cache: `datasets/data/cv_audio_format_probe.csv` (gitignored with datasets).
- Joined probe output to `datasets/data/cv_test_manifest.csv` by `utterance_id`.

## Overall file structure

| Original sample rate | Clip count | Source bitrate | Codec / channels |
|---:|---:|---:|---|
| 48,000 Hz | 8,940 | 64 kbps | MP3 / mono |
| 32,000 Hz | 7,403 | 48 kbps | MP3 / mono |
| 44,100 Hz | 55 | 64 kbps | MP3 / mono |

All files joined back to the manifest; no missing sample-rate probes.

## Main accent subset

Here, "main accent" means the six normalized Common Voice accent groups used in the current accent-fairness analysis: US, Indian, England, Canada, Australia, and African.

| Accent | 32 kHz rows | 48 kHz rows | Total |
|---|---:|---:|---:|
| US | 816 | 377 | 1,193 |
| Indian | 309 | 202 | 511 |
| England | 277 | 104 | 381 |
| Canada | 71 | 28 | 99 |
| Australia | 82 | 16 | 98 |
| African | 36 | 15 | 51 |

No main-accent rows are 44.1 kHz.

## Non-main accent rows

| Accent bucket | 32 kHz | 44.1 kHz | 48 kHz | Total | Primary-use decision |
|---|---:|---:|---:|---:|---|
| Six main accents | 1,591 | 0 | 742 | 2,333 | Use for main accent-fairness comparison. |
| Missing accent label | 5,071 | 53 | 8,129 | 13,253 | Exclude from accent-group fairness; may remain in overall WER. |
| Other long-tail/composite labels | 741 | 2 | 69 | 812 | Exclude from primary comparison unless doing exploratory long-tail analysis. |

Top long-tail/composite labels include Scotland (44), Hong Kong (26), Ireland (26), New Zealand (24), Wales (19), Malaysia (18), Filipino (18), Singapore (16), and composite labels such as `united_states_english,england_english` (16). There are 359 distinct non-main non-empty accent labels, many with very small counts.

## Decision

Use the original **32 kHz** Common Voice subset for the primary CV compression/resampling experiment.

Rationale:

- It has 1,591 main-accent rows versus 742 at 48 kHz.
- It covers all six main accent groups.
- It also fixes the source bitrate at 48 kbps, removing a second source-format confound.
- African accent drops to 36 rows after filtering, so it should be reported as exploratory if using an `n ≥ 50` threshold.

FairSpeech remains the cleaner secondary codec/bitrate dataset because all local source files are 16 kHz mono 16-bit PCM WAV.

## Project-state update

Updated `docs/project-state.html` Chapter 2 to replace the mixed-rate Common Voice baseline with a source-rate-controlled 32 kHz CV plan.
