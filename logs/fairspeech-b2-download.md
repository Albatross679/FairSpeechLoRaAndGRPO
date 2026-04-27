---
fileClass: Log
name: FairSpeech B2 Download
description: Downloaded and extracted the FairSpeech dataset from Backblaze B2, then updated the project-state HTML with exact local size and audio-format metrics.
created: 2026-04-27
updated: 2026-04-27
tags:
  - fairspeech
  - backblaze-b2
  - dataset
  - audio-format
aliases:
  - FairSpeech download
status: complete
subtype: data-prep
---

# FairSpeech B2 Download

Downloaded `b2://mlworkflow/datasets/meta_speech_fairness/` into `datasets/meta_speech_fairness/`.

## Downloaded objects

- `downloaded_file.tsv`: 3,972,173 bytes (3.79 MiB), 26,471 metadata rows.
- `An8OZvrqFdH2nxSzbrcFbtSjs0RWiKUXdfQ_DDJxlUbFxrtGKf66pKoCsbj_b8qMXpUKk3PJGLoIEUU2BeguUTsZj4bcit2SLRuMLLVop5w-V7JNRltRFu0L6g.zip`: 4,907,703,460 bytes (4.571 GiB), downloaded for extraction and then deleted locally on 2026-04-27.

## Extracted dataset

- Extracted WAV files: 26,471.
- Extracted WAV bytes: 6,456,234,270 bytes (6.013 GiB).
- Current apparent local size after deleting the zip archive: 6,460,206,443 bytes (6.017 GiB).
- `du -sk datasets/meta_speech_fairness` after deleting the zip archive: 6,358,572 KiB allocated (~6.064 GiB).

## Audio header probe

- Sample rate: 16,000 Hz for all 26,471 WAVs.
- Channels: mono for all 26,471 WAVs.
- Sample width: 2 bytes (16-bit PCM) for all 26,471 WAVs.
- Duration total: 55.55 hours.
- Duration mean / median / range: 7.55 s / 6.20 s / 1.36–65.62 s.

Updated `docs/project-state.html` to mark FairSpeech audio as local while preserving the caveat that FairSpeech raw ASR prediction/analysis tables are still missing.
