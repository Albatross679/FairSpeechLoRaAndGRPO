---
fileClass: Log
name: Common Voice Clips Localization
description: Moved and extracted Common Voice 24 audio clips into datasets/ and rewrote the Common Voice manifest to local audio paths.
status: complete
subtype: data-prep
created: 2026-04-27
updated: 2026-04-27
tags:
  - commonvoice
  - dataset
  - audio
  - manifest
aliases:
  - CV24 clips localization
---

# Common Voice Clips Localization

Localized the Common Voice clip archive for this repository.

## Changes

- Moved archive from `/Users/qifanwen/Downloads/common_voice_24_clips.zip` to `datasets/common_voice_24_clips.zip`.
- Extracted 16,398 MP3 clips to `datasets/common_voice_24_clips/`.
- Backed up the original external-path manifest to `datasets/data/cv_test_manifest_external_paths.csv`.
- Rewrote `datasets/data/cv_test_manifest.csv` so `audio_path` points to the local extracted MP3 files.

## Validation

- ZIP clip IDs and manifest utterance IDs matched exactly: 16,398 / 16,398.
- All rewritten manifest `audio_path` values resolve to existing local files.
