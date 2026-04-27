---
fileClass: Experiment
name: Common Voice 24 Clips ZIP Inspection
description: Quick inventory and manifest match check for /Users/qifanwen/Downloads/common_voice_24_clips.zip.
status: complete
created: 2026-04-27
updated: 2026-04-27
tags:
  - dataset-map
  - commonvoice
  - audio
  - zip-inspection
aliases:
  - common_voice_24_clips inspection
---

# Common Voice 24 Clips ZIP Inspection

Inspected `/Users/qifanwen/Downloads/common_voice_24_clips.zip` without extracting all audio.

## Summary

- Archive size: 640 MiB on disk (`671,411,664` bytes).
- Entries: 16,399 total.
  - Directories: 1 (`common_voice_24_clips/`).
  - Files: 16,398.
- File type: all files are `.mp3`.
- Uncompressed bytes: `691,941,949` bytes, about 660 MiB.
- Filename pattern: all files match `common_voice_24_clips/common_voice_en_<id>.mp3`.
- Unique file stems: 16,398; no duplicate clip IDs.
- ZIP timestamp range: `2025-12-08 09:06:30` to `2025-12-08 10:15:28`.

## Size distribution

- Minimum MP3 size: 4,653 bytes.
- Median MP3 size: 39,789 bytes.
- Mean MP3 size: 42,196.7 bytes.
- Maximum MP3 size: 845,409 bytes.

## Match against project manifest

Compared ZIP clip IDs with `datasets/data/cv_test_manifest.csv`:

- Manifest rows: 16,398.
- Manifest unique utterance IDs: 16,398.
- ZIP clip IDs missing from manifest: 0.
- Manifest IDs missing from ZIP: 0.

Conclusion: this ZIP appears to be the complete Common Voice audio clip bundle for the current `cv_test_manifest.csv`.
