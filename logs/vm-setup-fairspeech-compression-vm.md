---
fileClass: Log
name: FairSpeech Compression VM Setup
description: Set up the A40 VM for the FairSpeech compression experiment and fixed package metadata so editable install can resolve the HTML project overview.
created: 2026-04-27
updated: 2026-04-27
tags:
  - vm-setup
  - fairspeech
  - packaging
aliases:
  - FairSpeech VM setup
status: complete
subtype: setup
---

# FairSpeech Compression VM Setup

Set up `root@194.68.245.66:22018` for the FairSpeech-only compression/resampling experiment described in `docs/project-state.html`.

Migrated only the required active data payload:

- `datasets/meta_speech_fairness/downloaded_file.tsv`
- `datasets/meta_speech_fairness/asr_fairness_audio/`

Did not migrate Common Voice clips/results, LibriSpeech results, local archives, or previously generated prediction tables.

During setup, `pip install -e .` failed because `pyproject.toml` referenced a root `project-overview.html` file and did not specify an HTML content type. Updated the package metadata to use the existing `docs/project-overview.html` with `content-type = "text/html"`.
