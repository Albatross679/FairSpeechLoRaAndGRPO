---
fileClass: Log
name: Project State Data Inventory Page
description: Created a concise Palatino Burgundy project-state HTML page summarizing locally available manifests, inference outputs, perturbation outputs, and data caveats.
created: 2026-04-27
updated: 2026-04-27
tags:
  - project-state
  - data-inventory
  - commonvoice
  - perturbation
aliases:
  - project state data page
status: complete
subtype: research
---

# Project State Data Inventory Page

Created `docs/project-state.html` as a concise personal state file. The first chapter documents the local processed data state: Common Voice clips/manifests, clean and perturbed inference CSVs, analysis tables/figures, LibriSpeech clean baselines, and the current FairSpeech data gap.

Updated section 1.3 to explicitly define the 13 Common Voice conditions: clean, three additive-noise SNR levels, three reverberation levels, three silence-injection levels, and three chunk-masking levels.

Updated section 1.2 to define artifact categories: prediction CSVs, metadata JSONs, analysis outputs, perturbation summary tables, and figure/source-CSV outputs.

Replaced the abstract artifact-shape table in section 1.2 with concrete example files from the local result directories, including clean predictions, perturbed predictions, metadata JSON, clean summary, error decomposition, perturbation table, and figure artifact.

Replaced the directory-example table in section 1.2 with actual CSV row previews from a Common Voice prediction CSV and the accent perturbation summary CSV.

Added section 1.0 with a compact `datasets/` file tree showing local Common Voice clips/manifests, LibriSpeech manifest/results, Common Voice inference outputs, metadata, analysis outputs, perturbation tables, and figure artifacts.

Added section 1.2 to `docs/project-state.html` with the Common Voice 24 audio encoding audit: codec, channel count, sample-rate distribution, bitrate distribution, duration summary, FairSpeech/LibriSpeech local-status caveats, and terminology for codec, bitrate, sample rate, resampling, and model-internal compression.

Updated the FairSpeech entries after deleting the local B2 zip archive: `docs/project-state.html` now lists only the extracted WAV directory and metadata TSV under `datasets/meta_speech_fairness/`.

Added section 1.3 to `docs/project-state.html` after auditing the extracted FairSpeech structure. The section now documents the metadata TSV, WAV directory, `hash_name` to WAV join rule, 1:1 completeness, schema columns, demographic coverage counts, and the fact that the local zip archive was deleted after extraction.
