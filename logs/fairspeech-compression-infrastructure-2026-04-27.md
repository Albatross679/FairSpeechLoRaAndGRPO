---
fileClass: Log
name: FairSpeech compression infrastructure
description: Built CPU-side infrastructure for the FairSpeech compression and resampling ASR fairness experiment.
created: 2026-04-27
updated: 2026-04-27
tags:
  - fairspeech
  - compression
  - infrastructure
  - inference
aliases:
  - FairSpeech compression infrastructure
status: complete
subtype: feature
---

# FairSpeech compression infrastructure

Built the infrastructure requested by Chapter 2 of `docs/project-state.html`:

- Added `scripts/data/prepare_fairspeech_compression.py` for FairSpeech source manifests, derived-audio variant manifests, and optional ffmpeg-based variant generation.
- Added `scripts/validation/validate_fairspeech_compression.py` for row-count, audio existence, WAV-format, duration, and optional silence checks.
- Added `scripts/inference/build_duration_batch_plan.py` for duration-bucketed variable-size batch plans and calibration subsets.
- Added `scripts/inference/profile_batch_policy.py` for VM pilot-gate batch policy profiling with VRAM monitoring.
- Added `scripts/inference/prepare_model_cache.py` for nine-model cache/download manifests and smoke-command scaffolding.
- Extended `scripts/inference/run_inference.py` with `--audio_variant` labels and `--batch_plan` JSONL support.
- Added `scripts/metrics/compute_fairspeech_compression_metrics.py` and `scripts/plots/generate_fairspeech_compression_plots.py` as downstream result scaffolding.
- Added `tests/test_fairspeech_compression_infra.py` CPU-only coverage for manifest creation, batch planning, and batch-plan ID mapping.
- Added a `.gitignore` exception so new source files under `scripts/data/` remain trackable despite the local-data ignore pattern.

Local ignored artifacts were generated under `datasets/fairspeech_compression/`:

- six 26,471-row variant manifests,
- a passing baseline validation summary,
- one baseline duration-bucketed batch plan with 1,866 batches,
- a 28-row duration-stratified calibration subset,
- a dry-run model cache manifest for all nine benchmark models.

No full derived-audio generation, model download, GPU profiling, or ASR inference run was launched.
