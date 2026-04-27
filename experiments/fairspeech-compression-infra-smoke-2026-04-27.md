---
fileClass: Experiment
name: FairSpeech compression infrastructure smoke checks
description: CPU-side smoke checks for the FairSpeech compression experiment infrastructure.
created: 2026-04-27
updated: 2026-04-27
tags:
  - fairspeech
  - compression
  - smoke-test
  - infrastructure
aliases:
  - FairSpeech compression infra smoke
status: complete
algorithm: duration-bucketed inference planning
task: ASR fairness compression evaluation infrastructure
hardware: local CPU
---

# FairSpeech compression infrastructure smoke checks

## Commands run

- `python3 -m py_compile ...` over all new/modified infrastructure scripts.
- `python3 scripts/data/prepare_fairspeech_compression.py --max-samples 5 --variants all --output-dir /tmp/fairspeech_comp_smoke`
- `python3 scripts/inference/build_duration_batch_plan.py ... --max-padded-seconds 30 --max-samples 4`
- `python3 scripts/validation/validate_fairspeech_compression.py ... --expected-count 5`
- `python3 scripts/inference/prepare_model_cache.py --models wav2vec2-large,whisper-small ...`
- `python3 scripts/data/prepare_fairspeech_compression.py --max-samples 1 --variants baseline,bottleneck_8k,mp3_16k --generate-audio ...`
- `python3 scripts/validation/validate_fairspeech_compression.py ... --expected-count 1`
- Manual execution of the new test functions because `pytest` is not installed in the local Python environment.

## Results

- Syntax compilation passed.
- Five-row manifest-only smoke run passed.
- One-file ffmpeg generation for baseline, 8 kHz bottleneck, and MP3 16 kbps roundtrip passed validation.
- Baseline full manifest validation passed for 26,471 rows.
- Baseline batch plan was generated with 1,866 variable-size batches using a 160 padded-second / 16 max-sample candidate policy.
- Model cache dry-run manifest successfully listed all nine benchmark models.

## Caveat

`python3 -m pytest tests/test_fairspeech_compression_infra.py -q` could not run because `pytest` is not installed locally. The test functions were executed manually and passed.
