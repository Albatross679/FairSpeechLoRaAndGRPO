---
fileClass: Log
name: FairSpeech full eval supervisor
description: Added supervised full-matrix orchestration for the FairSpeech compression evaluation after the pilot gate.
created: 2026-04-27
updated: 2026-04-30
tags:
  - fairspeech
  - compression
  - inference
  - full-eval
aliases:
  - FairSpeech full evaluation launcher
status: complete
subtype: feature
---

# FairSpeech full eval supervisor

Implemented the post-pilot full-evaluation control layer for `tasks/prd-fairspeech-full-eval-inference.md`.

Key changes:

- Added `scripts/setup/run_fairspeech_full_eval.py`, a CPU-control supervisor for the full FairSpeech compression matrix.
- Added storage and GPU preflight checks for `/`, `/opt`, model cache, result, derived-audio, profile-work, W&B, and pip-cache paths.
- Added full audio generation/validation commands that keep manifests in the repo-side dataset directory while placing generated WAV payloads under `/opt` or another configured volume.
- Added six-variant batch-plan generation and compatibility validation for the selected `160s / max_samples=16` policy.
- Added run-matrix writing, tmux-gated resumable inference execution, per-run logs, JSONL status records, and prediction CSV validation.
- Added recursive prediction loading for full per-model output directories in `scripts/metrics/compute_fairspeech_compression_metrics.py`.
- Added CPU-only tests for external audio roots, full-plan/matrix generation, prediction validation, and recursive metric loading.

Execution follow-up on 2026-04-28:

- Ran the full-eval preflight against the large `/workspace/fairspeech-full-eval` payload root. `/opt` is on the small root overlay in this runtime, so heavy artifacts were intentionally kept under `/workspace`.
- Generated all six full FairSpeech compression audio variants under `/workspace/fairspeech-full-eval/variants/full/audio`, with 26,471 WAV files per variant.
- Wrote the six full manifests under `datasets/fairspeech_compression/manifests` and the prepare summary under `datasets/fairspeech_compression/summaries`.
- Validated count, readability, format, and duration for all six full variant manifests. Exhaustive silence checking was too slow, and sampled silence checking produced false positives on real speech, so the no-silent-file gate remains unclaimed pending validator refinement.
- Regenerated and validated six full batch plans with the selected `160s / max_samples=16` guard. Each plan contains 26,471 rows and 1,866 batches.
- Initial 2026-04-28 pass downloaded and one-file smoke-tested the four then-supported models: `wav2vec2-large`, `whisper-small`, `whisper-medium`, and `whisper-large-v3`.
- Initial 2026-04-28 pass wrote a 24-run inference matrix for the four smoke-passing models across the six variants. This was superseded by the full nine-model gate below.

Execution follow-up on 2026-04-28 for the full nine-model run:

- Installed the Gen 3 runtime stack in the project venv and verified imports for `qwen_asr` and NeMo `SALM`.
- Downloaded the five remaining Gen 3 model snapshots under `/workspace/fairspeech-full-eval/hf-cache/hub`: `qwen3-asr-0.6b`, `qwen3-asr-1.7b`, `canary-qwen-2.5b`, `granite-speech-3.3-2b`, and `granite-speech-3.3-8b`.
- One-file smoke-tested all nine target models. All nine produced non-empty transcripts; the Gen 3 models produced exact smoke transcripts for the selected sample.
- Hardened inference so batch exceptions fail fast by default for all model families; `--allow_batch_failures` is now required to write empty transcripts after batch failures.
- Added Whisper attention-mask passing for padded batched generation after the first full `whisper-small/baseline` attempt emitted a Transformers warning. That partial run was intentionally terminated, its partial CSV removed, and the matrix restarted from the six validated wav2vec2 outputs.
- Wrote and launched the full nine-model, six-variant 54-run matrix in `tmux` session `fairspeech-full-eval`, with a postprocess watcher in `fairspeech-full-eval-post`.
- Checkpoint state at this note: `wav2vec2-large` and `whisper-small` completed and validated across all six variants; `whisper-medium/baseline` was running with the attention-mask fix. Final matrix validation, metrics, and plots were awaiting completion of the remaining inference runs.

Acceleration follow-up on 2026-04-28:

- Generated additional `160s / max_samples=32` batch plans for the six FairSpeech compression variants. These plans keep the same total-duration guard while reducing each variant from 1,866 to 1,353 batches.
- Backed up the original max-16 matrix to `datasets/fairspeech_compression/full_eval/full_eval_run_matrix.max16-backup-20260428T143232Z.json`.
- Updated the active matrix so only `whisper-medium` and `whisper-large-v3` use the max-32 plans; `wav2vec2-large`, Qwen3-ASR, Canary-Qwen, and Granite remain on the proven max-16 plans.
- Let `whisper-medium/bottleneck_12k` finish cleanly, then restarted `fairspeech-full-eval` and `fairspeech-full-eval-post` so the next run picked up the mixed matrix.
- Verified that the restarted active run, `whisper-medium/bottleneck_8k`, is using `fairspeech_bottleneck_8k_total160s_max32_plan.jsonl`; early rows were writing cleanly with no duplicate utterance IDs, no blank hypotheses, and no missing WERs.

Final completion follow-up on 2026-04-30:

- Completed the full nine-model, six-variant FairSpeech compression inference matrix: 54/54 prediction CSVs, 26,471 rows per CSV, 1,429,434 utterance-level predictions total.
- No target model remained excluded after the smoke gate. The models that were not part of the first four-model pass were downloaded, one-file smoke-tested, and then included in the final full matrix: `qwen3-asr-0.6b`, `qwen3-asr-1.7b`, `canary-qwen-2.5b`, `granite-speech-3.3-2b`, and `granite-speech-3.3-8b`.
- Ran final prediction validation. `datasets/fairspeech_compression/full_eval/full_prediction_validation.json` reports `status: pass`, `validated_csvs: 54`, `passed_csvs: 54`, and `failed_csvs: 0`.
- Extended `scripts/metrics/compute_fairspeech_compression_metrics.py` with complete-run filtering and utterance-level bootstrap group WER confidence intervals, then computed final metrics from the complete matrix.
- Final metric artifacts live under `/workspace/fairspeech-full-eval/results/full_metrics/`; the 200-resample bootstrap CI artifacts live under `datasets/fairspeech_compression/full_eval/bootstrap_ci_200/`.
- Generated the final figure set under `/workspace/fairspeech-full-eval/results/full_plots/`: nine model-specific group WER bar charts, one degradation-curve figure, one fairness-gap heatmap, and one insertion-subtype stacked chart. All 12 PNGs were present and nonblank in the final check.
- Updated `docs/project-state.html` to mark FairSpeech compression data collection finished and updated `tasks/prd-fairspeech-full-eval-inference.md` so every execution item is complete.
- Wrote the final result memo at `reports/fairspeech-compression-final-results.md` and the reproducibility note at `reports/fairspeech-full-eval-reproducibility.md`.
