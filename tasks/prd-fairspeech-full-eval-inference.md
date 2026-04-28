---
fileClass: Task
name: FairSpeech Full Evaluation Inference Plan
description: Hierarchical implementation checklist for running the full 9 model x 6 variant FairSpeech compression evaluation after the pilot gate.
created: 2026-04-27
updated: 2026-04-28
tags:
  - fairspeech
  - inference
  - full-eval
  - compression
aliases:
  - FairSpeech full eval plan
  - Full FairSpeech inference runbook
status: in-progress
---

# FairSpeech Full Evaluation Inference Plan

Source reference: [project-state.html](../docs/project-state.html#chapter-plan), especially Chapter 2 "FairSpeech compression plan".

## Objective

Run the complete controlled FairSpeech input-audio compression evaluation:

- 9 ASR models.
- 6 audio variants.
- 26,471 FairSpeech utterances per run.
- 54 prediction CSVs.
- 1,429,434 utterance-level predictions.
- Final metric tables, figures, and result memo.

This is an evaluation/inference pipeline only. It must not be mixed with fine-tuning or training work.

## Current State

- [x] Full FairSpeech baseline manifest validates with 26,471 rows.
- [x] All six full variant manifests exist with 26,471 rows each.
- [x] Pilot audio for six variants was generated and validated under `/opt/fairspeech-variants/pilot`.
- [x] Pilot gate completed for `wav2vec2-large` across all six variants.
- [x] `wav2vec2-large`, `whisper-small`, and `whisper-medium` downloaded and smoke-tested.
- [x] Pilot duration profiling selected `160s` total duration with `max_samples=16` for smoke-passing models.
- [x] Full-run supervisor support added in `scripts/setup/run_fairspeech_full_eval.py` for disk/GPU preflight, full audio generation/validation, six guarded plans, run-matrix writing, tmux-gated resumable inference, prediction validation, metrics, and plots. No full heavy jobs were launched by this implementation pass.
- [x] Full derived audio for all six variants generated under `/workspace/fairspeech-full-eval/variants/full`.
- [x] Full six-variant batch plans regenerated with the selected `160s / 16-sample` guard.
- [x] Full nine-model cache and one-file smoke gate completed. 2026-04-28 note: all nine target models passed the one-file smoke transcript gate after installing the Gen 3 runtime stack under the project venv and storing model snapshots under `/workspace/fairspeech-full-eval/hf-cache`.
- [ ] Full nine-model, 54-run inference is launched and in progress under `tmux` session `fairspeech-full-eval`.
  - 2026-04-28 note: the full 54-run matrix at `datasets/fairspeech_compression/full_eval/full_eval_run_matrix.json` is running with `160s / max_samples=16` plans and runtime roots under `/workspace/fairspeech-full-eval`.
  - 2026-04-28 note: `wav2vec2-large` and `whisper-small` completed and validated across all six variants. A partial first `whisper-small/baseline` attempt was intentionally stopped after adding Whisper attention-mask support, then restarted cleanly. `whisper-medium/baseline` is currently running.

## Acceptance Criteria

- [x] Agents update this checklist as they work: mark completed items with `[x]`, leave future work as `[ ]`, and add a short blocked note under any item that cannot proceed.
- [x] Runtime storage is provisioned so model cache, derived audio, predictions, and logs do not fill root.
- [x] All nine model cache records end in `downloaded` plus smoke status `passed`, or have a documented intentional exclusion.
- [x] Six full variant audio folders validate at 26,471 files each.
- [x] Six full variant manifests validate at 26,471 rows each.
- [x] Full batch plans exist for each variant using `sum_duration_seconds <= 160` and `max_samples <= 16`.
- [ ] All planned model x variant inference runs complete or have documented retry/failure records.
- [ ] There are 54 prediction CSVs unless a model is intentionally excluded.
- [ ] Each completed prediction CSV has 26,471 rows.
- [ ] Metrics and plots are generated from the complete matrix.
- [ ] Final result memo separates sample-rate bottlenecks from MP3 codec artifacts.

## Hierarchical Action Items

- [ ] 0. Agent progress protocol
  - [x] 0.1. Before starting work, read this plan and identify the highest-priority unchecked item that matches the current task.
  - [x] 0.2. As work completes, update this file in the same turn.
    - [ ] Change completed items from `[ ]` to `[x]`.
    - [ ] Keep incomplete future work unchecked.
    - [ ] Add concise dated notes below blocked items instead of silently leaving them ambiguous.
  - [ ] 0.3. When adding new required work, add it as a checklist item in the relevant section.
  - [ ] 0.4. At handoff, summarize which checklist items changed and where the next agent should resume.

- [x] 1. Prepare runtime storage
  - [x] 1.1. Provision enough non-root storage for the full run.
    - [x] Confirm expected available space before downloads and derived audio generation.
    - [x] Prefer cache/results locations under `/opt` or an attached volume, not `/workspace`.
      - 2026-04-28 note: `/opt` is on the 20 GiB root overlay in this runtime, so heavy artifacts were placed under the large mounted `/workspace/fairspeech-full-eval` volume instead.
    - [x] Keep root free space above the 12 GiB guard throughout the run.
  - [x] 1.2. Set and verify runtime paths.
    - [x] Set `HF_HOME` and `HF_HUB_CACHE` to the large cache location.
    - [x] Set `TRANSFORMERS_CACHE` only as a compatibility alias to the same hub cache.
    - [x] Set derived audio root, profile work root, result root, W&B dir, and pip cache under non-root-heavy storage.
  - [x] 1.3. Add a preflight disk audit command to the full-run launcher.
    - [x] Check `/`, `/opt`, cache path, result path, and derived-audio path.
    - [ ] Stop before downloads if root free space is below 12 GiB.
      - 2026-04-27 note: `scripts/setup/run_fairspeech_full_eval.py preflight` blocks the full-eval path on low disk; model-cache download retries still live in the pilot/cache prep path.
    - [x] Stop before inference if prediction or temporary-work paths are near capacity.

- [x] 2. Complete model cache and smoke gate
  - [x] 2.1. Preserve known-good cache entries.
    - [x] Keep `wav2vec2-large`.
    - [x] Keep `whisper-small`.
    - [x] Keep `whisper-medium`.
  - [x] 2.2. Retry blocked models only after storage is fixed.
    - [x] Retry `whisper-large-v3`.
    - [x] Download and smoke `qwen3-asr-0.6b`.
    - [x] Download and smoke `qwen3-asr-1.7b`.
    - [x] Download and smoke `canary-qwen-2.5b`.
    - [x] Download and smoke `granite-speech-3.3-2b`.
    - [x] Download and smoke `granite-speech-3.3-8b`.
  - [ ] 2.3. Record cache metadata for every model.
    - [ ] Model key.
    - [ ] Hugging Face model ID.
    - [ ] Snapshot path.
    - [ ] Revision or commit hash if available.
    - [ ] Snapshot size.
    - [ ] Required runtime package versions.
    - [ ] Download status.
    - [ ] Smoke status.
    - [ ] Error text for failures.
  - [x] 2.4. Do not proceed to full inference until each included model has a one-file smoke transcript.
    - 2026-04-28 note: one-file smoke transcripts exist for all nine included models: `wav2vec2-large`, `whisper-small`, `whisper-medium`, `whisper-large-v3`, `qwen3-asr-0.6b`, `qwen3-asr-1.7b`, `canary-qwen-2.5b`, `granite-speech-3.3-2b`, and `granite-speech-3.3-8b`.

- [ ] 3. Generate full variant audio
  - [x] 3.1. Generate all six full FairSpeech audio variants.
    - [x] `baseline`.
    - [x] `bottleneck_12k`.
    - [x] `bottleneck_8k`.
    - [x] `mp3_64k`.
    - [x] `mp3_32k`.
    - [x] `mp3_16k`.
  - [x] 3.2. Store runtime-heavy audio under `/opt` or attached storage.
    - [x] Keep source metadata and manifests in the repo-side dataset area.
    - [x] Keep generated WAV payloads out of tracked workspace paths.
    - [x] 2026-04-27 implementation support: `prepare_fairspeech_compression.py --audio-output-dir` lets manifests remain under the repo-side dataset directory while generated WAV payloads live under `/opt` or another attached volume.
  - [ ] 3.3. Validate generated audio.
    - [x] Confirm 26,471 files per variant.
    - [x] Confirm 16 kHz mono PCM16 WAV after every variant transform.
    - [x] Confirm durations match the manifest within tolerance.
    - [ ] Confirm no missing, unreadable, or silent files.
      - 2026-04-28 note: no missing/unreadable files and no duration/format failures. Exhaustive silence checking was too slow; sampled silence checking produced false positives on real speech and needs refinement before claiming a no-silent-file gate.

- [x] 4. Regenerate full batch plans with selected guard
  - [x] 4.1. Use total-duration batching as the primary rule.
    - [x] `sum_duration_seconds <= 160`.
    - [x] `max_samples <= 16`.
    - [x] Keep `padded_audio_seconds` as diagnostic metadata.
  - [x] 4.2. Generate one full batch plan per audio variant.
    - [x] Baseline full plan with selected guard.
    - [x] 12 kHz bottleneck full plan.
    - [x] 8 kHz bottleneck full plan.
    - [x] MP3 64 kbps full plan.
    - [x] MP3 32 kbps full plan.
    - [x] MP3 16 kbps full plan.
  - [x] 4.3. Summarize each plan.
    - [x] Number of rows.
    - [x] Number of batches.
    - [x] Median samples per batch.
    - [x] Max samples per batch.
    - [x] Max summed duration.
    - [x] Max padded seconds.
    - [x] Duration bucket counts.
  - [x] 4.4. Check plan compatibility before inference.
    - [x] Every manifest row appears exactly once.
    - [x] No batch exceeds 160 total seconds.
    - [x] No batch exceeds 16 samples.
    - [x] Row identity maps by `utterance_id` for resume safety.
    - [x] 2026-04-27 implementation support: the full-run supervisor validates row coverage, max summed duration, max samples, duplicate IDs, and unknown IDs before writing/running the matrix.

- [x] 5. Define full inference run matrix
  - [x] 5.1. Create a machine-readable run matrix.
    - [x] 9 model keys.
    - [x] 6 audio variants.
    - [x] Manifest path per variant.
    - [x] Batch plan path per variant.
    - [x] Output directory per model.
    - [x] Log path per model x variant.
    - [x] 2026-04-27 implementation support: `scripts/setup/run_fairspeech_full_eval.py write-matrix` creates this matrix after smoke-passing models and six validated batch plans exist.
    - [x] 2026-04-28 full matrix: wrote the nine-model, six-variant 54-run matrix.
  - [x] 5.2. Mark model inclusion status.
    - [x] Include only models with passed smoke tests.
    - [x] If a model remains excluded, document why before running metrics.
      - 2026-04-28 note: no target model is excluded; all nine passed smoke.
  - [ ] 5.3. Preserve decoding settings.
    - [ ] Keep decoding fixed per model across all six variants.
    - [ ] Do not tune decoding on compressed variants.
    - [ ] Record device, batch plan, model revision, and runtime package versions in run metadata.

- [ ] 6. Execute full inference safely
  - [ ] 6.1. Run long commands inside `tmux`.
    - [x] Create one supervised session for the full matrix.
    - [x] Write a top-level run log.
    - [ ] Poll status JSON and logs until complete or blocked.
    - [x] 2026-04-27 implementation support: `run-matrix` refuses to run outside `tmux` by default, writes per-run logs, and appends a JSONL status ledger.
  - [ ] 6.2. Use GPU preflight before each run.
    - [ ] Run `nvidia-smi`.
    - [ ] Stop if GPU memory usage is above 80%.
    - [ ] Use `CUDA_VISIBLE_DEVICES` if multiple GPUs are present.
    - [x] 2026-04-27 implementation support: `preflight` and `run-matrix` query `nvidia-smi`, enforce the 80% default threshold, and propagate `CUDA_VISIBLE_DEVICES`.
  - [ ] 6.3. Run each model x variant with resume enabled.
    - [ ] Use `scripts/inference/run_inference.py`.
    - [ ] Pass `--audio_variant`.
    - [ ] Pass the selected full batch plan.
    - [ ] Pass `--resume`.
    - [ ] Flush predictions after every batch.
  - [ ] 6.4. Validate each prediction CSV immediately after completion.
    - [ ] File exists.
    - [ ] Row count is 26,471.
    - [ ] `utterance_id` coverage matches manifest.
    - [ ] Transcript column is not globally empty.
    - [ ] Per-utterance WER values are present or recomputable.
    - [ ] Metadata fields include model and variant.
    - [x] 2026-04-27 implementation support: the full-run supervisor validates each CSV after a run and records pass/fail state in the status ledger.
  - [ ] 6.5. Track failure and retry state.
    - [x] Record failed batch or run.
    - [x] Keep stderr/stdout logs.
    - [x] Retry with the same output directory and resume flag.
    - [ ] If OOM occurs, lower max samples first, then lower total-duration budget.
    - 2026-04-28 note: the first `whisper-small/baseline` full attempt was intentionally terminated after a fresh Transformers warning showed missing attention masks for padded Whisper batches. `scripts/inference/run_inference.py` now requests/passes Whisper attention masks, the partial CSV was removed, and the full matrix was relaunched; completed wav2vec2 runs were skipped by validation/resume.

- [ ] 7. Compute metrics
  - [ ] 7.1. Verify matrix completeness before metrics.
    - [ ] Confirm all expected prediction CSVs are present.
    - [ ] Confirm each included CSV has 26,471 rows.
    - [ ] Confirm baseline exists for every included model.
    - [x] 2026-04-27 implementation support: `validate-predictions` checks the matrix before the `all` command proceeds to metrics.
  - [ ] 7.2. Compute FairSpeech compression metrics.
    - [ ] WER by ethnicity.
    - [ ] MMR.
    - [ ] Relative WER gap.
    - [ ] Absolute WER gap.
    - [ ] Paired delta WER versus baseline.
    - [ ] Insertion rate.
    - [ ] Insertion subtypes: repetition, syntactic/function-word, content/semantic.
  - [ ] 7.3. Use full-evaluation group thresholds.
    - [ ] Keep the default 50-row minimum group size for final metrics.
    - [ ] Do not use the pilot-only `--min-group-size 1` override for final claims.
  - [ ] 7.4. Export metric tables.
    - [ ] Group metrics CSV.
    - [ ] Fairness metrics CSV.
    - [ ] Paired delta CSV.
    - [ ] Summary JSON.

- [ ] 8. Generate figures
  - [ ] 8.1. Generate core figures.
    - [ ] Ethnicity WER bars.
    - [ ] Degradation curves.
    - [ ] Fairness-gap heatmaps.
    - [ ] Insertion-subtype stacked charts.
  - [ ] 8.2. Check figure quality.
    - [ ] Axis labels are readable.
    - [ ] Model and variant labels are consistent.
    - [ ] Sample-rate bottleneck variants and MP3 variants are visually separable.
    - [ ] Values match exported metric tables.

- [ ] 9. Write result memo
  - [ ] 9.1. Summarize completion state.
    - [ ] Models included.
    - [ ] Models excluded, if any.
    - [ ] Number of completed runs.
    - [ ] Number of prediction rows.
  - [ ] 9.2. Interpret sample-rate bottlenecks separately.
    - [ ] Baseline versus 12 kHz bottleneck.
    - [ ] Baseline versus 8 kHz bottleneck.
    - [ ] Uniform degradation versus ethnicity-specific amplification.
  - [ ] 9.3. Interpret MP3 codec artifacts separately.
    - [ ] Baseline versus 64 kbps MP3.
    - [ ] Baseline versus 32 kbps MP3.
    - [ ] Baseline versus 16 kbps MP3.
    - [ ] Uniform degradation versus ethnicity-specific amplification.
  - [ ] 9.4. Tie results back to the compression hypothesis.
    - [ ] Compare older encoder-only, Whisper, and newer LLM-ASR families.
    - [ ] Avoid final ethnicity claims until the full matrix is complete.
    - [ ] Document limitations and failed models.

- [ ] 10. Archive and reproducibility
  - [ ] 10.1. Preserve run metadata.
    - [ ] VM hardware.
    - [ ] GPU memory.
    - [ ] Branch and commit.
    - [ ] Model cache manifest.
    - [ ] Batch plan summaries.
    - [ ] Runtime package versions.
  - [ ] 10.2. Preserve logs.
    - [ ] Download logs.
    - [ ] Smoke logs.
    - [ ] Inference logs.
    - [ ] Metrics logs.
    - [ ] Plot logs.
  - [ ] 10.3. Maintain this plan as the execution ledger.
    - [ ] Check off completed items as agents progress.
    - [ ] Add brief blocker notes with date, command/log path, and next action.
    - [ ] Update the `updated` frontmatter date when substantive checklist changes are made.
  - [ ] 10.4. Keep heavy artifacts out of Git.
    - [ ] Audio payloads under `/opt` or ignored storage.
    - [ ] Prediction payloads under `/opt` or ignored storage.
    - [ ] Only commit scripts, small manifests/summaries, task files, and logs appropriate for the repo.

## Suggested Execution Order

- [ ] First: fix storage and model cache capacity.
- [ ] Second: finish model downloads and one-file smoke tests.
- [ ] Third: generate and validate full derived audio.
- [ ] Fourth: regenerate six full batch plans with `160s / max_samples=16`.
- [ ] Fifth: run the full model x variant inference matrix in `tmux`.
- [ ] Sixth: validate all prediction CSVs.
- [ ] Seventh: compute metrics and generate plots.
- [ ] Eighth: write the result memo.

## Known Blockers

- [ ] Root volume is too small for the full nine-model cache. `whisper-large-v3` already failed after filling the 50 GiB root volume.
- [ ] Full Gen 3 model cache and smoke tests are still unproven in this VM environment.
- [ ] Full derived audio generation may require substantially more disk than the pilot.
- [ ] Full evaluation should not start until storage paths and disk guards are confirmed.
