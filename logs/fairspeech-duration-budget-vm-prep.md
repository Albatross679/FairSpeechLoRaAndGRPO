---
fileClass: Log
name: FairSpeech duration-budget VM prep
description: Updated FairSpeech compression batching to learn total-duration budgets under a 75% VRAM threshold before VM pilot inference.
created: 2026-04-27
updated: 2026-04-27
tags:
  - fairspeech
  - compression
  - batching
  - vm-setup
aliases:
  - FairSpeech total-duration batching
status: complete
subtype: setup
---

# FairSpeech duration-budget VM prep

Updated the VM preparation plan and tooling so pilot-gate profiling uses summed audio duration as the primary batch budget instead of a fixed sample count or padded-seconds budget.

Key decisions:

- Use `sum_duration_seconds <= selected_total_duration_seconds` as the primary pre-splitting rule.
- Preserve padded-audio seconds in batch-plan records as diagnostic metadata.
- Use 75% VRAM as the profiling safety threshold.
- Keep `max_samples` only as a secondary per-example overhead guard.
- Added `scripts/setup/run_fairspeech_vm_prep.sh` to execute the supervised VM pilot gate in `tmux`, keep model caches/results under `/opt`, record all nine model download/smoke statuses, profile smoke-passing models, and stop after the `wav2vec2-large` six-variant pilot.
- Corrected Hugging Face cache placement to use `$HF_HOME/hub`/`$HF_HUB_CACHE` for snapshots; passing `$HF_HOME` directly caused duplicate model copies under `/opt/hf-cache`.
- Capped `transformers` to `<5` because the VM has torch 2.4.1, and transformers 5 blocks legacy `.bin` weight loading unless torch is at least 2.6.
- Added a resumable `START_STEP` guard plus failed-download cache cleanup after `whisper-large-v3` filled the 50 GiB root volume during VM prep. Profiling and pilot inference now fail fast if root free space is below the configured 12 GiB floor.
- Added a `--min-group-size` metric override and use `--min-group-size 1` for the 28-row pilot gate so pilot metrics and plots are emitted; full evaluation keeps the default 50-row demographic floor.

This supports the FairSpeech compression plan in `docs/project-state.html`, where full inference should be pre-split into duration-aware batches after real VM profiling.
