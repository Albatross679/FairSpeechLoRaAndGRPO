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

This supports the FairSpeech compression plan in `docs/project-state.html`, where full inference should be pre-split into duration-aware batches after real VM profiling.
