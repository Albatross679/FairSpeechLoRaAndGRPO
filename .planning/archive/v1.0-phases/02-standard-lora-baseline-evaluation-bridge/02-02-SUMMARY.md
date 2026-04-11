---
phase: 02-standard-lora-baseline-evaluation-bridge
plan: 02
status: complete
started: 2026-04-07
completed: 2026-04-07
---

# Plan 02-02 Summary: Evaluation Bridge

## What was delivered

Standalone evaluation script that loads a LoRA adapter, runs inference on Fair-Speech and Common Voice, and computes per-group WER and fairness metrics.

## Key results

- **Fair-Speech WER:** 2.11% (199 utterances, ethnicity axis)
- **Common Voice WER:** 2.90% (500 utterances, accent axis)
- **Per-group WER ranges:** FS 0-3.1%, CV 0-9.1%
- **Fairness metrics:** Not computed — prototype sample sizes too small (no group ≥ 50). Full-scale evaluation needed.

## Critical fix: generation prompt prefix

LoRA-adapted model required `language English<asr_text>` prefix in the generation prompt. Without it, the model outputs `<|im_end|>` immediately. Root cause: the adapter was trained with prefix masking (labels=-100 for prefix tokens), so it only learned to generate transcript tokens after `<asr_text>`.

## Artifacts

| Path | Description |
|------|-------------|
| `scripts/training/evaluate_adapter.py` | Evaluation bridge script |
| `outputs/standard-lora/eval/predictions_standard-lora_fairspeech.csv` | FS prediction CSV |
| `outputs/standard-lora/eval/predictions_standard-lora_commonvoice.csv` | CV prediction CSV |
| `outputs/standard-lora/eval/analysis_standard-lora.json` | Per-group fairness analysis JSON |

## Deviations

- **Prototype-scale evaluation:** Used eval subset (199 FS) + limited CV dev (500) instead of full datasets. Groups too small for fairness metric computation.
- **Bootstrap CIs skipped:** `--skip_bootstrap` used for speed. Can rerun with CIs on full data.

## Requirements status

- **EVAL-01 (FS evaluation):** Partial — script works, needs full-scale run for fairness metrics
- **EVAL-02 (CV evaluation):** Partial — same as above
- **EVAL-03 (fairness metrics):** Script computes them, but groups < MIN_GROUP_SIZE at prototype scale
- **EVAL-04 (both axes):** ✓ Both ethnicity and accent axes evaluated
