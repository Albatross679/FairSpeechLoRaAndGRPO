---
phase: 02-standard-lora-baseline-evaluation-bridge
type: verification
created: 2026-04-07
updated: 2026-04-07
method: retroactive-reconstruction + gap-bridging
---

# Phase 2 Verification: Standard LoRA Baseline + Evaluation Bridge

## Verdict: PASS (79 tests passing, 1 xfail — data limitation)

```bash
pytest tests/test_phase02_validation.py -v   # 79 passed, 1 xfailed in 1.12s
```

## Gap Bridging Summary

| Gap | Before | After | How |
|-----|--------|-------|-----|
| Fairness metrics | Not computable (groups < 50) | **PASS** — FS 7 groups, CV 7 groups, all >= 50 | Reran eval on larger subsets |
| Bootstrap CIs | Skipped | **PASS** — 14 groups with 95% CIs | Enabled `--n_bootstrap 1000` |
| LibriSpeech eval | Skipped (no manifest) | **PASS** — 2,620 utterances, WER 1.77% | Generated LS manifest from test-clean |
| Speaker-disjoint (CV) | No speaker_id in manifest | **PASS** — `client_id` preserved as `speaker_id` | Updated `generate_manifests.py` |
| Speaker-disjoint (FS) | No speaker_id | **XFAIL** — data limitation | Fair-Speech has no speaker identifier |

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| BASE-01 | **PASS** | Adapter at `outputs/standard-lora/adapter/adapter_model.safetensors` (8.6 MB), loss 2.31 → 0.27, VRAM peak 9.97 GB |
| EVAL-01 | **PASS** | Fair-Speech eval: WER 3.24%, 799 utterances, 7 ethnicity groups with CIs |
| EVAL-02 | **PASS** | Common Voice eval: WER 2.66%, 632 utterances, 7 accent groups with CIs |
| EVAL-03 | **PASS** | Fairness metrics computed: FS max-min 3.43, gap 242.8%; CV max-min 3.69, gap 269.5% |
| EVAL-04 | **PASS** | Both ethnicity (FS) and accent (CV) axes evaluated |

## Baseline Fairness Results

### Fair-Speech (ethnicity axis)

| Ethnicity | WER | N | 95% CI |
|-----------|-----|---|--------|
| Hispanic | 1.52% | 79 | [0.41%, 3.09%] |
| Asian | 1.67% | 109 | [0.69%, 2.90%] |
| White | 2.19% | 159 | [1.40%, 3.10%] |
| Native American | 3.36% | 131 | [1.80%, 5.46%] |
| Pacific Islander | 3.38% | 50 | [1.58%, 5.62%] |
| Middle Eastern | 3.59% | 50 | [1.49%, 5.57%] |
| Black/AA | 5.21% | 221 | [4.05%, 6.57%] |

**Fairness: max-min ratio 3.43, relative gap 242.8%, WER std 0.012**

### Common Voice (accent axis)

| Accent | WER | N | 95% CI |
|--------|-----|---|--------|
| us | 1.65% | 100 | [0.73%, 2.61%] |
| Northumbrian | 1.71% | 51 | [0.77%, 2.69%] |
| african | 2.04% | 81 | [0.79%, 3.44%] |
| australia | 2.13% | 100 | [1.16%, 3.42%] |
| canada | 2.29% | 100 | [1.08%, 3.71%] |
| england | 2.45% | 100 | [1.34%, 3.66%] |
| indian | 6.11% | 100 | [3.89%, 8.48%] |

**Fairness: max-min ratio 3.69, relative gap 269.5%, WER std 0.014**

### LibriSpeech (test-clean, no demographic axis)

| Dataset | WER | N |
|---------|-----|---|
| test-clean | 1.77% | 2,620 |

## Artifact Verification

| Artifact | Exists | Valid |
|----------|--------|-------|
| `outputs/hp-sweep/best_params.json` | Yes | Valid JSON with HP keys |
| `outputs/hp-sweep/top3_configs.json` | Yes | 3 entries with eval_loss |
| `outputs/hp-sweep/all_trials.csv` | Yes | 20 rows |
| `outputs/standard-lora/locked_config.json` | Yes | Valid HP config |
| `outputs/standard-lora/adapter/adapter_model.safetensors` | Yes | 8.6 MB |
| `outputs/standard-lora/adapter/adapter_config.json` | Yes | Valid PEFT config |
| `outputs/standard-lora/training_config.json` | Yes | Loss + VRAM metrics |
| `outputs/standard-lora/eval/analysis_standard-lora.json` | Yes | 3 datasets, fairness metrics |
| `outputs/standard-lora/eval/predictions_standard-lora_fairspeech.csv` | Yes | 799 rows |
| `outputs/standard-lora/eval/predictions_standard-lora_commonvoice.csv` | Yes | 632 rows |
| `outputs/standard-lora/eval/predictions_standard-lora_librispeech.csv` | Yes | 2,620 rows |
| `outputs/manifests/ls_test_clean.csv` | Yes | 2,620 rows, 40 speakers |
| `scripts/training/train_standard_lora.py` | Yes | Speaker-disjoint split with GroupShuffleSplit |
| `scripts/training/evaluate_adapter.py` | Yes | 18 KB, all features |
| `scripts/training/generate_manifests.py` | Yes | CV speaker_id, LS manifest |

## Remaining Deviation

| Deviation | Impact | Resolution |
|-----------|--------|------------|
| Fair-Speech has no speaker_id | Potential train/eval speaker overlap for FS data | True dataset limitation — hash_name is 1:1 per utterance, not a speaker identifier. Random split used with warning. |

## Test Suite

```
tests/test_phase02_validation.py
├── TestHPSweepArtifacts (7 tests)           — sweep outputs valid
├── TestTrainScriptCodePatterns (9 tests)     — train_standard_lora.py code quality
├── TestValidationAndLockedConfig (5 tests)   — validation results + locked config
├── TestAdapterArtifacts (5 tests)            — adapter files valid
├── TestTrainingConfig (5 tests)              — training metrics valid
├── TestEvalScriptCodePatterns (7 tests)      — evaluate_adapter.py code quality
├── TestPredictionCSVs (21 tests)             — prediction CSV format (3 datasets × 7)
├── TestAnalysisJSON (12 tests)               — analysis JSON structure + fairness metrics
├── TestBridgedGaps (7 tests)                 — previously-xfailed gaps now passing
└── TestRemainingDeviations (1 xfail test)    — FS speaker_id data limitation
```

## Code Changes for Gap Bridging

| File | Change |
|------|--------|
| `scripts/training/generate_manifests.py` | Added `speaker_id` from `client_id` to CV manifests; added LibriSpeech `test-clean` manifest generation |
| `scripts/training/train_standard_lora.py` | Updated `create_speaker_disjoint_split()` to handle mixed speaker_id (CV has it, FS doesn't) with pseudo-IDs |
| `tests/test_phase02_validation.py` | Converted 3 xfails to passing tests in `TestBridgedGaps`; added LS prediction CSV fixture; 1 xfail remains |
