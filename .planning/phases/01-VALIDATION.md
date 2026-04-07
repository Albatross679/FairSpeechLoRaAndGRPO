# Phase 1 (1.0 + 1.1) Validation Map

**Generated:** 2026-04-07
**Test file:** `tests/test_phase1_validation.py`
**Runner:** `pytest tests/test_phase1_validation.py -v`
**Result:** 45 passed, 7 skipped (whisper dep missing)

## Phase 1.0 -- Prepare Fine-Tuning Dataset

| Task ID | Requirement | Test Class | Tests | Command | Status |
|---------|-------------|------------|-------|---------|--------|
| DATA-01 | FS/CV splits with demographic labels | TestASRFairnessDatasetColumnValidation | 4 | `pytest tests/test_phase1_validation.py::TestASRFairnessDatasetColumnValidation -v` | green |
| DATA-01 | Speaker-disjoint stratified splitting | TestSplitFairSpeech | 3 | `pytest tests/test_phase1_validation.py::TestSplitFairSpeech -v` | green |
| DATA-02 | Data loaders yield (audio, transcript, group) batches | TestCollateFn | 4 | `pytest tests/test_phase1_validation.py::TestCollateFn -v` | green |
| DATA-03 | Each demographic group has sufficient representation | TestDemographicStratifiedSampler | 5 | `pytest tests/test_phase1_validation.py::TestDemographicStratifiedSampler -v` | green |
| DATA-03 | Split validation checks detect issues | TestValidateSplitsChecks | 4 | `pytest tests/test_phase1_validation.py::TestValidateSplitsChecks -v` | green |

## Phase 1.1 -- LoRA Prototype

| Task ID | Requirement | Test Class | Tests | Command | Status |
|---------|-------------|------------|-------|---------|--------|
| INFRA-01 | Data collator constants and masking logic | TestDataCollatorConstants, TestFindTranscriptStart, TestPadBatch | 8 | `pytest tests/test_phase1_validation.py::TestDataCollatorConstants tests/test_phase1_validation.py::TestFindTranscriptStart tests/test_phase1_validation.py::TestPadBatch -v` | green |
| INFRA-02 | LoRA hyperparameters match design docs | TestLoraPrototypeConstants | 6 | `pytest tests/test_phase1_validation.py::TestLoraPrototypeConstants -v` | green |
| INFRA-02 | LoRA targets decoder attention only (not MLP) | TestLoraTargetModules | 1 | `pytest tests/test_phase1_validation.py::TestLoraTargetModules -v` | green |
| INFRA-03 | Stratified subset creation (100 FS + 100 CV) | TestCreateStratifiedSubset | 2 | `pytest tests/test_phase1_validation.py::TestCreateStratifiedSubset -v` | green |
| INFRA-03 | VRAM profiling handles no-GPU | TestPrintGpuMemory | 1 | `pytest tests/test_phase1_validation.py::TestPrintGpuMemory -v` | green |
| INFRA-04 | Validation thresholds (VRAM, WER, chatty) | TestValidateLoraPrototypeConstants | 3 | `pytest tests/test_phase1_validation.py::TestValidateLoraPrototypeConstants -v` | skipped (whisper) |
| INFRA-04 | Loss trend detection | TestCheckLossTrend | 3 | `pytest tests/test_phase1_validation.py::TestCheckLossTrend -v` | skipped (whisper) |
| INFRA-05 | Forward patch safety (raises on bad model) | TestPatchOuterForward | 3 | `pytest tests/test_phase1_validation.py::TestPatchOuterForward -v` | green |

## Import Smoke Tests

| Scope | Test Class | Tests | Command | Status |
|-------|------------|-------|---------|--------|
| All Phase 1 modules importable | TestImports | 5 (1 skipped) | `pytest tests/test_phase1_validation.py::TestImports -v` | green (4/5) |

## Skipped Tests (Environment Gaps)

7 tests skipped because `whisper` (openai-whisper) is not installed in this environment.
These tests cover `scripts/training/validate_lora_prototype.py` which has `from whisper.normalizers import EnglishTextNormalizer` at module level.

**To run skipped tests:** `pip install openai-whisper && pytest tests/test_phase1_validation.py -v`

## Remaining HPC/GPU-Only Validation (Cannot Automate Here)

These require real data on HPC and/or GPU:
1. End-to-end `prepare_splits.py` on real Fair-Speech data
2. End-to-end `validate_splits.py` on generated splits
3. Data loader smoke test with real audio files
4. LoRA prototype training on GPU (100 steps)
5. VRAM budget validation on RTX A4000
6. Transcription quality + adapter round-trip checks
