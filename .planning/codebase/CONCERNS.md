# Codebase Concerns

**Analysis Date:** 2026-04-05

## Tech Debt

### Hardcoded Absolute Paths Throughout Codebase

**Issue:** Hardcoded user-specific paths embedded in 20+ scripts make them non-portable and unsuitable for shared environments, clusters, or different users.

**Files affected:**
- `scripts/run_inference.py` (lines 94-96): `/users/PAS2030/srishti/asr_fairness/data`, `/users/PAS2030/srishti/asr_fairness/results/commonvoice`, `/users/PAS2030/srishti/asr_fairness/perturbed_audio`
- `scripts/compute_fairness_metrics.py` (lines 84, 90): `/users/PAS2030/srishti/asr_fairness/results/commonvoice`, `/users/PAS2030/srishti/asr_fairness/results/analysis`
- `scripts/compute_fairness_metrics_fs.py` (lines 73, 75): `/users/PAS2030/srishti/asr_fairness/results/fairspeech`
- `scripts/error_decomposition.py` (lines 20-21): `/users/PAS2030/srishti/asr_fairness/results/commonvoice`, `/users/PAS2030/srishti/asr_fairness/results/commonvoice/analysis`
- `scripts/error_decomposition_fs.py` (lines 37, 39): `/users/PAS2030/srishti/asr_fairness/results/fairspeech`
- `scripts/generate_perturbations.py` (line 36): `/users/PAS2030/srishti/asr_fairness`
- `scripts/generate_all_plots.py` (lines 102-108): `/users/PAS2030/srishti/asr_fairness/results/`
- `scripts/prepare_dataset.py` (lines 19-20): `/users/PAS2030/srishti/bootcamp/data/commonvoice/`, `/users/PAS2030/srishti/asr_fairness/data`
- `scripts/prepare_fairspeech.py` (lines 19, 21): `/users/PAS2030/srishti/asr_fairness/data/fairspeech`
- `scripts/prepare_librispeech.py` (line 14): `/users/PAS2030/srishti/asr_fairness`
- `scripts/extract_test_clips.py` (lines 13, 14, 16): `/users/PAS2030/srishti/bootcamp/data/commonvoice/`, `/users/PAS2030/srishti/asr_fairness/`
- `scripts/prepare_h1_datasets.py` (lines 8, 36, 37): `/users/PAS2030/srishti/asr_fairness/data/h1`, source data paths
- Plus: `validate_*_test.py`, `download_*.sh`, `launch_*.sh`, `parse_bootstrap_cis.py`, `regenerate_figures_large_fonts.py`, `whisper_hallucination_analysis.py`, `prepare_overleaf.py`

**Impact:** Scripts fail to run without manual path editing. CI/CD impossible. Collaboration blocked. Code is environment-bound.

**Fix approach:** 
1. Create a centralized config file (e.g., `config.yaml` or `.env.example`) with all path variables
2. Update all scripts to read paths from environment variables with fallback to config file
3. Document required paths in README
4. Use relative paths where possible (e.g., project root + relative dirs)

---

### Code Duplication Across Analysis Scripts

**Issue:** Core analysis functions are duplicated across multiple similar scripts instead of being shared, making maintenance harder and introducing inconsistency risks.

**Duplicated functions:**
- `compute_group_wer()` appears in `compute_fairness_metrics.py` and `compute_fairness_metrics_fs.py` with nearly identical logic but minor parameter differences
- `bootstrap_wer()` appears in both files with identical implementation
- `compute_fairness_metrics()` appears in both with similar MMR/gap calculation logic
- `parse_args()` pattern repeated across many scripts with hardcoded defaults
- `load_audio()` function appears in both `run_inference.py` and `generate_perturbations.py`

**Files affected:**
- `scripts/compute_fairness_metrics.py` (lines 109-195): WER and fairness metric calculations
- `scripts/compute_fairness_metrics_fs.py` (lines 82-161): Nearly identical implementations

**Impact:** Changes to metric computation require updates in multiple places. Risk of divergence and inconsistent results. Harder to test and verify correctness.

**Fix approach:**
1. Extract shared functions to `scripts/lib/analysis.py` or `scripts/utils.py`
2. Create a shared module: `metrics.py` with `compute_group_wer()`, `bootstrap_wer()`, `compute_fairness_metrics()`
3. Update imports in both CV and Fair-Speech scripts
4. Use parametrized versions to handle both datasets (demographic axes, group ordering, etc.)

---

### Bare Exception Handlers Hiding Errors

**Issue:** Many except blocks catch all exceptions silently or with minimal logging, making debugging difficult and hiding bugs.

**Files affected:**
- `scripts/run_inference.py` (lines 176, 225, 286, 361, 422, 521): `except Exception` → print warning, continue silently
- `scripts/whisper_hallucination_analysis.py` (lines 88, 196): `except Exception` → return empty result or continue
- `scripts/compute_fairness_metrics.py` (line 162): `except Exception` → `continue` in bootstrap loop
- `scripts/generate_perturbations.py` (lines 88-96): Falls back between librosa and torchaudio without clear error tracking

**Impact:** Silent failures in audio loading, inference batches, or metric computation can produce invalid results without warning. Error propagation is broken. Hard to diagnose production issues.

**Fix approach:**
1. Replace `except Exception` with specific exception types: `except (IOError, RuntimeError, ValueError) as e`
2. Add detailed logging with traceback: `logging.error(f"Failed to load {path}", exc_info=True)`
3. Return sentinel values with metadata: `{"result": None, "error": str(e), "file": path}`
4. Add a logging setup at module level: `import logging; log = logging.getLogger(__name__)`

---

## Missing Error Handling & Validation

### Inadequate Input Validation in Data Preparation

**Issue:** Scripts don't validate critical input data assumptions before processing, risking corrupt output.

**Files affected:**
- `scripts/prepare_dataset.py` (lines 82-88): Only checks audio file existence, doesn't validate:
  - Audio format (duration, sample rate after loading)
  - Reference transcript content (empty strings allowed)
  - Demographic label presence/validity
- `scripts/prepare_fairspeech.py` (lines 94-95): Skips missing audio without checking if remaining samples are sufficient for analysis
- `scripts/run_inference.py` (line 674): Loads manifest CSV without schema validation (missing column checks)

**Impact:** Invalid datasets slip through, causing failures downstream. Demographic subgroups may fall below MIN_GROUP_SIZE silently.

**Fix approach:**
1. Add schema validation: `assert all(col in df.columns for col in required_cols)`
2. Validate demographic coverage: Check that each demographic axis has minimum required samples
3. Log warnings when too much data is dropped: `if dropped_pct > 10%: log.warning(...)`
4. Add optional `--strict` flag to fail on data quality issues

---

### Missing File Existence Checks for Critical Dependencies

**Issue:** RIR and MUSAN noise manifest files are checked in `generate_perturbations.py`, but many other critical files have no validation.

**Files affected:**
- `scripts/error_decomposition.py`: No check if RESULTS_DIR or prediction files exist before processing
- `scripts/compute_fairness_metrics.py`: Assumes results directory structure exists
- `scripts/generate_all_plots.py` (lines 332-334, 393-395): Skips silently if expected CSV files don't exist instead of failing fast
- `scripts/run_inference.py` (line 674): Reads manifest CSV without catching FileNotFoundError

**Impact:** Silent skips in figure generation lead to incomplete results. WER computation fails if prediction CSVs are missing but without clear error message.

**Fix approach:**
1. Check manifest files at start: Add `if not manifest_path.exists(): raise FileNotFoundError(...)`
2. Validate output directories are writable: `os.access(output_dir, os.W_OK)`
3. Log skipped files with reason (not just silent continue)
4. Add `--check-deps` flag to verify all inputs before running

---

## Known Issues & Fragility

### Float Infinity Handling in Fairness Metrics

**Issue:** Division by zero produces `float("inf")` values that are serialized to JSON and can cause downstream failures.

**Files affected:**
- `scripts/compute_fairness_metrics.py` (lines 191-192): `max(wer_list) / min(wer_list) if min(wer_list) > 0 else float("inf")`, `relative_gap_pct` division
- `scripts/compute_fairness_metrics_fs.py` (lines 156-158): Same pattern

**Impact:** JSON output contains `Infinity` values that may fail in downstream tools. Statistical comparisons break. Figure generation may fail silently.

**Fix approach:**
1. Return `None` or special sentinel instead of `float("inf")`
2. Add validation in JSON encoder: Custom JSONEncoder to catch inf/nan
3. Document these edge cases: When less than 2 groups pass MIN_GROUP_SIZE
4. Add test case for edge case: WER = 0 for one group

---

### Incomplete Error Decomposition Edge Case Handling

**Issue:** Error decomposition logic uses `max(1, ...)` to avoid division by zero, but can still produce invalid results when all error types are zero.

**Files affected:**
- `scripts/error_decomposition.py` (lines 51-53): `sub_pct_of_errors = substitutions / max(1, total_errors) * 100`
- `scripts/compute_fairness_metrics_fs.py` (lines 108-110): Same pattern

**Impact:** Perfect recognition (no errors) produces undefined percentage breakdowns (0/1 = 0%, meaningless). Results may be misinterpreted.

**Fix approach:**
1. Return `None` for percentages when total_errors = 0 (or separate "N/A" field)
2. Add explicit check: `if total_errors == 0: return None`
3. Document assumption: "Error percentages are undefined for perfect recognition"

---

### Silent Failure in Batch Processing with Partial Failures

**Issue:** When a batch fails, the entire batch is marked with empty hypothesis, but frame-level tracking is lost. If some audio in batch loads and some doesn't, inconsistency can occur.

**Files affected:**
- `scripts/run_inference.py` (lines 262-288): Batch loading collects valid audio indices, but if batch inference fails (lines 276-289), ALL items get empty string regardless of partial success
- `scripts/run_inference.py` (lines 513-523): Similar pattern for Whisper audio loading

**Impact:** Cannot distinguish between "batch failed" and "individual samples had no hypothesis". Metrics on partial batches are unreliable. Debugging is harder.

**Fix approach:**
1. Track per-sample status: `{"idx": ..., "hypothesis": ..., "status": "success" | "load_failed" | "inference_failed"}`
2. Save partial batches even on error (successful items shouldn't be lost)
3. Log batch error details: Which indices failed in inference vs. audio loading

---

### Memory Inefficiency in NoisePool and RIR Caching

**Issue:** `NoisePool._cache` in `generate_perturbations.py` (line 127) loads all noise files into memory lazily. Comment says "~2GB" but no size limits or eviction policy.

**Files affected:**
- `scripts/generate_perturbations.py` (lines 127-151): NoisePool cache can grow unbounded

**Impact:** Long-running perturbation generation could exhaust memory. OOM kills process without warning. No way to profile or limit cache size.

**Fix approach:**
1. Add optional max cache size limit: `max_cache_bytes=2e9`
2. Implement LRU eviction when cache exceeds limit
3. Add `--no-cache` flag for memory-constrained environments
4. Log cache size periodically: "Cache: 1.5GB/2GB"

---

### Hard-Coded Magic Constants Across Scripts

**Issue:** Constants like MIN_GROUP_SIZE, SAMPLE_RATE, batch size are hardcoded in multiple scripts, inconsistently.

**Files affected:**
- `MIN_GROUP_SIZE = 50`: Hardcoded in `prepare_dataset.py`, `prepare_fairspeech.py`, `compute_fairness_metrics.py`, `compute_fairness_metrics_fs.py`, `error_decomposition.py`, `error_decomposition_fs.py`
- `SAMPLE_RATE = 16000`: Hardcoded in `run_inference.py` (line 93), `generate_perturbations.py` (line 51), `compute_perturbation_metrics.py`, `prepare_librispeech.py`
- SNR levels, silence percentages, batch sizes scattered across scripts

**Impact:** Changing assumptions requires edits in many places. Risk of inconsistent values between scripts. Hard to reproduce experiments with different parameters.

**Fix approach:**
1. Create `scripts/config.py` with all shared constants
2. Import: `from config import MIN_GROUP_SIZE, SAMPLE_RATE`
3. Add argparse override for each constant: `parser.add_argument("--min-group-size", default=MIN_GROUP_SIZE)`

---

## Performance Bottlenecks

### Inefficient Resampling Fallback Pattern

**Issue:** `generate_perturbations.py` (lines 88-96) tries librosa first, falls back to torchaudio. Both are imported locally, causing repeated import overhead.

**Files affected:**
- `scripts/generate_perturbations.py` (lines 88-96): Inside `load_audio()` function called per file

**Impact:** Slight overhead per file, adds up in large-scale perturbation generation. Better to detect once and set global flag.

**Fix approach:**
1. Move import check to module level
2. Set global flag: `_use_librosa = _test_import("librosa")`
3. Call appropriate resampler once per audio

---

### N² Pairwise Significance Testing

**Issue:** Mann-Whitney U tests run on all pairs of demographic groups without limiting to significant candidates.

**Files affected:**
- `scripts/compute_fairness_metrics.py` (lines 199-230): `pairwise_significance()` uses `itertools.combinations()` on all groups

**Impact:** For 7 ethnic groups, that's 21 tests (with multiple comparison correction, very strict). For many axes, becomes slow. No filtering or early stopping.

**Fix approach:**
1. Only test pairs where WER difference > threshold (e.g., 5%)
2. Apply Bonferroni correction explicitly and document
3. Optional `--fast-stats` flag to skip pairwise tests

---

## Security Considerations

### No Input Sanitization for Generated File Paths

**Issue:** Utterance IDs and demographic group names are used directly in file paths without sanitization.

**Files affected:**
- `scripts/run_inference.py` (line 683): `os.path.join(perturbed_base, f"{uid}.wav")`
- `scripts/generate_perturbations.py`: Output path construction from dataset manifest fields

**Impact:** If utterance ID contains path traversal characters (unlikely but possible), could write files outside intended directory. CSV injection if exported without escaping.

**Fix approach:**
1. Validate utterance IDs: `assert re.match(r'^[a-zA-Z0-9_-]+$', uid)`
2. Use `pathlib.Path` with strict validation: `.resolve()` and check parent
3. Escape demographic names in table output: `group.replace("'", "''")` for CSV

---

### JSON Output Serialization of Untrusted WER Data

**Issue:** Bootstrap CIs and fairness metrics are serialized to JSON without validation of numeric values.

**Files affected:**
- `scripts/compute_fairness_metrics.py` (lines 500-503): Custom JSON encoder handles numpy types but not inf/nan
- `scripts/error_decomposition.py` (lines 142-156): Similar

**Impact:** Malformed JSON if inf/nan slip through. Downstream scripts fail parsing.

**Fix approach:**
1. Validate numeric ranges before JSON: `assert np.isfinite(value), f"Invalid WER: {value}"`
2. Replace inf/nan with explicit string: `"inf"` or `null`
3. Add JSON schema validation to output

---

## Test Coverage Gaps

### No Unit Tests for Core Metrics

**Issue:** Critical WER and fairness metric calculations have no unit tests. Only manual validation.

**Files affected:**
- `scripts/compute_fairness_metrics.py`: No tests for `compute_group_wer()`, `bootstrap_wer()`, `compute_fairness_metrics()`
- `scripts/error_decomposition.py`: No tests for `compute_error_decomposition()`
- `scripts/generate_perturbations.py`: No tests for noise/RIR mixing

**Risk:** 
- Regressions in metric calculation go unnoticed
- Assumptions about jiwer API behavior not verified
- Edge cases (empty groups, perfect recognition, all deletions) untested

**Fix approach:**
1. Create `tests/test_metrics.py` with fixtures:
   - Perfect recognition case (WER = 0)
   - Complete failure case (WER = 1.0)
   - Balanced error mix
   - Empty group (0 utterances)
2. Add parametrized tests for demographic grouping logic
3. Validate against hand-computed WER for known transcripts

---

### No Integration Tests for End-to-End Pipeline

**Issue:** Individual scripts work, but pipeline orchestration is untested. Manifest flow (prepare → inference → metrics) has no validation.

**Files affected:**
- All scripts: No CI/CD pipeline test

**Risk:**
- Silent data loss between pipeline stages
- Manifest format mismatches between prepare and inference
- Missing output files go undetected

**Fix approach:**
1. Create `tests/integration/test_cv_pipeline.py` with mock data
2. Run: prepare_dataset → run_inference (small batch) → compute_fairness_metrics → verify outputs exist
3. Check manifest columns propagate correctly

---

## Scaling Limits

### Inference Script Memory Usage with Large Batches

**Issue:** `run_inference.py` loads entire batch of audio into memory before inference. No memory profiling or limits.

**Files affected:**
- `scripts/run_inference.py` (lines 257-295): Batch loading with no size limit in bytes

**Current capacity:** ~32 utterances × 30s × 16kHz × 2 bytes = ~30MB per batch
**Limit:** When audio length varies, could exceed GPU memory (8GB cards common)

**Scaling path:**
1. Add `--batch_bytes_limit` parameter (default 100MB)
2. Dynamically adjust batch size based on audio durations
3. Profile and log memory per batch: "Batch uses 250MB"

---

### Perturbation Generation I/O Bottleneck

**Issue:** `generate_perturbations.py` generates perturbations on-the-fly for all conditions. With 12 perturbation types × ~50k utterances = 600k audio files written.

**Files affected:**
- `scripts/generate_perturbations.py`: No parallelization, sequential generation

**Impact:** Single-threaded I/O-bound operation. On HPC with many small files, slow.

**Scaling path:**
1. Add optional multiprocessing: `--workers 8` pool
2. Or generate to tar streams to reduce file count
3. Benchmark: Current time estimate?

---

## Missing Critical Features

### No Logging Configuration

**Issue:** Scripts use raw `print()` statements. No structured logging, no log files, no severity levels.

**Impact:** Hard to debug long-running jobs. No audit trail. Errors mixed with progress output.

**Fix approach:**
1. Add logging setup to all scripts:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   log = logging.getLogger(__name__)
   ```
2. Redirect to file: `logging.FileHandler(f'logs/{script}_{timestamp}.log')`

---

### No Resume/Checkpointing in Long-Running Jobs

**Issue:** Inference and perturbation generation restart from scratch on failure. No checkpointing.

**Files affected:**
- `scripts/run_inference.py`: No resume logic (partial CSV reuse attempted but incomplete)
- `scripts/generate_perturbations.py`: No way to skip already-generated perturbations

**Impact:** Hour-long jobs that crash near the end must restart. Wasted compute.

**Fix approach:**
1. Write checkpoint file every N batches: `checkpoint.json` with `last_batch_idx`
2. On start, check: `if checkpoint exists: start from last_batch_idx`
3. Validate output files on resume to avoid duplicates

---

### No Configuration File Support

**Issue:** All configuration is command-line args or hardcoded constants. No way to save/reuse configurations.

**Files affected:**
- `scripts/run_inference.py`: 10+ argparse arguments, each with hardcoded defaults
- `scripts/generate_perturbations.py`: Dataset, perturbation types, paths all in code

**Impact:** Complex experiments hard to reproduce. No easy way to document which parameters were used.

**Fix approach:**
1. Create `config_template.yaml`:
   ```yaml
   paths:
     data_dir: /users/.../asr_fairness/data
     results_dir: /users/.../asr_fairness/results
   inference:
     batch_size: 32
     device: cuda
   ```
2. Load: `config = yaml.safe_load(open("config.yaml"))`
3. Add `--save-config` flag to dump final config used

---

## Fragile Areas

### Demographic Ordering Logic Scattered Across Files

**Issue:** Logical ordering for demographic groups (ETHNICITY_ORDER, AGE_ORDER, etc.) is defined separately in multiple files, risking divergence.

**Files affected:**
- `scripts/compute_fairness_metrics_fs.py` (lines 63-67): Defines ordering for Fair-Speech
- `scripts/generate_all_plots.py`: May define different order
- `scripts/regenerate_figures_large_fonts.py`: Another copy?

**Impact:** Figures with mismatched group ordering confuse readers. Metrics and plots show different group ranks.

**Fix approach:**
1. Extract all demographic orderings to `scripts/lib/demographics.py`
2. Single source of truth: `DEMOGRAPHIC_ORDERINGS = {"ethnicity": [...], "age": [...]}`

---

### Model Registry vs. Argument Validation

**Issue:** `MODEL_REGISTRY` in `run_inference.py` defines available models, but argparse doesn't validate against it.

**Files affected:**
- `scripts/run_inference.py` (lines 27-91): MODEL_REGISTRY defined, but `parser.add_argument("--model")` has no `choices`

**Impact:** Typos in model name (e.g., `whisper-large` vs `whisper-large-v3`) silently fail at inference time, not argument parsing.

**Fix approach:**
1. Use `choices=list(MODEL_REGISTRY.keys())` in argparse
2. Fail fast with helpful error: "Unknown model: X. Choose from: [list]"

