# Testing Patterns

**Analysis Date:** 2026-04-05

## Test Framework

**Runner:**
- No pytest or unittest framework detected
- No test discovery setup (no `conftest.py`, `pytest.ini`, `setup.cfg`)

**Assertion Library:**
- Not applicable — no automated test suite present

**Run Commands:**
- No standard test command available
- Validation is performed via standalone scripts with manual inspection

## Test Strategy

**Approach:** Manual validation scripts instead of automated unit tests

The codebase uses dedicated validation/sanity-check scripts rather than a formal test framework. These scripts:
1. Read output from processing scripts
2. Verify data integrity and correctness
3. Check against expected invariants
4. Report issues and exit with status codes

This pattern is appropriate for research code where outputs are large (CSVs with thousands of rows) and validation requires domain knowledge (ASR fairness metrics).

## Test File Organization

**Location:**
- Validation scripts placed in `scripts/` directory alongside production scripts
- No separate `tests/` directory
- Scripts follow naming pattern: `validate_*.py`

**Examples in codebase:**
- `scripts/validate_cv_perturbed_test.py` - Validates perturbation test batch (108 files × 9 models)
- `scripts/validate_perturbed_reverb_masking_test.py` - Validates reverb/masking conditions
- `scripts/validate_perturbed_silence_test.py` - Validates silence perturbation conditions

## Validation Test Structure

**Pattern from `validate_cv_perturbed_test.py`:**

```python
# 1. Setup: Define what to check
MODELS = [...]
PERTURBATION_FAMILIES = [("SNR", [...]), ...]
CHATTY_PATTERNS = [...]  # Patterns indicating bad outputs

# 2. Collect validation issues
issues = []  # List of failure descriptions

# 3. Run checks in loops
for model in MODELS:
    for pert in ALL_PERTURBATIONS:
        # Check file exists
        # Check row count
        # Check column values
        # Check for bad patterns
        if not valid:
            issues.append(f"ISSUE: {details}")

# 4. Report monotonicity (perturbation severity increases WER)
for family_name, levels in PERTURBATION_FAMILIES:
    for model in MODELS:
        wers = [all_results.get((model, lv)) for lv in levels]
        monotonic = wers[0] <= wers[1] + 5 and wers[1] <= wers[2] + 5
        if not monotonic:
            issues.append(f"{model}/{family_name}: WER not monotonic")

# 5. Summary and exit
print(f"  Issues: {len(issues)}")
if issues:
    for issue in issues:
        print(f"    - {issue}")
    sys.exit(1)  # Validation failed
else:
    sys.exit(0)  # All checks passed
```

## Data Validation Checklist

**File-level checks:**
- File exists at expected path: `if not os.path.exists(fpath): issues.append(...)`
- Row count matches expectation: `if len(df) != expected_count: issues.append(...)`
- No missing critical columns

**Content-level checks:**
- No empty values in required fields: `(df["hypothesis_raw"].fillna("") == "").sum()`
- Pattern matching for bad outputs: `check_chatty(text)` detects chatbot wrapper text
- Metadata correctness: `if pert_vals[0] != expected_pert: issues.append(...)`

**Semantic checks:**
- Perturbation severity increases error: weak monotonicity with tolerance
  ```python
  monotonic = wers[0] <= wers[1] + 5 and wers[1] <= wers[2] + 5  # 5% tolerance
  ```
- Perturbed condition WER > clean condition WER: `if snr_20db_wer < clean_wer - 5: issues.append(...)`
- Group size meets minimum threshold: `if len(group) < MIN_GROUP_SIZE: skip`

## Specific Validation Scripts

**`validate_cv_perturbed_test.py` - Location: `scripts/validate_cv_perturbed_test.py`**

Validates all 108 prediction files (9 models × 12 perturbation conditions):
```
Checks:
  1. All 108 files exist (predictions_{model}_{pert_suffix}.csv)
  2. Each file has exactly 5 rows
  3. No empty hypotheses
  4. No chatbot wrapper text (patterns from Granite)
  5. Perturbation column correctly set
  6. WER monotonic within perturbation families (SNR, Reverb, Silence, Masking)
  7. Mildest perturbation (SNR 20dB) WER > clean baseline WER
```

Usage:
```bash
python scripts/validate_cv_perturbed_test.py
python scripts/validate_cv_perturbed_test.py --output_dir results/perturbed_cv_test
```

Exit code: 0 (all pass) or 1 (issues found)

**`validate_perturbed_reverb_masking_test.py` - Location: `scripts/validate_perturbed_reverb_masking_test.py`**

Similar structure, focused on reverb and masking perturbations specifically.

## Hypothesis Testing Patterns

**Statistical significance testing in `compute_fairness_metrics.py`:**

```python
def pairwise_significance(df: pd.DataFrame, group_col: str, min_size: int = MIN_GROUP_SIZE) -> dict:
    """Run Mann-Whitney U tests between all pairs of demographic groups."""
    for (g1, df1), (g2, df2) in combinations(valid_groups.items(), 2):
        wers1 = df1["wer"].dropna().values
        wers2 = df2["wer"].dropna().values
        
        stat, p_value = stats.mannwhitneyu(wers1, wers2, alternative="two-sided")
        results[f"{g1}_vs_{g2}"] = {
            "p_value": p_value,
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01,
            ...
        }
```

Tests for fairness hypothesis: significant WER differences between demographic groups.

## Bootstrap Confidence Intervals

**Pattern in `compute_fairness_metrics.py`:**

```python
def bootstrap_wer(df: pd.DataFrame, n_bootstrap: int = 1000) -> tuple:
    """Compute bootstrap 95% confidence interval for WER."""
    valid = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if len(valid) < 10:
        return None, None, None
    
    refs_clean, hyps_clean = zip(*valid)
    n = len(refs_clean)
    
    wers = []
    rng = np.random.RandomState(42)  # Reproducible seed
    for _ in range(n_bootstrap):
        indices = rng.choice(n, n, replace=True)
        ref_sample = [refs_clean[i] for i in indices]
        hyp_sample = [hyps_clean[i] for i in indices]
        try:
            wers.append(jiwer.wer(ref_sample, hyp_sample))
        except Exception:
            continue
    
    return np.mean(wers), np.percentile(wers, 2.5), np.percentile(wers, 97.5)
```

Used to quantify uncertainty in group-level WER metrics. Called during `analyze_model()` for each demographic group.

## Data Preparation Validation

**In `prepare_dataset.py`:**

Sanity checks after data preparation:
```python
missing_audio = 0
for row in rows:
    audio_path = os.path.join(clips_dir, audio_file)
    if not os.path.isfile(audio_path):
        missing_audio += 1
        continue
    manifest.append({...})

print(f"WARNING: {missing_audio} missing audio files")
```

**In `prepare_librispeech.py`:**

Final verification:
```python
missing = df[~df["audio_path"].apply(os.path.exists)]
if len(missing) > 0:
    print(f"WARNING: {len(missing)} audio files missing!")
else:
    print("All audio files verified.")
```

## Sanity Checks During Processing

**In `run_inference.py`:**

Incremental output writing ensures data is on disk:
```python
class IncrementalCSVWriter:
    def flush(self, predictions):
        """Write a batch of predictions to CSV. Called after each inference batch."""
        if not predictions:
            return
        rows = [self._build_row(p) for p in predictions]
        df = pd.DataFrame(rows)
        df.to_csv(self.output_path, mode="a", header=not self._header_written, index=False)
        self._header_written = True
        self.count += len(rows)
```

Benefits:
- If job times out, partial results are not lost
- Can resume from last completed batch
- Verification can run on partial results

## Test Data

**Located in:** `scripts/` (scripts generate test data, no checked-in test fixtures)

**Pattern:**
- Test data comes from public datasets (Common Voice, LibriSpeech, Fair-Speech)
- Scripts download/prepare datasets on demand
- Validation scripts use small test subsets (e.g., 5 samples per condition)

**Example from `validate_cv_perturbed_test.py`:**
```python
# Small 5-sample test of each model × perturbation combo
MODELS = [...9 models...]
ALL_PERTURBATIONS = [...12 conditions...]
# Validation checks 9 × 12 = 108 files
```

## Coverage Gaps

**No explicit test coverage:**
- No coverage tracking or metrics
- No mocking of external services (Hugging Face, model downloads)
- No testing of edge cases (empty audio, corrupted files) — handled by graceful fallback in production

**What IS tested through validation:**
- Complete inference pipelines (end-to-end model loading → inference → output)
- Fairness metrics computation
- Data integrity of results
- Statistical properties (monotonicity, significance)

## Error Testing Pattern

**In inference functions (e.g., `infer_whisper()`):**

```python
for i in range(0, n, args.batch_size):
    batch = manifest_df.iloc[i:i + args.batch_size]
    audio_list, valid_idx = [], []
    batch_preds = []
    
    # Load audio, catch missing files
    for j, (_, row) in enumerate(batch.iterrows()):
        audio = load_audio(row["audio_path"])  # Returns None on error
        if audio is not None:
            audio_list.append(audio)
            valid_idx.append(i + j)
        else:
            batch_preds.append({"idx": i + j, "hypothesis_raw": ""})  # Empty fallback
    
    # Inference with broad exception handling
    try:
        inputs = processor(audio_list, ...)
        with torch.no_grad():
            ...
        texts = processor.batch_decode(pred_ids, ...)
        for idx, text in zip(valid_idx, texts):
            batch_preds.append({...})
    except Exception as e:
        print(f"  WARNING: Batch {i} failed: {e}")
        for idx in valid_idx:
            batch_preds.append({"idx": idx, "hypothesis_raw": ""})  # Fallback empty
```

Tests error resilience: individual file failures don't crash the job.

---

*Testing analysis: 2026-04-05*
