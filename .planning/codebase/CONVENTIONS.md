# Coding Conventions

**Analysis Date:** 2026-04-05

## Naming Patterns

**Files:**
- Snake case with descriptive names: `run_inference.py`, `compute_fairness_metrics.py`, `error_decomposition.py`
- Utilities and generators end in descriptive verbs: `prepare_dataset.py`, `generate_perturbations.py`, `extract_test_clips.py`
- Validation scripts prefix with `validate_`: `validate_cv_perturbed_test.py`

**Functions:**
- Snake case: `compute_group_wer()`, `bootstrap_wer()`, `plot_wer_by_group()`
- Descriptive action-based names: `load_audio()`, `normalize_text()`, `parse_args()`
- Helper/utility functions use underscores: `_extract_granite_transcription()`, `_build_row()`, `_header_written`

**Variables:**
- Snake case throughout: `model_registry`, `sample_rate`, `min_group_size`, `output_dir`
- Constants in UPPERCASE: `SAMPLE_RATE = 16000`, `MIN_GROUP_SIZE = 50`, `DATA_DIR = "/users/..."`
- Boolean prefixes with `is_` or `has_`: `is_perturbed`, `_header_written`, `has_demographics`
- Collection suffixes for clarity: `all_results`, `batch_preds`, `group_wers`

**Types:**
- Class names in PascalCase: `IncrementalCSVWriter()`
- No explicit type hints (older Python/flexibility style) but function docstrings document expected types

**Dictionary Keys:**
- Use descriptive strings as dict keys in metadata structures: `"model"`, `"architecture"`, `"generation"`, `"params"`

## Code Style

**Formatting:**
- No explicit formatter (no `.prettierrc` or `pyproject.toml` found)
- Line length varies (some lines exceed 80 chars)
- Indentation: 4 spaces
- No blank line enforcement between functions/classes observed

**Linting:**
- No linter config found (no `.flake8`, `.pylintrc`)
- Follows loose PEP 8 style

**Comments:**
- Extensive module-level docstrings for scripts: `"""Compute fairness metrics from ASR inference..."""`
- Section dividers using comment blocks:
  ```python
  # ── Model registry ──────────────────────────────────────────────────────────
  # ═════════════════════════════════════════════════════════════════════════════
  # MODEL RUNNERS
  # ═════════════════════════════════════════════════════════════════════════════
  ```
- Inline comments for non-obvious logic
- Comment usage: when, why, and implementation details

## Import Organization

**Order:**
1. Standard library: `import sys`, `import json`, `import argparse`, `import os`
2. Third-party packages: `import numpy`, `import pandas`, `import torch`, `import torchaudio`
3. Scientific/ML frameworks: `from transformers import ...`, `import jiwer`, `from scipy import stats`
4. Visualization: `import matplotlib`, `import seaborn`
5. Local/project imports: `from whisper.normalizers import EnglishTextNormalizer`

**Path Configuration:**
- Absolute paths only, no relative imports
- Paths defined as module-level constants: `DATA_DIR`, `RESULTS_DIR`, `PERTURBED_DIR`
- Path construction via `os.path.join()` for cross-platform compatibility

## Error Handling

**Patterns:**
- Bare `except Exception as e:` for broad error catching in inference loops
- Specific errors logged with `print(f"  WARNING: ...")` for user visibility
- Graceful degradation: missing audio returns `None` or empty string, not crash
- Try/except blocks around model loading and batch processing:
  ```python
  try:
      wer_val = jiwer.wer(ref, hyp)
  except Exception:
      wer_val = 1.0
  ```
- Division-by-zero protection: `output.substitutions / max(1, output.hits)`
- File existence checks before processing: `if not os.path.isfile(audio_path): continue`

**Exit codes:**
- Scripts use `sys.exit(0)` for success, `sys.exit(1)` for errors
- Validation scripts return 0 if all checks pass, 1 if issues found

## Logging

**Framework:** `print()` statements (no logging module used)

**Patterns:**
- Header sections: `print(f"\n{'='*60}")` for major milestones
- Progress reporting: `print(f"  [{i:,}/{n:,}] ({100*i/n:.0f}%)")`
- Warnings prefixed with "WARNING:" or "⚠": `print(f"  WARNING: Failed to load {path}: {e}")`
- Verbose section headers with dashes: `print(f"\n{'─'*60}")`
- Summary metrics at end: `print(f"Overall WER: {overall_wer*100:.2f}% (n={len(valid):,})")`

## Function Design

**Size:** Functions range from 5–50 lines, with longer sequences (inference loops) kept in single functions for state management

**Parameters:**
- Positional for required args: `def compute_group_wer(df: pd.DataFrame) -> dict:`
- Keyword args with defaults for options: `def bootstrap_wer(df: pd.DataFrame, n_bootstrap: int = 1000)`
- Args class for CLI scripts: `def parse_args()` returns parsed namespace

**Return Values:**
- Functions return dicts for structured results: `{"wer": val, "n_utterances": count, "ci_low": lo, "ci_high": hi}`
- Tuples for multi-value returns: `return np.mean(wers), np.percentile(wers, 2.5), np.percentile(wers, 97.5)`
- None for missing data or on error: `return None, None, None`
- Collections in predictable order (list of dicts, DataFrame)

## Module Design

**Exports:**
- No `__all__` declarations observed
- Scripts are executable (entry point is `if __name__ == "__main__": main()`)

**Class Design:**
- Minimal OOP: only one main class found `IncrementalCSVWriter` in `run_inference.py`
- Class groups related state/methods: `__init__()` stores config, `_build_row()` formats data, `flush()` writes batch

**Constants Organization:**
- Model registries as module-level dicts: `MODEL_REGISTRY`, `MODEL_ORDER`, `MODEL_SHORT`, `MODEL_COLORS`
- Perturbation configs as dicts: `PERTURBATION_LABELS`, `PERTURBATION_FAMILIES`, `PERTURBATION_CONFIGS`
- Mapping tables for data normalization: `GENDER_MAP`, `ACCENT_MAP`, `AGE_MAP`
- Numeric thresholds as constants: `MIN_GROUP_SIZE = 50`, `SAMPLE_RATE = 16000`

## Batch Processing Pattern

**Observed in:** `run_inference.py`, `compute_perturbation_metrics.py`, validation scripts

**Pattern:**
```python
for i in range(0, n, batch_size):
    batch = df.iloc[i:i + batch_size]
    # Process batch
    try:
        # Model inference or computation
        results = model.process(batch)
    except Exception as e:
        print(f"  WARNING: Batch {i} failed: {e}")
        results = fallback_empty_results()
    
    if writer:
        writer.flush(results)  # Incremental disk writes
    
    if (i // batch_size) % 20 == 0:
        print(f"  [{i:,}/{n:,}] ({100*i/n:.0f}%)")
```

## Data Validation Pattern

**Observed in:** `validate_cv_perturbed_test.py`, `prepare_dataset.py`

**Pattern:**
- Collect issues in a list: `issues = []`
- Run multiple checks, append failures
- Report summary at end: `print(f"  Issues: {len(issues)}")`
- Exit with code based on validation: `sys.exit(1 if issues else 0)`

---

*Convention analysis: 2026-04-05*
