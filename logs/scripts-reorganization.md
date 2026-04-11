---
fileClass: Log
name: scripts/ reorganization into role-based subdirectories
description: Moved 29 flat scripts into 7 role-based subdirectories (data, inference, metrics, plots, validation, analysis, setup); left scripts/training untouched; updated docstrings and CLAUDE.md
created: 2026-04-11
updated: 2026-04-11
tags: [refactor, structure, scripts, housekeeping]
aliases: []
status: complete
subtype: refactor
---

## Summary

Reorganized the flat `scripts/` directory (26 Python files + 4 shell scripts at the top level) into **7 role-based subdirectories**, keeping `scripts/training/` untouched because its Python package path (`from scripts.training.X`) is load-bearing for cross-imports. Also moved the root-level `project-overview.html` into `docs/`.

No logic changes, no simplifications — purely structural. All moves were done with `git mv` so history is preserved.

## New layout

| Subdir | Files | Role |
|---|---|---|
| `scripts/data/` | 7 | `prepare_*.py`, `extract_test_clips.py`, `generate_perturbations.py` — produce manifests and audio |
| `scripts/inference/` | 1 | `run_inference.py` — MODEL_REGISTRY dispatch, only GPU-eval entry point |
| `scripts/metrics/` | 5 | `compute_fairness_metrics*.py`, `compute_perturbation_metrics.py`, `error_decomposition*.py` |
| `scripts/plots/` | 3 | `generate_all_plots.py`, `generate_perturbation_plots.py`, `regenerate_figures_large_fonts.py` |
| `scripts/validation/` | 6 | `validate_splits.py`, `validate_*_perturbed_test.py`, `validate_test_run.py` |
| `scripts/analysis/` | 3 | `whisper_hallucination_analysis.py`, `parse_bootstrap_cis.py`, `prepare_overleaf.py` |
| `scripts/setup/` | 4 | shell scripts: `download_musan.sh`, `download_rirs.sh`, `setup_nemo.sh`, `launch_priority_batch.sh` |
| `scripts/training/` | (unchanged) | package: `scripts.training` — **must not be moved** |

## What was updated alongside the moves

- **Docstring usage examples** — all `python scripts/foo.py ...` examples in the moved Python files updated to `python scripts/<subdir>/foo.py ...` (19 files touched).
- **Help-text strings** — `scripts/data/generate_perturbations.py` prints `bash scripts/setup/download_{musan,rirs}.sh`; `scripts/validation/validate_perturbed_test.py` prints `bash scripts/setup/launch_priority_batch.sh`.
- **Self-reference in `scripts/setup/launch_priority_batch.sh`** — header comment updated to the new path.
- **`CLAUDE.md`** — Directory Structure section rewritten to show the new tree; Architecture section updated so the pipeline refers to the new paths; added a "Subdirectory roles" block explaining what each folder is for and the warning about `scripts/training/`.

## What was deliberately NOT changed

- **`scripts/training/`** — 19 files untouched. The package path is imported from inside training code (`from scripts.training.data_loader import ...`) and from `scripts/training/tune_vram.py:206` which subprocesses `scripts/training/train_standard_lora.py` as an argv string. Moving any of these would break cross-phase training runs silently.
- **Historical refs in `.planning/`, `logs/`, and `experiments/`** — ~13 markdown files mention old paths like `scripts/run_inference.py`. These are frozen historical records; updating them would rewrite history for no benefit.
- **Root PDFs** (`colm2026_conference.pdf`, `llm-asr-fairness-midterm.pdf`) — left in place. They are committed binaries but moving them would produce churn without solving the underlying "binaries in git" issue (a Git LFS migration is the real fix, out of scope here).
- **40 tracked files inside `outputs/`** — gitignored now but committed before gitignore added. Several are intentional experiment records (`grid_results.md`, `vram_config.json`, `io_remediation_report.md`), so untracking needs case-by-case judgment. Deferred.
- **`autoresearch/`** — appears to be an abandoned parallel sub-project; needs user decision before touching.
- **Code simplification** — the flagged near-duplicates (4 `validate_perturbed_*.py` scripts; `error_decomposition.py` vs `_fs.py`; `compute_fairness_metrics.py` vs `_fs.py`) were **not** consolidated. Each has dataset-specific logic (different axis enums, different JSON output schemas consumed by plotting scripts, different sanity checks), and the refactor risk outweighs the ~100 lines of saved duplication. If consolidation is desired later, it should be done file-by-file with the downstream JSON consumers verified.

## Verification

- `grep -rn "python scripts/" scripts/ --include="*.py"` returns only the updated new-path usage strings — no stale references remain inside moved files.
- `ls scripts/` shows only the 7 new subdirectories plus `training/`.
- `git status` shows all moves as renames (history preserved).

## Blast radius caveats for the user

External scripts that live outside this repo (cluster slurm scripts, shell aliases, cron jobs) that invoke `python scripts/run_inference.py` etc. will break and need manual updating. No such references exist inside the repo.
