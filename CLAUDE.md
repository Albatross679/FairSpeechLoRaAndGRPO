# CLAUDE.md

## Project Overview

NLP class project investigating fairness in Automatic Speech Recognition (ASR) systems across demographic subgroups. Evaluates three generations of ASR models (wav2vec2, Whisper, Qwen3-ASR) on fairness metrics (WER gaps, ratios, statistical significance) using LibriSpeech, CommonVoice, and FairSpeech datasets. Includes robustness analysis via audio perturbations (noise, reverb, silence).

## Directory Structure

```
NLPClassProject/
├── .claude/           # Claude Code settings and memory
├── .git/
├── .gitignore
├── .vscode/           # VS Code workspace settings
├── CLAUDE.md          # This file
├── fileClasses/       # FileClass definitions (Metadata Menu schema)
├── scripts/           # All Python and shell scripts
├── logs/              # One file per log entry (<topic>.md)
├── experiments/       # One file per experiment (<topic>.md)
├── issues/            # One file per issue (<topic>.md)
├── knowledge/         # Domain knowledge and reference (<topic>.md)
├── references/        # One file per reference (<topic>.md)
├── tasks/             # PRDs and task specs (prd-<feature>.md)
├── colm2026_conference.pdf   # Conference paper PDF
└── llm-asr-fairness-midterm.pdf  # Midterm report PDF
```


## Credentials

- GitHub: `Albatross679`, email `qifan_wen@outlook.com`

## Documentation (IMPORTANT)

Claude Code MUST document **as it goes** — immediately after each change, not batched at the end of the session. Each entry is a **separate file** in its subdirectory.

Every Markdown documentation file MUST include `fileClass` as the **first** frontmatter property:

| What | Where | Naming | When | fileClass |
|---|---|---|---|---|
| Logs | `logs/` | `<topic>.md` | After any code change that adds, fixes, or modifies functionality | `Log` |
| Experiments | `experiments/` | `<topic>.md` | After running a simulation, test, or investigation | `Experiment` |
| Issues | `issues/` | `<topic>.md` | When encountering a bug or error (before or alongside the fix) | `Issue` |
| Knowledge | `knowledge/` | `<topic>.md` | When capturing domain knowledge or reference material | `Knowledge` |
| References | `references/` | `<topic>.md` | When capturing external references or citations | `Reference` |
| Tasks | `tasks/` | `prd-<feature>.md` | When planning a feature or task (PRDs) | `Task` |

### Required properties by fileClass

All fileClasses share these **common properties**: `fileClass`, `name`, `description`, `created`, `updated`, `tags`, `aliases`.

In addition, each fileClass has specific properties that MUST be set:

- **`Log`**: `status` (draft | complete), `subtype` (fix | training | tuning | research | refactor | setup | feature | evaluation | data-prep | inference)
- **`Experiment`**: `status` (planned | running | complete | failed), optional: `algorithm`, `task`, `wandb_project`, `hardware`
- **`Issue`**: `status` (open | investigating | resolved | wontfix), `severity` (low | medium | high | critical), `subtype` (training | inference | data | compatibility | system | performance | fairness)
- **`Knowledge`**: common properties only
- **`Reference`**: `source`, `url`, `authors`
- **`Task`**: `status` (planned | in-progress | complete | cancelled)

**Threshold for logging:** A change warrants a log if it modifies behavior, fixes a bug, or changes configuration. Trivial edits (typos, whitespace, comment-only changes) do not need a log entry.

## Architecture

Script-oriented pipeline: `prepare_*.py` → `generate_perturbations.py` → `run_inference.py` → `compute_*_metrics.py` → `generate_*_plots.py`. Validation scripts (`validate_*.py`) catch inference failures. Data flows as CSV manifests → prediction CSVs → analysis JSONs → figures. Model dispatch via `MODEL_REGISTRY` dict in `run_inference.py`.

## GPU Safety

Before launching any GPU-intensive task:
1. Run `nvidia-smi` to check current GPU utilization and memory
2. Do NOT launch if GPU memory usage > 80% (another job may be running)
3. Use `CUDA_VISIBLE_DEVICES` to target specific GPUs when multiple are available


