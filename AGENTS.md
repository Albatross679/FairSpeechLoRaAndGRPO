# AGENTS.md

## Project Overview

NLP class project investigating fairness in Automatic Speech Recognition (ASR) systems across demographic subgroups. Evaluates three generations of ASR models (wav2vec2, Whisper, Qwen3-ASR) on fairness metrics (WER gaps, ratios, statistical significance) using LibriSpeech, CommonVoice, and FairSpeech datasets. Includes robustness analysis via audio perturbations (noise, reverb, silence).

## Directory Structure

```
FairSpeechLoRaAndGRPO/
├── .pkm/              # PKM/Obsidian workspace metadata
├── .codex/            # Local Codex hooks/settings (gitignored)
├── .pkm/              # PKM/Obsidian workspace metadata; DB gitignored
├── .planning/         # GSD planning artifacts (roadmaps, phases, research)
├── AGENTS.md          # This file
├── pyproject.toml     # Project config
├── autoresearch/      # Automated research/scaffolding artifacts
├── archive/           # Tracked archived/stale project lines
├── archive-local/     # Local-only archived data/results (gitignored)
├── fileClasses/       # FileClass definitions (Metadata Menu schema)
├── scripts/
│   ├── data/          # prepare_*, extract_*, generate_perturbations
│   ├── inference/     # run_inference.py (MODEL_REGISTRY dispatch)
│   ├── metrics/       # compute_fairness_*, compute_perturbation_*, error_decomposition_*
│   ├── plots/         # generate_*_plots, regenerate_figures_large_fonts
│   ├── validation/    # validate_splits, validate_*_perturbed_test, validate_test_run
│   ├── analysis/      # parse_bootstrap_cis, prepare_overleaf, and other one-off paper utilities
│   ├── setup/         # shell: download_musan, download_rirs, setup_nemo, launch_priority_batch
│   └── training/      # LoRA / GRPO training infrastructure (package: scripts.training)
├── docs/              # HTML explainers and project-overview
├── datasets/          # (gitignored; created locally) audio and manifests/results
├── outputs/           # training run artifacts (tracked)
├── logs/              # One file per log entry (<topic>.md)
├── experiments/       # One file per experiment (<topic>.md)
├── issues/            # One file per issue (<topic>.md); create if absent
├── knowledge/         # Domain knowledge and reference (<topic>.md)
├── references/        # One file per reference (<topic>.md); create if absent
├── tasks/             # PRDs and task specs (prd-<feature>.md)
├── tests/             # pytest modules
├── colm2026_conference.pdf       # Conference paper PDF
└── llm-asr-fairness-midterm.pdf  # Midterm report PDF
```

**Subdirectory roles in `scripts/`** (reorganized 2026-04-11):
- **data/** — anything that produces a manifest or audio file that downstream steps consume
- **inference/** — the `MODEL_REGISTRY` dispatch and the only script that touches GPUs for eval
- **metrics/** — consumes prediction CSVs → produces analysis JSONs
- **plots/** — consumes analysis JSONs → produces figures
- **validation/** — sanity checks for splits and perturbation runs (exit code based)
- **analysis/** — one-off deep-dives and paper-prep utilities; hallucination/head-surgery analysis was archived on 2026-04-27
- **setup/** — one-time install and data download scripts (bash)
- **training/** — LoRA/GRPO training; imported as the `scripts.training` Python package. **Do not move files inside** — package path is load-bearing for cross-imports.

Archived head-surgery/hallucination work lives under `archive/head-surgery-2026-04-27/`; local ignored payloads live under `archive-local/head-surgery-2026-04-27/`.


## Credentials

- GitHub: `Albatross679`, email `qifan_wen@outlook.com`

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

Script-oriented pipeline: `scripts/data/prepare_*.py` → `scripts/data/generate_perturbations.py` → `scripts/inference/run_inference.py` → `scripts/metrics/compute_*.py` → `scripts/plots/generate_*.py`. Validation scripts (`scripts/validation/validate_*.py`) catch inference failures. Data flows as CSV manifests → prediction CSVs → analysis JSONs → figures. Model dispatch via `MODEL_REGISTRY` dict in `scripts/inference/run_inference.py`.

## GPU Safety

Before launching any GPU-intensive task:
1. Run `nvidia-smi` to check current GPU utilization and memory
2. Do NOT launch if GPU memory usage > 80% (another job may be running)
3. Use `CUDA_VISIBLE_DEVICES` to target specific GPUs when multiple are available

## HTML Style

When editing or creating HTML files, use concise and structured design (tables, lists, cards, grids) rather than wordy paragraph layouts. Prefer scannable content over prose.

## graphify

This project may have a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` for god nodes and community structure **if it exists**.
- If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw files.
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost), unless graphify is unavailable in the environment.
