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

<!-- GSD:project-start source:PROJECT.md -->
## Project

**GRPO Fairness-Aware ASR Fine-Tuning**

A fairness-aware fine-tuning pipeline for Qwen3-ASR-1.7B using Group Relative Policy Optimization (GRPO) with a composite accuracy-fairness reward. The system trains LoRA adapters on the frozen encoder, sweeps the fairness weight λ to trace a Pareto frontier of accuracy vs. demographic fairness, and compares against four baselines (standard LoRA, FairLoRA, Group-DRO, ICASSP 2026 fairness-prompted fine-tuning). This is the primary intervention planned in the CSE 5525 final project, building on midterm benchmarking that found Qwen3-ASR-1.7B has the best raw accuracy but the worst relative ethnicity gap (+203%).

**Core Value:** Produce a working GRPO training pipeline that traces the accuracy-fairness Pareto frontier for Qwen3-ASR-1.7B, demonstrating whether fairness-aware RL fine-tuning can close demographic gaps without sacrificing accuracy.

### Constraints

- **Hardware**: Single RTX A4000 (16GB VRAM) — LoRA required, full fine-tuning infeasible
- **Tech stack**: Must integrate with existing Python/PyTorch/HuggingFace pipeline
- **Timeline**: Class project final deliverable
- **Baselines**: 4 comparison methods needed to contextualize GRPO results
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.7+ - All source code, scripts, and data processing pipelines
- Bash - HPC job submission and data pipeline orchestration (`scripts/download_musan.sh`, `scripts/download_rirs.sh`, `scripts/launch_priority_batch.sh`, `scripts/setup_nemo.sh`)
## Runtime
- Python 3.x (unspecified, commonly 3.9+)
- pip - Primary dependency management
- Lockfile: Not detected (no requirements.txt or Pipfile.lock found)
## Frameworks
- Transformers (huggingface) - Model loading and inference for wav2vec2, Whisper, Granite
- PyTorch (torch) - Neural network computation and model loading
- torchaudio - Audio loading, resampling, feature extraction (16kHz standardization)
- NeMo - NVIDIA Speech AI toolkit for Canary-Qwen SALM model (`nemo.collections.speechlm2.models.SALM`)
- qwen_asr - Qwen3-ASR model wrapper (`Qwen3ASRModel.from_pretrained()`)
- Pandas - DataFrame operations for manifests, predictions, analysis
- NumPy - Numerical computations for signal processing
- soundfile (sf) - Audio file I/O for reading/writing WAV files
- scipy - Signal processing (convolution for reverberation via `scipy.signal.fftconvolve`)
- librosa (optional) - Audio resampling alternative if torchaudio unavailable
- jiwer - Word Error Rate (WER) computation and error decomposition (substitutions, insertions, deletions)
- scipy.stats - Mann-Whitney U tests, bootstrap confidence intervals
- Matplotlib - Low-level plotting (non-interactive Agg backend on HPC)
- Seaborn - Statistical visualization for fairness metrics
- tqdm - Progress bars for batch processing
- datasets (huggingface) - Dataset loading for CommonVoice, FairSpeech
## Key Dependencies
- Transformers [version unspecified] - Loads wav2vec2, Whisper, Granite models from HuggingFace Hub
- PyTorch [version unspecified] - Tensor computation, GPU support via `torch.cuda` or `torch.device`
- torchaudio [version unspecified] - Audio preprocessing and resampling
- jiwer [version unspecified] - WER computation (corpus-level via `jiwer.wer()`, error breakdown via `jiwer.process_words()`)
- NeMo >= 2.5.0 - Required for Canary-Qwen model support (optional installation: `pip install 'nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git'`)
- qwen-asr - Qwen3-ASR package (model-specific wrapper)
- Whisper normalizer - Text normalization (`whisper.normalizers.EnglishTextNormalizer` or fallback `whisper_normalizer.english`)
- Pandas - CSV reading/writing for manifests and results
- NumPy - Numerical array operations
- soundfile - WAV file I/O
- scipy - Signal processing (FFT convolution)
- datasets - HuggingFace dataset API
- scipy.stats - Statistical significance testing
- Matplotlib/Seaborn - Publication-ready figures
## Configuration
- HPC cluster paths hardcoded in scripts (e.g., `/users/PAS2030/srishti/asr_fairness/`)
- No .env file detected
- Dataset and results directories defined as constants in each script
- No explicit build system (pure Python scripts)
- Bash scripts for HPC job submission and data downloads
## Platform Requirements
- Python 3.7+
- CUDA-capable GPU (optional but recommended)
- 16GB+ RAM for model loading
- Linux/HPC environment
- NVIDIA GPU (CUDA 11.x or 12.x) for model inference
- 8GB+ VRAM (varies by model size, 0.6B–8B parameters)
- 16GB+ system RAM
- Local filesystem for audio data and results storage
- Local filesystem for audio datasets (CommonVoice v24.0, Fair-Speech)
- CSV manifests for metadata and predictions
- JSON metadata for analysis results
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Snake case with descriptive names: `run_inference.py`, `compute_fairness_metrics.py`, `error_decomposition.py`
- Utilities and generators end in descriptive verbs: `prepare_dataset.py`, `generate_perturbations.py`, `extract_test_clips.py`
- Validation scripts prefix with `validate_`: `validate_cv_perturbed_test.py`
- Snake case: `compute_group_wer()`, `bootstrap_wer()`, `plot_wer_by_group()`
- Descriptive action-based names: `load_audio()`, `normalize_text()`, `parse_args()`
- Helper/utility functions use underscores: `_extract_granite_transcription()`, `_build_row()`, `_header_written`
- Snake case throughout: `model_registry`, `sample_rate`, `min_group_size`, `output_dir`
- Constants in UPPERCASE: `SAMPLE_RATE = 16000`, `MIN_GROUP_SIZE = 50`, `DATA_DIR = "/users/..."`
- Boolean prefixes with `is_` or `has_`: `is_perturbed`, `_header_written`, `has_demographics`
- Collection suffixes for clarity: `all_results`, `batch_preds`, `group_wers`
- Class names in PascalCase: `IncrementalCSVWriter()`
- No explicit type hints (older Python/flexibility style) but function docstrings document expected types
- Use descriptive strings as dict keys in metadata structures: `"model"`, `"architecture"`, `"generation"`, `"params"`
## Code Style
- No explicit formatter (no `.prettierrc` or `pyproject.toml` found)
- Line length varies (some lines exceed 80 chars)
- Indentation: 4 spaces
- No blank line enforcement between functions/classes observed
- No linter config found (no `.flake8`, `.pylintrc`)
- Follows loose PEP 8 style
- Extensive module-level docstrings for scripts: `"""Compute fairness metrics from ASR inference..."""`
- Section dividers using comment blocks:
- Inline comments for non-obvious logic
- Comment usage: when, why, and implementation details
## Import Organization
- Absolute paths only, no relative imports
- Paths defined as module-level constants: `DATA_DIR`, `RESULTS_DIR`, `PERTURBED_DIR`
- Path construction via `os.path.join()` for cross-platform compatibility
## Error Handling
- Bare `except Exception as e:` for broad error catching in inference loops
- Specific errors logged with `print(f"  WARNING: ...")` for user visibility
- Graceful degradation: missing audio returns `None` or empty string, not crash
- Try/except blocks around model loading and batch processing:
- Division-by-zero protection: `output.substitutions / max(1, output.hits)`
- File existence checks before processing: `if not os.path.isfile(audio_path): continue`
- Scripts use `sys.exit(0)` for success, `sys.exit(1)` for errors
- Validation scripts return 0 if all checks pass, 1 if issues found
## Logging
- Header sections: `print(f"\n{'='*60}")` for major milestones
- Progress reporting: `print(f"  [{i:,}/{n:,}] ({100*i/n:.0f}%)")`
- Warnings prefixed with "WARNING:" or "⚠": `print(f"  WARNING: Failed to load {path}: {e}")`
- Verbose section headers with dashes: `print(f"\n{'─'*60}")`
- Summary metrics at end: `print(f"Overall WER: {overall_wer*100:.2f}% (n={len(valid):,})")`
## Function Design
- Positional for required args: `def compute_group_wer(df: pd.DataFrame) -> dict:`
- Keyword args with defaults for options: `def bootstrap_wer(df: pd.DataFrame, n_bootstrap: int = 1000)`
- Args class for CLI scripts: `def parse_args()` returns parsed namespace
- Functions return dicts for structured results: `{"wer": val, "n_utterances": count, "ci_low": lo, "ci_high": hi}`
- Tuples for multi-value returns: `return np.mean(wers), np.percentile(wers, 2.5), np.percentile(wers, 97.5)`
- None for missing data or on error: `return None, None, None`
- Collections in predictable order (list of dicts, DataFrame)
## Module Design
- No `__all__` declarations observed
- Scripts are executable (entry point is `if __name__ == "__main__": main()`)
- Minimal OOP: only one main class found `IncrementalCSVWriter` in `run_inference.py`
- Class groups related state/methods: `__init__()` stores config, `_build_row()` formats data, `flush()` writes batch
- Model registries as module-level dicts: `MODEL_REGISTRY`, `MODEL_ORDER`, `MODEL_SHORT`, `MODEL_COLORS`
- Perturbation configs as dicts: `PERTURBATION_LABELS`, `PERTURBATION_FAMILIES`, `PERTURBATION_CONFIGS`
- Mapping tables for data normalization: `GENDER_MAP`, `ACCENT_MAP`, `AGE_MAP`
- Numeric thresholds as constants: `MIN_GROUP_SIZE = 50`, `SAMPLE_RATE = 16000`
## Batch Processing Pattern
## Data Validation Pattern
- Collect issues in a list: `issues = []`
- Run multiple checks, append failures
- Report summary at end: `print(f"  Issues: {len(issues)}")`
- Exit with code based on validation: `sys.exit(1 if issues else 0)`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- **Script-oriented:** 28 independent Python scripts organized by functional stage
- **Data-driven:** CSV manifests flow through inference → metrics → visualization stages
- **Multi-model:** Registry-based model abstraction supporting 9 ASR models across 3 generations
- **Perturbation-first:** Systematic degradation of audio conditions to measure robustness disparity
- **Demographic stratification:** Explicit grouping by gender, accent, age, ethnicity to quantify fairness gaps
- **Batch processing:** GPU-optimized batch inference with incremental CSV writing for fault tolerance
## Layers
- Purpose: Standardize datasets and create unified manifests
- Location: `scripts/prepare_*.py` (`prepare_dataset.py`, `prepare_librispeech.py`, `prepare_fairspeech.py`, `prepare_h1_datasets.py`)
- Contains: CSV manifest creation, audio file validation, demographic label standardization
- Depends on: Raw dataset files (Common Voice, LibriSpeech, Fair-Speech), HuggingFace datasets API
- Used by: Inference pipeline
- Purpose: Generate systematically degraded audio for robustness testing
- Location: `scripts/generate_perturbations.py`
- Contains: 4 perturbation types (noise, reverberation, silence, masking) × 3 severity levels = 12 conditions
- Depends on: MUSAN noise pool, RIR (room impulse response) files, SoundFile/NumPy for signal processing
- Used by: Inference pipeline to override audio paths
- Purpose: Generate transcriptions across models and conditions
- Location: `scripts/run_inference.py`
- Contains: Model registry (wav2vec2, Whisper Gen 1/2, Qwen3-ASR, Granite, Canary), loader functions, batch inference loops
- Depends on: HuggingFace transformers, TorchAudio, model-specific SDKs (qwen-asr, NeMo)
- Used by: Metrics computation
- Purpose: Quantify fairness gaps and error patterns
- Location: `scripts/compute_fairness_metrics.py`, `compute_fairness_metrics_fs.py`, `compute_perturbation_metrics.py`, `error_decomposition.py`, `error_decomposition_fs.py`
- Contains: WER aggregation, fairness metrics (max-min ratio, gap %, std), bootstrap CIs, significance tests (Mann-Whitney U), error decomposition (sub/ins/del rates)
- Depends on: JiWER for WER computation, SciPy for statistics, prediction CSVs
- Used by: Visualization and reporting
- Purpose: Generate publication-ready figures and tables
- Location: `scripts/generate_all_plots.py`, `generate_perturbation_plots.py`, `regenerate_figures_large_fonts.py`, `parse_bootstrap_cis.py`
- Contains: Heatmaps, bar charts, line plots, fairness metric summaries
- Depends on: Matplotlib, Seaborn, analysis JSON files
- Used by: Paper/presentation deliverables
- Purpose: Catch inference failures early (WER > 80%, empty outputs, hallucinations)
- Location: `scripts/validate_*.py` (`validate_test_run.py`, `validate_cv_perturbed_test.py`, `validate_perturbed_*.py`)
- Contains: WER threshold checks, empty hypothesis detection, chatbot preamble pattern matching, hallucination length heuristics
- Depends on: Pandas CSV reading
- Used by: Human review during batch inference runs
## Data Flow
- **Manifest:** Single source of truth for dataset definition and audio paths
- **Predictions CSVs:** Intermediate form; incremental writes for fault tolerance
- **Analysis JSONs:** Pre-computed fairness metrics keyed by `model → axis → group`
- **Figures:** Rendered once from analysis JSONs
## Key Abstractions
- Purpose: Abstract over heterogeneous ASR architectures
- Location: `scripts/run_inference.py`, lines 27–91 (`MODEL_REGISTRY` dict)
- Pattern: Each entry maps model name → `{hf_id, generation, architecture, params, type}`
- Examples: `wav2vec2-large` (type: CTC), `whisper-large-v3` (type: Encoder-Decoder), `qwen3-asr-1.7b` (type: LLM-ASR)
- Used to dispatch inference: `if model_info["type"] == "wav2vec2": run_wav2vec2(...)`
- Purpose: Define audio transformation parameters
- Location: `scripts/generate_perturbations.py`, lines 54–71 (`PERTURBATION_CONFIGS` dict)
- Pattern: `{perturbation_label: {type, param1, param2, ...}}`
- Examples:
- Purpose: Fault-tolerant result accumulation
- Location: `scripts/run_inference.py`, lines 148–211
- Pattern: Flushes predicted rows to CSV after each batch, header written once
- Handles resume logic: skip utterances already in output file
- Purpose: Quantify demographic disparity
- Location: `scripts/compute_fairness_metrics.py`, lines 172–195 (`compute_fairness_metrics` function)
- Metrics computed per axis:
## Entry Points
- Location: `scripts/run_inference.py` (main function, line 655)
- Triggers: Manual execution with arguments `--model`, `--dataset`, `--perturbation`
- Responsibilities:
- Location: `scripts/compute_fairness_metrics.py` (main, inferred from structure)
- Triggers: Post-inference analysis to generate JSON report
- Responsibilities: Load all prediction CSVs → compute fairness metrics for each model/axis → save JSON
- Location: `scripts/generate_all_plots.py` (main, inferred)
- Triggers: Rendering from pre-computed analysis JSONs
- Responsibilities: Load JSON → transform to long-form dataframes → matplotlib/seaborn rendering → save PNG/PDF
- Location: `scripts/validate_test_run.py` (main, line 70)
- Triggers: After inference run to catch errors
- Responsibilities: Check WER not > 80%, no empty outputs, no chatbot preambles, no hallucinations
## Error Handling
## Cross-Cutting Concerns
- Uses Whisper's `EnglishTextNormalizer` (case, punctuation, numbers, contractions)
- Applied identically to reference and hypothesis for fair WER comparison
- Location: `scripts/run_inference.py`, lines 136–145
- Deterministic noise selection per utterance via `hashlib.md5(utt_id)` (lines 137–144 in `generate_perturbations.py`)
- Bootstrap RNG seeded with 42 for reproducible CIs (line 155 in `compute_fairness_metrics.py`)
- Fixed batch size (default 16 for inference, configurable)
- Audio lengths variable but padding handled by model processors
- GPU memory auto-scaled for batch size
- Inference supports `--resume` flag to skip completed utterances (lines 708–716 in `run_inference.py`)
- Checks existing CSV for done `utterance_id`s before starting
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
