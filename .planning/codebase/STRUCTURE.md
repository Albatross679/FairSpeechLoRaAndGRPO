# Codebase Structure

**Analysis Date:** 2026-04-05

## Directory Layout

```
/workspace/NLPClassProject/
├── scripts/                    # All 28 Python scripts + shell utilities
│   ├── run_inference.py        # Core ASR inference engine
│   ├── compute_fairness_metrics.py    # Metrics aggregation
│   ├── generate_perturbations.py      # Audio degradation
│   ├── generate_all_plots.py          # Unified visualization
│   ├── error_decomposition.py         # Error-type analysis
│   ├── prepare_*.py            # Dataset preparation suite
│   ├── validate_*.py           # Sanity check validation
│   ├── download_*.sh           # Data download utilities
│   └── setup_nemo.sh           # NeMo dependency setup
├── .planning/
│   └── codebase/               # Architecture documentation
├── colm2026_conference.pdf     # Paper PDF
└── scripts.zip                 # Backup archive
```

## Directory Purposes

**`/workspace/NLPClassProject/scripts/`:**
- Purpose: Core research pipeline — no shared utilities, all scripts are standalone modules
- Contains: Python inference/analysis scripts, shell setup/download utilities
- Key files: See "Key File Locations" below
- Organization: Functional grouping (data prep → inference → metrics → viz)

**`/workspace/NLPClassProject/.planning/codebase/`:**
- Purpose: GSD documentation generated during mapping
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md (future), TESTING.md (future)

## Key File Locations

**Entry Points & Main Pipelines:**

- `scripts/run_inference.py` (790 lines): Core ASR inference engine
  - Loads manifest CSV, dispatches to model runners, writes predictions incrementally
  - Supports 9 models across 3 generations with pluggable architecture
  - Usage: `python scripts/run_inference.py --model whisper-large-v3 --device cuda`

- `scripts/compute_fairness_metrics.py` (550+ lines): Fairness metrics aggregation
  - Groups predictions by demographic axis, computes WER per group, derives fairness gaps
  - Runs bootstrap CIs and Mann-Whitney U significance tests
  - Output: JSON with per-model, per-axis statistics

- `scripts/generate_perturbations.py` (500+ lines): Audio perturbation generation
  - Produces 12 perturbed variants per utterance (noise, reverb, silence, masking)
  - Deterministic noise selection by utterance hash for reproducibility
  - Writes to `{PERTURBED_DIR}/{dataset}/{perturbation}/`

- `scripts/generate_all_plots.py` (750+ lines): Unified visualization pipeline
  - Renders 12+ publication-quality figures from pre-computed analysis JSONs
  - Figures: ethnicity heatmaps, accent WER distributions, generation comparisons, etc.

- `scripts/error_decomposition.py` (200 lines): Error-type breakdown
  - Computes substitution/insertion/deletion rates per demographic group
  - Tests hypothesis about Gen-3 error profiles

**Dataset Preparation:**

- `scripts/prepare_dataset.py`: Common Voice v24.0 → manifest CSV
  - Standardizes demographic labels (gender, accent, age)
  - Filters missing audio files
  - Output: `cv_test_manifest.csv`

- `scripts/prepare_librispeech.py`: LibriSpeech test-clean → manifest CSV
  - Loads from HuggingFace, converts to 16kHz WAV files
  - Output: `ls_manifest.csv`

- `scripts/prepare_fairspeech.py`: Fair-Speech dataset → manifest CSV
  - Includes ethnicity, socioeconomic, L1 language metadata
  - Output: `fs_manifest.csv`

- `scripts/prepare_h1_datasets.py`: Hypothesis 1 test datasets
  - Curates subsets for specific H1 claims (accents, ages, socioeconomic)

**Validation & Testing:**

- `scripts/validate_test_run.py`: Post-inference sanity checks
  - Checks WER < 80%, no empty outputs, no hallucinations
  - Usage: `python scripts/validate_test_run.py --output_dir results/`

- `scripts/validate_cv_perturbed_test.py`: Validates Common Voice perturbed predictions
- `scripts/validate_perturbed_test.py`: Generic perturbed condition validator
- `scripts/validate_perturbed_*.py`: Condition-specific validators (silence, reverb, masking)

**Utilities & Post-Processing:**

- `scripts/parse_bootstrap_cis.py`: Parse and format bootstrap confidence intervals
- `scripts/generate_perturbation_plots.py`: Perturbation-specific visualization
- `scripts/regenerate_figures_large_fonts.py`: Re-renders existing figures with larger fonts
- `scripts/whisper_hallucination_analysis.py`: Detailed hallucination classification
- `scripts/extract_test_clips.py`: Audio file extraction utilities
- `scripts/prepare_overleaf.py`: Export figures/tables for LaTeX

**Setup & Infrastructure:**

- `scripts/download_musan.sh`: Downloads MUSAN noise corpus (for perturbations)
- `scripts/download_rirs.sh`: Downloads room impulse responses (for reverb)
- `scripts/setup_nemo.sh`: Installs NeMo toolkit for Canary model
- `scripts/launch_priority_batch.sh`: HPC batch job submission template

## Naming Conventions

**Files:**

- **Inference runners:** `run_*.py` → `run_inference.py`
- **Metrics/analysis:** `compute_*.py` → `compute_fairness_metrics.py`, `compute_perturbation_metrics.py`
- **Data preparation:** `prepare_*.py` → `prepare_dataset.py`, `prepare_librispeech.py`, `prepare_fairspeech.py`
- **Visualization:** `generate_*.py` → `generate_all_plots.py`, `generate_perturbation_plots.py`
- **Error analysis:** `error_*.py` → `error_decomposition.py`
- **Validation:** `validate_*.py` → `validate_test_run.py`, `validate_cv_perturbed_test.py`
- **Setup/utilities:** `*.sh` → `download_*.sh`, `setup_*.sh`

**Directories:**
- Input data: `data/` (manifests CSV files)
- Perturbed audio: `perturbed_audio/{dataset}/{perturbation}/`
- Results: `results/{dataset}/` (prediction CSVs)
- Analysis: `results/{dataset}/analysis/` (JSON analysis files)
- Figures: `results/figures/` (PNG/PDF outputs)

**Constants & Paths:**
- Model paths: HuggingFace model IDs (e.g., `facebook/wav2vec2-large-960h-lv60-self`)
- Audio sample rate: 16000 Hz (hardcoded as `SAMPLE_RATE`)
- Data directories: Set in each script (e.g., `/users/PAS2030/srishti/asr_fairness/data/`)

## Where to Add New Code

**New ASR Model:**
1. Add entry to `MODEL_REGISTRY` in `scripts/run_inference.py` (lines 27–91)
   ```python
   "new-model-name": {
       "hf_id": "org/model-id",
       "generation": 3,
       "architecture": "Description",
       "params": "2B",
       "type": "new_type",  # or reuse existing type
   }
   ```
2. Create loader + inference functions:
   - `load_new_model(args)` → return `{"model": model, "processor": processor}`
   - `infer_new_model(manifest_df, args, loaded, writer=None)` → batch inference loop
3. Add dispatch in `main()` function (around line 725):
   ```python
   elif model_info["type"] == "new_type":
       run_new_model(df, args, writer=writer)
   ```

**New Perturbation Type:**
1. Add to `PERTURBATION_CONFIGS` in `scripts/generate_perturbations.py` (lines 54–71)
   ```python
   "new_perturbation_Xdb": {"type": "new_type", "param": value},
   ```
2. Implement audio transformation function:
   - Follow pattern of `apply_noise()`, `apply_reverb()`, etc. (around lines 200+)
   - Takes numpy audio array → returns perturbed array
3. Call in main loop (around line 400+)

**New Fairness Metric:**
1. Add computation function in `scripts/compute_fairness_metrics.py`
   - Example: `compute_max_min_ratio()` (lines 172–195)
2. Call in `compute_fairness_metrics()` function to populate result dict
3. Export to JSON for visualization

**New Demographic Axis:**
1. Add column to manifest CSV (e.g., `new_demographic`)
2. Add to `DEMOGRAPHIC_AXES` dict in `scripts/compute_perturbation_metrics.py` (lines 64–67)
3. Filter and group in metric computation loops (standard pandas groupby pattern)
4. Add to visualization plot functions in `scripts/generate_all_plots.py`

## Special Directories

**`scripts/__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes (auto-created by Python)
- Committed: No (in `.gitignore`)

**`perturbed_audio/`:** (exists at runtime)
- Purpose: Stores generated perturbed audio files
- Structure: `{dataset}/{perturbation}/{utterance_id}.wav`
- Generated: Yes (by `generate_perturbations.py`)
- Size: ~100GB+ for full Fair-Speech/Common Voice with all 12 perturbations

**`results/`:** (exists at runtime)
- Purpose: Prediction CSVs and analysis outputs
- Structure:
  - `{dataset}/predictions_*.csv` (inference outputs)
  - `{dataset}/analysis/full_analysis*.json` (aggregated metrics)
  - `figures/` (PNG/PDF plots)
- Generated: Yes (by `run_inference.py` and analysis scripts)

## Data Flow Across Files

**Inference Pipeline:**
1. `prepare_dataset.py` → `cv_test_manifest.csv`
2. `generate_perturbations.py` → reads manifest, writes `/perturbed_audio/{dataset}/{perturbation}/`
3. `run_inference.py` → reads manifest, reads audio from `perturbed_audio/` (if perturbed), writes `results/{dataset}/predictions_*.csv`
4. `compute_fairness_metrics.py` → reads `predictions_*.csv`, writes `results/{dataset}/analysis/full_analysis.json`
5. `generate_all_plots.py` → reads `full_analysis.json`, writes `results/figures/*.png`

**Key Manifest Columns:**
- Required: `utterance_id`, `audio_path`, `sentence` (reference text)
- Optional demographics: `gender`, `accent`, `age`, `ethnicity`, `first_language`, `socioeconomic`, `l1_group`

## Configuration & Defaults

**Hardcoded Paths:**
- Data directory: `/users/PAS2030/srishti/asr_fairness/data/`
- Results directory: `/users/PAS2030/srishti/asr_fairness/results/commonvoice` or `fairspeech`
- Perturbed audio: `/users/PAS2030/srishti/asr_fairness/perturbed_audio/`

**Runtime Parameters (CLI args in `run_inference.py`):**
- `--model`: Model name (required)
- `--manifest`: Path to manifest CSV (default: `{DATA_DIR}/cv_test_manifest.csv`)
- `--output_dir`: Results directory
- `--batch_size`: Inference batch size (default: 16)
- `--device`: cuda or cpu (default: cuda)
- `--perturbation`: Single perturbation condition (default: clean)
- `--perturbations`: Comma-separated list to run sequentially
- `--resume`: Skip completed utterances (default: False)

**Constants:**
- Sample rate: 16000 Hz
- Min group size for reporting: 50 utterances
- Bootstrap iterations: 1000
- Max WER for validation: 80%

---

*Structure analysis: 2026-04-05*
