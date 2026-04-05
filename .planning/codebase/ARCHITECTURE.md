# Architecture

**Analysis Date:** 2026-04-05

## Pattern Overview

**Overall:** Modular pipeline architecture for ASR fairness research

**Key Characteristics:**
- **Script-oriented:** 28 independent Python scripts organized by functional stage
- **Data-driven:** CSV manifests flow through inference → metrics → visualization stages
- **Multi-model:** Registry-based model abstraction supporting 9 ASR models across 3 generations
- **Perturbation-first:** Systematic degradation of audio conditions to measure robustness disparity
- **Demographic stratification:** Explicit grouping by gender, accent, age, ethnicity to quantify fairness gaps
- **Batch processing:** GPU-optimized batch inference with incremental CSV writing for fault tolerance

## Layers

**Stage 1: Data Preparation**
- Purpose: Standardize datasets and create unified manifests
- Location: `scripts/prepare_*.py` (`prepare_dataset.py`, `prepare_librispeech.py`, `prepare_fairspeech.py`, `prepare_h1_datasets.py`)
- Contains: CSV manifest creation, audio file validation, demographic label standardization
- Depends on: Raw dataset files (Common Voice, LibriSpeech, Fair-Speech), HuggingFace datasets API
- Used by: Inference pipeline

**Stage 2: Audio Perturbation**
- Purpose: Generate systematically degraded audio for robustness testing
- Location: `scripts/generate_perturbations.py`
- Contains: 4 perturbation types (noise, reverberation, silence, masking) × 3 severity levels = 12 conditions
- Depends on: MUSAN noise pool, RIR (room impulse response) files, SoundFile/NumPy for signal processing
- Used by: Inference pipeline to override audio paths

**Stage 3: ASR Inference**
- Purpose: Generate transcriptions across models and conditions
- Location: `scripts/run_inference.py`
- Contains: Model registry (wav2vec2, Whisper Gen 1/2, Qwen3-ASR, Granite, Canary), loader functions, batch inference loops
- Depends on: HuggingFace transformers, TorchAudio, model-specific SDKs (qwen-asr, NeMo)
- Used by: Metrics computation

**Stage 4: Metrics & Analysis**
- Purpose: Quantify fairness gaps and error patterns
- Location: `scripts/compute_fairness_metrics.py`, `compute_fairness_metrics_fs.py`, `compute_perturbation_metrics.py`, `error_decomposition.py`, `error_decomposition_fs.py`
- Contains: WER aggregation, fairness metrics (max-min ratio, gap %, std), bootstrap CIs, significance tests (Mann-Whitney U), error decomposition (sub/ins/del rates)
- Depends on: JiWER for WER computation, SciPy for statistics, prediction CSVs
- Used by: Visualization and reporting

**Stage 5: Visualization & Reporting**
- Purpose: Generate publication-ready figures and tables
- Location: `scripts/generate_all_plots.py`, `generate_perturbation_plots.py`, `regenerate_figures_large_fonts.py`, `parse_bootstrap_cis.py`
- Contains: Heatmaps, bar charts, line plots, fairness metric summaries
- Depends on: Matplotlib, Seaborn, analysis JSON files
- Used by: Paper/presentation deliverables

**Stage 6: Validation & Sanity Checks**
- Purpose: Catch inference failures early (WER > 80%, empty outputs, hallucinations)
- Location: `scripts/validate_*.py` (`validate_test_run.py`, `validate_cv_perturbed_test.py`, `validate_perturbed_*.py`)
- Contains: WER threshold checks, empty hypothesis detection, chatbot preamble pattern matching, hallucination length heuristics
- Depends on: Pandas CSV reading
- Used by: Human review during batch inference runs

## Data Flow

**Primary Inference Pipeline:**

1. **Input:** Dataset manifest CSV
   - Columns: `utterance_id`, `audio_path`, `sentence` (reference), demographic fields (`gender`, `accent`, `age`, `ethnicity`, etc.)
   - Example files: `cv_test_manifest.csv` (Common Voice), `fs_manifest.csv` (Fair-Speech)

2. **Perturbation (optional):**
   - For each perturbation condition (e.g., `snr_20db`), `generate_perturbations.py` overwrites audio paths in manifest
   - Perturbed audio saved to `/perturbed_audio/{dataset}/{perturbation}/`
   - Original audio paths preserved as fallback

3. **Inference:** `run_inference.py`
   - Loads manifest → batch loads audio files
   - Dispatches to model-specific runner (wav2vec2, Whisper, Qwen3, Granite, Canary)
   - Each runner: preprocesses audio → model forward pass → decode predictions
   - **Incremental writing:** Flushes batch predictions to CSV after each batch (prevents data loss on timeout)
   - Output: `predictions_{model_key}_{perturbation}.csv` with columns:
     - `utterance_id`, `reference`, `hypothesis`, `hypothesis_raw`
     - `wer`, `num_hyp_words`, `num_ref_words`
     - Demographics: `gender`, `accent`, `age`, `ethnicity`, etc.
     - Model metadata: `model`, `generation`, `architecture`

4. **Metrics Computation:**
   - `compute_fairness_metrics.py`: Groups predictions by demographic axis → aggregates WER per group → calculates fairness metrics
   - `compute_perturbation_metrics.py`: Compares clean vs. perturbed WER by demographic to quantify fairness gap amplification
   - `error_decomposition.py`: Breaks down errors into substitutions/insertions/deletions per demographic group
   - Output: JSON files with per-model, per-group statistics

5. **Visualization:**
   - `generate_all_plots.py`: Reads analysis JSONs → produces heatmaps, bar charts
   - Figures saved as PNG (150 DPI) and PDF

**State Management:**
- **Manifest:** Single source of truth for dataset definition and audio paths
- **Predictions CSVs:** Intermediate form; incremental writes for fault tolerance
- **Analysis JSONs:** Pre-computed fairness metrics keyed by `model → axis → group`
- **Figures:** Rendered once from analysis JSONs

## Key Abstractions

**Model Registry:**
- Purpose: Abstract over heterogeneous ASR architectures
- Location: `scripts/run_inference.py`, lines 27–91 (`MODEL_REGISTRY` dict)
- Pattern: Each entry maps model name → `{hf_id, generation, architecture, params, type}`
- Examples: `wav2vec2-large` (type: CTC), `whisper-large-v3` (type: Encoder-Decoder), `qwen3-asr-1.7b` (type: LLM-ASR)
- Used to dispatch inference: `if model_info["type"] == "wav2vec2": run_wav2vec2(...)`

**Perturbation Config:**
- Purpose: Define audio transformation parameters
- Location: `scripts/generate_perturbations.py`, lines 54–71 (`PERTURBATION_CONFIGS` dict)
- Pattern: `{perturbation_label: {type, param1, param2, ...}}`
- Examples:
  - `snr_20db: {type: "noise", snr_db: 20}`
  - `reverb_0.3s: {type: "reverb", rt60: "0.3s"}`
  - `silence_25pct: {type: "silence", fraction: 0.25}`

**IncrementalCSVWriter:**
- Purpose: Fault-tolerant result accumulation
- Location: `scripts/run_inference.py`, lines 148–211
- Pattern: Flushes predicted rows to CSV after each batch, header written once
- Handles resume logic: skip utterances already in output file

**Fairness Metrics:**
- Purpose: Quantify demographic disparity
- Location: `scripts/compute_fairness_metrics.py`, lines 172–195 (`compute_fairness_metrics` function)
- Metrics computed per axis:
  - `max_min_ratio`: WER_worst_group / WER_best_group
  - `relative_gap_pct`: (WER_worst - WER_best) / WER_best × 100
  - `wer_std`: standard deviation across groups
  - Per-group WER with bootstrap 95% CI

## Entry Points

**Command-line Inference:**
- Location: `scripts/run_inference.py` (main function, line 655)
- Triggers: Manual execution with arguments `--model`, `--dataset`, `--perturbation`
- Responsibilities:
  1. Parse CLI args
  2. Load manifest CSV
  3. Override audio paths if perturbed condition
  4. Filter missing files
  5. Initialize IncrementalCSVWriter
  6. Dispatch to model-specific runner
  7. Log summary WER by demographic

**Batch Analysis:**
- Location: `scripts/compute_fairness_metrics.py` (main, inferred from structure)
- Triggers: Post-inference analysis to generate JSON report
- Responsibilities: Load all prediction CSVs → compute fairness metrics for each model/axis → save JSON

**Figure Generation:**
- Location: `scripts/generate_all_plots.py` (main, inferred)
- Triggers: Rendering from pre-computed analysis JSONs
- Responsibilities: Load JSON → transform to long-form dataframes → matplotlib/seaborn rendering → save PNG/PDF

**Validation Checks:**
- Location: `scripts/validate_test_run.py` (main, line 70)
- Triggers: After inference run to catch errors
- Responsibilities: Check WER not > 80%, no empty outputs, no chatbot preambles, no hallucinations

## Error Handling

**Strategy:** Graceful degradation with detailed logging

**Patterns:**
1. **Audio Loading Failures:** Log warning, append empty hypothesis to batch, continue (line 226, `run_inference.py`)
2. **Batch Inference Failures:** Catch exception, replace batch with empty predictions, log warning, continue (lines 286–288)
3. **WER Computation Edge Cases:** If reference or hypothesis empty, WER = 1.0 or 0.0 (lines 178–179)
4. **Bootstrap Failures:** Skip utterance if WER computation throws (lines 160–163 in `compute_fairness_metrics.py`)
5. **Missing Demographic Data:** Filter groups with < 50 samples; skip if zero (lines 174–176)

## Cross-Cutting Concerns

**Text Normalization:** 
- Uses Whisper's `EnglishTextNormalizer` (case, punctuation, numbers, contractions)
- Applied identically to reference and hypothesis for fair WER comparison
- Location: `scripts/run_inference.py`, lines 136–145

**Sampling & Reproducibility:**
- Deterministic noise selection per utterance via `hashlib.md5(utt_id)` (lines 137–144 in `generate_perturbations.py`)
- Bootstrap RNG seeded with 42 for reproducible CIs (line 155 in `compute_fairness_metrics.py`)

**Batch Processing:**
- Fixed batch size (default 16 for inference, configurable)
- Audio lengths variable but padding handled by model processors
- GPU memory auto-scaled for batch size

**Resume & Checkpointing:**
- Inference supports `--resume` flag to skip completed utterances (lines 708–716 in `run_inference.py`)
- Checks existing CSV for done `utterance_id`s before starting

---

*Architecture analysis: 2026-04-05*
