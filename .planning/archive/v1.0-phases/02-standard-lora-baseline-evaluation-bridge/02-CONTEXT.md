# Phase 2: Standard LoRA Baseline + Evaluation Bridge - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Train a standard LoRA adapter (SFT, lambda=0, no fairness reward) on Qwen3-ASR-1.7B and build a standalone evaluation script that produces per-group fairness metrics on Fair-Speech, Common Voice, and LibriSpeech. This establishes the accuracy-only baseline that all fairness methods (GRPO, FairLoRA, Group-DRO, ICASSP) will be compared against.

</domain>

<decisions>
## Implementation Decisions

### Training data composition
- **D-01:** Train on both Fair-Speech and Common Voice combined. Fair-Speech provides ethnicity axis, Common Voice provides accent axis.
- **D-02:** Use 1-2K sample prototype subset for initial training (~500-1K from each dataset, equal split). Follow ML prototyping skill: validate pipeline at small scale before committing to full dataset.
- **D-03:** Speaker-disjoint train/eval splits to prevent speaker memorization (Pitfall #9).
- **D-04:** Three-dataset strategy: train on FS+CV, evaluate on all three (FS, CV, LibriSpeech). LibriSpeech serves as a held-out accuracy benchmark without demographic labels.

### Hyperparameter selection
- **D-05:** Run Optuna HP sweep first using existing `phase2_hp_sweep.py` (20 trials, 100 steps/trial, 500-sample subset via TPE).
- **D-06:** Follow ML prototyping skill 4-phase process: (1) pipeline sanity [done], (2) HP sweep, (3) validate top 2-3 configs on full 1-2K subset, (4) final training with locked config. Claude decides step counts per phase.
- **D-07:** Sweep the existing 6 hyperparameters as-is: learning rate (1e-5 to 1e-3 log), LoRA rank (4/8/16/32), alpha ratio (1x/2x rank), dropout (0.0-0.15), target MLP (true/false), weight decay (1e-6 to 1e-2 log).

### Evaluation script
- **D-08:** New standalone script `scripts/training/evaluate_adapter.py`. Takes --adapter_path and --manifest as input. Loads base model + LoRA adapter, runs inference, computes per-group WER and fairness metrics. Does NOT modify existing benchmarking scripts.
- **D-09:** Output format matches existing benchmarking pipeline (same prediction CSV columns and analysis JSON structure). Enables direct comparison with midterm results and reuse of existing plotting scripts.

### Experiment tracking
- **D-10:** Use W&B for training metrics (loss, LR, VRAM, HP sweep results). Write final evaluation results to CSV/JSON files for cross-method comparison.

### Claude's Discretion
- Exact step counts for HP sweep validation and final training (guided by ML prototyping skill)
- W&B project naming and run organization
- Specific demographic group minimum sample sizes for evaluation
- Whether to include gradient accumulation for larger effective batch sizes

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Research and architecture
- `.planning/research/SUMMARY.md` — Overall project architecture, Phase 2 rationale, component boundaries
- `.planning/research/ARCHITECTURE.md` — Training subsystem and evaluation bridge component design, data flow diagrams
- `.planning/research/PITFALLS.md` — Pitfall #9 (evaluation methodology, speaker-disjoint splits), Pitfall #4 (VRAM exhaustion)

### ML prototyping methodology
- ML prototyping skill (ml-prototype) — 4-phase process: sanity → HP sweep → validate → full run. Guides step counts and data subsetting.

### Existing training infrastructure
- `scripts/training/phase2_hp_sweep.py` — Existing Optuna HP sweep script with 6 HPs defined
- `scripts/training/phase1_sanity.py` — Pipeline sanity checks (prerequisite, already built)
- `scripts/training/lora_prototype.py` — LoRA training patterns for Qwen3-ASR
- `scripts/training/data_loader.py` — ASRFairnessDataset, DemographicStratifiedSampler
- `scripts/training/data_collator.py` — DataCollatorForQwen3ASR

### Existing evaluation pipeline
- `scripts/compute_fairness_metrics.py` — Per-group WER, gap%, max-min ratio, bootstrap CIs (output format to match)
- `scripts/run_inference.py` — Model loading patterns, IncrementalCSVWriter, text normalization

### Requirements
- `.planning/REQUIREMENTS.md` — BASE-01, EVAL-01, EVAL-02, EVAL-03, EVAL-04

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/training/phase2_hp_sweep.py`: Optuna sweep already written with model loading, LoRA application, and stratified subsetting. Can run as-is for HP selection.
- `scripts/training/data_loader.py`: ASRFairnessDataset handles both FS (ethnicity) and CV (accent) manifests with soundfile audio loading.
- `scripts/training/data_collator.py`: DataCollatorForQwen3ASR handles variable-length audio padding and label preparation.
- `scripts/training/lora_prototype.py`: `patch_outer_forward()`, `print_gpu_memory()`, and LoRA application patterns are proven and reusable.
- `scripts/compute_fairness_metrics.py`: Fairness metric computation (gap%, ratio, std, bootstrap CIs) — evaluation script should produce compatible output.
- `scripts/training/generate_manifests.py`: Creates train/eval manifests from raw datasets.

### Established Patterns
- Forward patching: `model.thinker.forward()` delegation for HuggingFace Trainer compatibility
- Audio loading: soundfile.read() (not torchaudio.load due to torchcodec dependency issue)
- Encoder freezing: Freeze audio_tower params, re-freeze after PEFT wrapping (PEFT may add LoRA to encoder q/k/v_proj)
- Gradient checkpointing: Always enabled with `use_reentrant=False`
- VRAM profiling: `print_gpu_memory()` at key training checkpoints

### Integration Points
- Manifests: `outputs/manifests/fs_train.csv`, `outputs/manifests/cv_train.csv`, `outputs/manifests/cv_dev.csv`
- Model: `Qwen/Qwen3-ASR-1.7B` loaded via qwen_asr.Qwen3ASRModel
- Output directory convention: `outputs/` for training artifacts

</code_context>

<specifics>
## Specific Ideas

- Follow the ML prototyping skill's 4-phase process strictly — do not scale up until each gate passes
- The user emphasized checking dataset sizes carefully (Common Voice raw is ~1.15M utterances, much larger than Fair-Speech's ~26K)
- Existing `phase2_hp_sweep.py` already has the Optuna sweep logic — reuse rather than rewrite

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-standard-lora-baseline-evaluation-bridge*
*Context gathered: 2026-04-06*
