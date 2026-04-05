# Architecture Patterns

**Domain:** GRPO fairness-aware ASR fine-tuning for Qwen3-ASR-1.7B
**Researched:** 2026-04-05

## Recommended Architecture

### High-Level System View

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXISTING PIPELINE (read-only)                    │
│  prepare_*.py → run_inference.py → compute_fairness_metrics.py     │
│                      → generate_all_plots.py                        │
└──────────────┬──────────────────────────────┬───────────────────────┘
               │ reuse manifest CSVs          │ reuse metrics code
               ▼                              ▼
┌──────────────────────────┐    ┌──────────────────────────────────┐
│   TRAINING SUBSYSTEM     │    │   EVALUATION BRIDGE              │
│                          │    │                                  │
│  ┌──────────────────┐    │    │  infer_with_adapter() ──────┐    │
│  │ FairnessDataLoader│    │    │  (loads LoRA + base model)  │    │
│  │ (demographic-aware│    │    │                             │    │
│  │  batch sampler)   │    │    │  compute_fairness_metrics() │    │
│  └────────┬─────────┘    │    │  (reuse existing code)  ◄───┘    │
│           ▼              │    │                             │    │
│  ┌──────────────────┐    │    │  pareto_frontier_plot()     │    │
│  │ Reward Computer   │    │    └──────────────────────────────────┘
│  │ R=(1-λ)(1-WER)   │    │
│  │  +λ(-|WER_g-WER|)│    │
│  └────────┬─────────┘    │
│           ▼              │
│  ┌──────────────────┐    │
│  │ GRPO Trainer      │    │
│  │ (policy update    │    │
│  │  with clipped     │    │
│  │  ratio + KL)      │    │
│  └──────────────────┘    │
└──────────────────────────┘
```

### Integration Principle

**Do not modify existing scripts.** The 28 existing scripts form a stable benchmarking pipeline. New training code lives in a separate `scripts/training/` subdirectory and imports shared utilities (text normalization, WER computation) from the existing codebase. Evaluation reuses `run_inference.py`'s model loading patterns and `compute_fairness_metrics.py`'s metric functions.

## Component Boundaries

| Component | Responsibility | Communicates With | New vs Existing |
|-----------|---------------|-------------------|-----------------|
| **FairnessDataLoader** | Demographic-stratified batching of audio + transcripts | Reads manifest CSVs; feeds Training Loop | NEW |
| **RewardComputer** | Computes composite accuracy-fairness reward per group | Receives hypotheses from model; uses JiWER for WER | NEW |
| **GRPOTrainer** | Policy gradient updates with clipped ratio + KL penalty | Wraps model forward pass; calls RewardComputer | NEW |
| **BaselineTrainers** | Standard LoRA, FairLoRA, Group-DRO, Fairness-Prompted | Share DataLoader and evaluation; differ in loss/update | NEW |
| **AdapterManager** | LoRA config, loading, saving, merging for Qwen3-ASR | Used by all trainers; produces adapter checkpoints | NEW |
| **EvaluationBridge** | Loads adapter + base model, runs inference, computes fairness metrics | Reuses `infer_qwen3_asr()` pattern + `compute_fairness_metrics()` | NEW (wraps existing) |
| **ParetoPlotter** | Generates accuracy-vs-fairness Pareto frontier across methods/lambdas | Reads evaluation JSONs; produces figures | NEW |
| **Manifest CSVs** | Dataset definition with demographic annotations | Read by DataLoader and EvaluationBridge | EXISTING |
| **run_inference.py** | Model loading patterns, text normalization, WER computation | Pattern reuse (not direct import due to script structure) | EXISTING |
| **compute_fairness_metrics.py** | Fairness metric computation (gap%, ratio, std, bootstrap) | Function reuse or reimport | EXISTING |

## Data Flow

### Training Data Flow

```
1. Manifest CSV (Fair-Speech / Common Voice)
   │  columns: utterance_id, audio_path, sentence, ethnicity, accent, ...
   ▼
2. FairnessDataLoader
   │  - Loads audio as waveform tensors (16kHz, bf16)
   │  - Stratified sampling: each batch has balanced demographic representation
   │  - Groups utterances by demographic for reward computation
   │  - Returns: {audio_tensors, references, demographic_labels, group_ids}
   ▼
3. Model Forward Pass (G times per prompt for GRPO)
   │  - Qwen3-ASR-1.7B with LoRA adapters on decoder layers
   │  - AuT encoder (300M, FROZEN) → projector → Qwen3 LLM decoder (LoRA TRAINABLE)
   │  - Each prompt generates G candidate transcriptions
   │  - Returns: {hypotheses[G], log_probs[G]}
   ▼
4. RewardComputer
   │  - For each candidate: WER_i = jiwer.wer(reference, hypothesis_i)
   │  - Accuracy reward: (1 - WER_i)
   │  - Fairness penalty: -|WER_group - WER_overall|
   │  - Composite: R_i = (1-λ)(1-WER_i) + λ(-|WER_g - WER_mean|)
   │  - Group-normalize: A_i = (R_i - mean(R)) / std(R)
   │  - Returns: {advantages[G], rewards[G]}
   ▼
5. GRPO Policy Update
   │  - Clipped importance ratio: min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
   │  - KL penalty: β * D_KL(π_θ || π_ref)
   │  - Gradient step on LoRA parameters only
   │  - Save reference policy π_ref = π_θ_old at interval
   ▼
6. Checkpoint
   │  - Save LoRA adapter weights (adapter_model.safetensors)
   │  - Save training state (optimizer, scheduler, step)
   │  - Log: step, reward_mean, reward_std, WER, fairness_gap, KL
```

### Evaluation Data Flow

```
1. Adapter Checkpoint (LoRA weights)
   │
   ▼
2. Load Base Model + Merge Adapter
   │  - Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B")
   │  - PeftModel.from_pretrained(base, adapter_path)
   │  - model.merge_and_unload() for inference speed
   ▼
3. Inference (reuse infer_qwen3_asr pattern)
   │  - Manifest CSV → batch audio paths → model.transcribe()
   │  - Write predictions CSV (same format as existing pipeline)
   │  - predictions_{method}_{lambda}.csv
   ▼
4. Fairness Metrics (reuse compute_fairness_metrics.py logic)
   │  - Group by ethnicity/accent → per-group WER
   │  - Compute: max_min_ratio, relative_gap_pct, wer_std
   │  - Bootstrap CIs for significance
   │  - Output: analysis JSON per method/lambda
   ▼
5. Pareto Frontier
   │  - Collect (overall_WER, fairness_gap) for each method × lambda
   │  - Plot accuracy vs fairness across all 5 methods
   │  - Identify Pareto-optimal configurations
```

## Qwen3-ASR-1.7B Internal Architecture (Training Target)

```
┌─────────────────────────────────────────────────────┐
│                Qwen3-ASR-1.7B                       │
│                                                      │
│  ┌──────────────────┐                                │
│  │ AuT Encoder      │  300M params, FROZEN           │
│  │ (AED architecture)│                                │
│  │ Fbank 128-dim     │                                │
│  │ → 8x downsample   │                                │
│  │ → 12.5Hz tokens   │                                │
│  └────────┬─────────┘                                │
│           ▼                                          │
│  ┌──────────────────┐                                │
│  │ Projector        │  Small, FROZEN or TRAINABLE     │
│  │ (linear layer)   │  Maps encoder dim → LLM dim    │
│  └────────┬─────────┘                                │
│           ▼                                          │
│  ┌──────────────────┐                                │
│  │ Qwen3-1.7B LLM  │  ~1.4B params                  │
│  │ Decoder          │                                │
│  │                  │  LoRA on: q_proj, k_proj,       │
│  │  [LoRA adapters] │  v_proj, o_proj, gate_proj,     │
│  │                  │  up_proj, down_proj              │
│  │                  │  rank=8-16, ~1-2% trainable     │
│  └────────┬─────────┘                                │
│           ▼                                          │
│  Token predictions (autoregressive text generation)   │
└─────────────────────────────────────────────────────┘
```

**LoRA placement rationale:** Apply LoRA to the LLM decoder only. The AuT encoder is a specialized audio encoder pretrained on 40M hours -- modifying it risks destroying audio understanding for marginal gains. The decoder is where language generation bias manifests, making it the right target for fairness intervention. This keeps trainable parameters at ~14-28M (1-2% of total), feasible on 16GB VRAM.

**Confidence:** MEDIUM -- Qwen3-ASR uses the `qwen-asr` package which wraps the HuggingFace model. LoRA integration requires accessing the underlying `transformers` model, which the `qwen-asr` package may or may not expose cleanly. This is a key integration risk to validate early.

## Baseline Architectures

All four baselines share the same DataLoader, AdapterManager, and EvaluationBridge. They differ only in their training objective.

### Baseline 1: Standard LoRA (lambda=0)

```
Loss = CrossEntropy(hypothesis, reference)
```

Standard supervised fine-tuning with LoRA. Equivalent to GRPO with lambda=0 but uses SFT instead of RL. This is the "accuracy-only" baseline.

- **Trainer:** Standard HuggingFace `Trainer` with LoRA
- **No demographic awareness** in loss function
- **Purpose:** Establishes what accuracy improvement is possible without fairness

### Baseline 2: FairLoRA

```
Loss = L_CE + λ_fair * Σ_g (L_g - mean(L))²
```

Adds a fairness regularizer that penalizes variance in per-group loss. Adapted from the vision domain (Sukumaran et al., 2024).

- **Trainer:** Custom trainer that computes per-group losses within each batch
- **Requires:** Demographic labels per sample (already in manifests)
- **Key hyperparameter:** λ_fair regularization strength (sweep 0.01-100)
- **Architecturally identical** to Standard LoRA except loss function

### Baseline 3: Group-DRO

```
Loss = max_g L_g(θ)  (optimize for worst-group performance)
```

Distributionally robust optimization that minimizes the maximum group loss rather than average loss. Following Sagawa et al. (2020) with exponential gradient updates on group weights.

- **Trainer:** Custom trainer tracking per-group losses and adjusting group weights
- **Group weights** updated each step: w_g ← w_g * exp(η * L_g), then normalize
- **Key hyperparameter:** η (group weight step size)
- **Does not use RL** -- pure supervised objective with reweighted groups

### Baseline 4: Fairness-Prompted Fine-Tuning (ICASSP 2026)

```
Loss = λ_e * L_ERM + λ_s * L_SD + λ_i * L_IRM + λ_a * L_GroupDRO
```

Combines empirical risk minimization with Spectral Decoupling (SD), Invariant Risk Minimization (IRM), and Group-DRO in a fusion loss. Based on the fairness-prompted approach from the ICASSP 2026 paper (arXiv:2510.18374).

- **Trainer:** Custom trainer computing 4 loss terms per batch
- **Spectral Decoupling:** Adds ||logits||² penalty to reduce overconfident predictions
- **IRM:** Penalizes gradients that vary across demographic groups
- **Reference hyperparameters:** λ_e=1, λ_s=0.06, λ_i=0.01, λ_a=1
- **Most complex baseline** -- multiple interacting loss terms

### GRPO (Primary Method)

```
J = E[ min(ratio * A, clip(ratio) * A) - β * D_KL(π_θ || π_ref) ]
A_i = (R_i - mean(R)) / std(R)
R_i = (1-λ)(1-WER_i) + λ(-|WER_g - WER_mean|)
```

Group Relative Policy Optimization with composite fairness reward. The key differentiator: this is RL-based, not supervised. The model generates multiple candidate transcriptions and is rewarded for both accuracy AND demographic parity.

- **Trainer:** Custom GRPO loop (no HF Trainer -- manual policy gradient)
- **Group size G:** Number of candidate transcriptions per input (start with G=4, memory-permitting)
- **Key hyperparameters:** λ (fairness weight, sweep 0-1), β (KL coefficient), ε (clip range), G (group size)
- **Reference policy:** Frozen copy of initial model weights (bf16, kept on CPU to save VRAM)

## Suggested Directory Structure

```
scripts/
├── training/                          # NEW: all training code
│   ├── __init__.py
│   ├── config.py                      # Shared training configs, paths, hyperparams
│   ├── data_loader.py                 # FairnessDataLoader with demographic batching
│   ├── reward.py                      # RewardComputer (WER + fairness composite)
│   ├── adapter.py                     # AdapterManager (LoRA setup, save/load)
│   ├── train_grpo.py                  # GRPO training loop (primary method)
│   ├── train_standard_lora.py         # Baseline 1: standard SFT + LoRA
│   ├── train_fairlora.py              # Baseline 2: FairLoRA regularizer
│   ├── train_group_dro.py             # Baseline 3: Group-DRO
│   ├── train_fairness_prompted.py     # Baseline 4: ICASSP fusion loss
│   ├── evaluate.py                    # EvaluationBridge (adapter → inference → metrics)
│   └── plot_pareto.py                 # Pareto frontier visualization
├── run_inference.py                   # EXISTING (unchanged)
├── compute_fairness_metrics.py        # EXISTING (unchanged)
└── ...                                # EXISTING (unchanged)

results/
├── training/                          # NEW: training outputs
│   ├── grpo_lambda_0.0/              # adapter weights + training logs
│   ├── grpo_lambda_0.25/
│   ├── grpo_lambda_0.5/
│   ├── grpo_lambda_0.75/
│   ├── grpo_lambda_1.0/
│   ├── standard_lora/
│   ├── fairlora/
│   ├── group_dro/
│   ├── fairness_prompted/
│   └── evaluation/                    # prediction CSVs + analysis JSONs
│       ├── predictions_grpo_0.5_fairspeech.csv
│       └── analysis_all_methods.json
└── figures/
    └── pareto_frontier.pdf            # NEW: key deliverable figure
```

## Patterns to Follow

### Pattern 1: Manifest-as-Contract

**What:** All data flows through CSV manifests with standardized columns. Training code reads the same manifests as inference code.

**Why:** The existing pipeline already uses this pattern. Consistency means training data preparation is zero-cost -- reuse `prepare_fairspeech.py` and `prepare_dataset.py` outputs directly.

**Example:**
```python
# Training loads same manifest as inference
manifest = pd.read_csv("data/fs_manifest.csv")
# Demographic columns already present: ethnicity, accent, gender, age
# Audio paths already validated
# Reference transcripts in 'sentence' column
```

### Pattern 2: Shared Text Normalization

**What:** Use `whisper_normalizer.english.EnglishTextNormalizer` identically in training reward computation and evaluation.

**Why:** WER reward during training must match WER computation during evaluation. Different normalization = reward signal misaligned with evaluation metric.

**Example:**
```python
from whisper_normalizer.english import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()
# Same normalizer in reward.py AND evaluate.py
```

### Pattern 3: Incremental Checkpointing

**What:** Save adapter weights and training state at regular intervals. Resume from last checkpoint on failure.

**Why:** RTX A4000 with 16GB VRAM -- OOM crashes are likely during GRPO with G>2. Lambda sweeps mean many training runs.

**Example:**
```python
# Every N steps
adapter.save_pretrained(f"results/training/grpo_lambda_{lam}/step_{step}")
torch.save(optimizer.state_dict(), f"results/training/grpo_lambda_{lam}/step_{step}/optim.pt")
```

### Pattern 4: Demographic-Aware Batch Sampling

**What:** Each training batch must contain samples from multiple demographic groups to compute the fairness reward meaningfully.

**Why:** The fairness reward R = ... + λ(-|WER_g - WER_mean|) requires computing per-group WER within each batch. If a batch has only one demographic group, the fairness signal is trivially zero.

**Example:**
```python
class DemographicBatchSampler:
    """Ensures each batch has samples from >= K demographic groups."""
    def __init__(self, manifest, group_col="ethnicity", min_groups=3, batch_size=8):
        self.groups = manifest.groupby(group_col).indices
        # Round-robin across groups to fill each batch
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Modifying Existing Scripts

**What:** Adding training logic to `run_inference.py` or `compute_fairness_metrics.py`.

**Why bad:** The existing pipeline is validated for benchmarking 9 models. Changes risk breaking midterm results. Two team members share the codebase.

**Instead:** Create `scripts/training/` as a separate module. Import or copy-paste shared utilities (text normalization, WER computation) rather than modifying originals.

### Anti-Pattern 2: Full-Model GRPO Sampling on GPU

**What:** Keeping both policy model and reference model on GPU simultaneously.

**Why bad:** Qwen3-ASR-1.7B in bf16 = ~3.4GB. With LoRA adapters + optimizer states + G=4 candidate generations + KL computation against reference, 16GB VRAM is extremely tight.

**Instead:** Keep reference model on CPU (or use a frozen copy sharing weights with base model). Use gradient checkpointing. Start with G=2 and increase only if VRAM permits.

### Anti-Pattern 3: Computing Fairness Reward Per-Sample

**What:** Computing the fairness penalty for each individual sample independently.

**Why bad:** Fairness is a group-level property. |WER_g - WER_mean| requires aggregating WER across a demographic group, not a single utterance. Per-sample computation yields a meaningless signal.

**Instead:** Accumulate WER within demographic groups across the batch (or across a window of recent batches), then assign the group-level fairness penalty to all samples in that group.

### Anti-Pattern 4: Using qwen-asr Package for Training

**What:** Trying to backpropagate through the `qwen_asr.Qwen3ASRModel.transcribe()` API.

**Why bad:** The `qwen-asr` package is an inference wrapper. It likely does not expose the underlying model in a way that supports gradient computation through the decoding process.

**Instead:** Load the model directly via HuggingFace `transformers` (AutoModelForCausalLM or the specific Qwen3 model class), attach LoRA via PEFT, and implement the generation + log-probability computation manually. Use the `qwen-asr` package only for final evaluation inference.

## VRAM Budget (RTX A4000, 16GB)

| Component | Estimated VRAM | Notes |
|-----------|---------------|-------|
| Qwen3-ASR-1.7B base (bf16) | ~3.4 GB | Encoder + projector + decoder |
| LoRA adapters (rank 8) | ~14 MB | Negligible |
| Optimizer states (AdamW) | ~28 MB | 2x LoRA params |
| Gradient checkpointing overhead | ~0.5 GB | Recomputation buffers |
| G=2 candidate generations (bf16) | ~1.5 GB | KV cache for 2 sequences |
| G=4 candidate generations (bf16) | ~3.0 GB | KV cache for 4 sequences |
| Reference model KL (CPU offload) | 0 GB | Computed on CPU |
| Audio batch (8 utterances, 10s) | ~0.1 GB | Raw waveforms |
| **Total (G=2)** | **~5.5 GB** | Comfortable headroom |
| **Total (G=4)** | **~7.0 GB** | Feasible with monitoring |

**Recommendation:** Start with G=2 to ensure stability, then test G=4. G=8 likely infeasible without reducing batch size to 1-2.

## Build Order (Dependencies)

The components have strict build-order dependencies. Build bottom-up:

```
Phase 1: Foundation (no training dependencies)
  ├── config.py           (paths, hyperparameters, shared constants)
  ├── adapter.py          (LoRA setup for Qwen3-ASR, save/load)
  └── data_loader.py      (manifest → demographic-aware batches)

Phase 2: Supervised Baselines (needs Phase 1)
  ├── train_standard_lora.py    (simplest: SFT + LoRA, validates adapter works)
  └── evaluate.py               (adapter → inference → metrics bridge)

Phase 3: Fairness Baselines (needs Phase 2 patterns)
  ├── reward.py                 (composite reward function, needed by GRPO too)
  ├── train_fairlora.py         (adds fairness regularizer to Phase 2)
  ├── train_group_dro.py        (adds group reweighting to Phase 2)
  └── train_fairness_prompted.py (fusion loss, most complex baseline)

Phase 4: GRPO (needs Phases 1-3)
  └── train_grpo.py             (RL loop using reward.py + adapter.py)

Phase 5: Analysis
  └── plot_pareto.py            (Pareto frontier across all methods)
```

**Rationale:** Standard LoRA (Phase 2) validates that the adapter mechanism works with Qwen3-ASR before adding complexity. FairLoRA and Group-DRO (Phase 3) are simpler fairness approaches that establish the reward function and demographic batching before GRPO needs them. GRPO (Phase 4) is the most complex and depends on all prior components being validated.

## Integration Points with Existing Code

| Existing Code | What We Reuse | How |
|--------------|--------------|-----|
| `prepare_fairspeech.py` output | `fs_manifest.csv` with demographic columns | Read directly in `data_loader.py` |
| `prepare_dataset.py` output | `cv_test_manifest.csv` | Read directly in `data_loader.py` |
| `run_inference.py` model loading | Pattern for Qwen3-ASR loading + inference | Adapt `load_qwen3_asr()` to return underlying transformers model |
| `run_inference.py` text normalization | `EnglishTextNormalizer` usage | Import same normalizer in `reward.py` and `evaluate.py` |
| `compute_fairness_metrics.py` | WER grouping, fairness metric formulas, bootstrap CIs | Extract into importable functions or reimplement in `evaluate.py` |
| `generate_all_plots.py` | Matplotlib/Seaborn style, color palettes | Follow same visual style for Pareto plot |
| `IncrementalCSVWriter` pattern | Fault-tolerant result writing | Adapt for training checkpoints and evaluation predictions |

**Key risk:** The existing scripts are standalone (no `__init__.py`, no shared module). Functions cannot be directly imported without restructuring. The pragmatic approach: copy the ~20 lines of fairness metric computation into `evaluate.py` rather than refactoring the existing codebase.

## Scalability Considerations

| Concern | Current Scale (class project) | If Scaling Up |
|---------|-------------------------------|--------------|
| Lambda sweep | 5-7 values (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0) | Bayesian optimization over continuous lambda |
| Training data | ~43K utterances (Fair-Speech + CV) | Millions of utterances need streaming dataloader |
| GRPO group size | G=2-4 (VRAM limited) | G=16-64 on multi-GPU for better advantage estimates |
| Baselines | 4 methods, sequential training | Parallel training on multi-GPU cluster |
| Demographic axes | 2 (ethnicity, accent) | Intersectional (ethnicity x accent x gender) |

## Sources

- [GRPO for Speech Recognition (Amazon, ASRU 2025)](https://arxiv.org/abs/2509.01939) -- validates GRPO for ASR with rule-based rewards
- [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v2) -- model architecture: AuT encoder (300M) + projector + Qwen3-1.7B
- [FairLoRA (Sukumaran et al., 2024)](https://arxiv.org/abs/2410.17358) -- per-group loss variance regularizer for LoRA
- [Fairness-Prompted Finetuning (ICASSP 2026)](https://arxiv.org/html/2510.18374) -- ERM + SD + IRM + Group-DRO fusion loss
- [FairGRPO for Clinical Reasoning](https://arxiv.org/abs/2510.19893) -- adaptive importance weighting in GRPO for demographic fairness
- [GRPO Illustrated Breakdown](https://epichka.com/blog/2025/grpo/) -- objective function, clipped ratio, advantage normalization
- [Qwen3-ASR GitHub Repository](https://github.com/QwenLM/Qwen3-ASR) -- model loading, inference API, finetuning directory

---

*Architecture analysis: 2026-04-05*
