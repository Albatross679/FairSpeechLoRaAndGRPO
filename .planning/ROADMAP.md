# Roadmap: GRPO Fairness-Aware ASR Fine-Tuning

## Overview

This roadmap delivers a GRPO fairness-aware fine-tuning pipeline for Qwen3-ASR-1.7B. Phase 1 validates the critical risk (LoRA on Qwen3-ASR decoder) and proves the full adapter lifecycle with a standard LoRA baseline. Phase 2 builds the core GRPO training loop with composite fairness reward. Phase 3 implements the three fairness baselines for comparison. Phase 4 runs the lambda sweep, produces the Pareto frontier, and completes strengthening experiments. Every phase builds on proven infrastructure from the phase before it.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1.0: Prepare Fine-Tuning Dataset** - Build train/eval splits with demographic labels for Fair-Speech and Common Voice, create data loaders for fine-tuning
- [ ] **Phase 1.1: LoRA Prototype** - Attach LoRA to Qwen3-ASR decoder, validate forward pass and VRAM on a small subset, test adapter save/load lifecycle
- [ ] **Phase 1.2: Standard LoRA Baseline** - Scale up to full training, train standard LoRA baseline (lambda=0), evaluate with fairness metrics on both datasets
- [ ] **Phase 2: GRPO Training Loop** - Implement custom GRPO loop with composite accuracy-fairness reward and demographic-stratified batching
- [ ] **Phase 3: Fairness Baselines** - Implement FairLoRA, Group-DRO, and ICASSP fairness-prompted baselines using shared adapter infrastructure
- [ ] **Phase 4: Lambda Sweep and Pareto Frontier** - Run lambda sweep, produce Pareto frontier visualization, complete statistical testing and ablations

## Phase Details

### Phase 1.0: Prepare Fine-Tuning Dataset
**Goal**: Train and eval splits with demographic labels are ready for fine-tuning, with data loaders that yield (audio, transcript, demographic_group) tuples
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. Fair-Speech and Common Voice 24 train/eval splits are created with demographic labels (ethnicity, accent) preserved
  2. Data loaders yield batches of (audio_tensor, reference_transcript, demographic_group) suitable for training
  3. Splits are balanced enough that each demographic group has sufficient representation for per-group metrics
**Plans**: 2 plans
Plans:
- [x] 01.0-01-PLAN.md -- Create Fair-Speech train/eval splits and validation script
- [x] 01.0-02-PLAN.md -- Build PyTorch Dataset and DataLoader with demographic-stratified sampling

### Phase 1.1: LoRA Prototype
**Goal**: LoRA attaches to Qwen3-ASR-1.7B decoder and produces valid transcriptions within VRAM budget, validated on a small data subset
**Depends on**: Phase 1.0
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05
**Success Criteria** (what must be TRUE):
  1. LoRA adapter attaches to Qwen3-ASR-1.7B decoder and forward pass produces valid transcriptions (not garbage text)
  2. Peak VRAM under 14GB on RTX A4000 with gradient checkpointing enabled
  3. Adapter save/load/merge lifecycle works end-to-end (train a few steps -> save -> load -> produce identical output)
  4. A short training run (e.g., 50-100 steps) on a small subset shows decreasing loss
**Plans**: 2 plans
Plans:
- [ ] 01.1-01-PLAN.md -- LoRA model setup, data collator, and prototype training script
- [ ] 01.1-02-PLAN.md -- Validation script and end-to-end GPU verification

### Phase 1.2: Standard LoRA Baseline
**Goal**: Standard LoRA baseline is fully trained and evaluated with fairness metrics on both datasets
**Depends on**: Phase 1.1
**Requirements**: BASE-01, EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. Standard LoRA training (lambda=0) runs to completion without OOM on RTX A4000
  2. Trained adapter achieves reasonable WER (competitive with or better than base model)
  3. Evaluation produces WER and fairness metrics (max-min ratio, gap%, std, bootstrap CIs) for both ethnicity and accent axes on Fair-Speech and Common Voice 24
  4. Results are saved in a format compatible with the existing analysis pipeline
**Plans**: 2 plans
Plans:
- [x] 01.0-01-PLAN.md -- Create Fair-Speech train/eval splits and validation script
- [x] 01.0-02-PLAN.md -- Build PyTorch Dataset and DataLoader with demographic-stratified sampling

### Phase 2: GRPO Training Loop
**Goal**: Users can train Qwen3-ASR with GRPO using a composite fairness reward and observe per-group WER dynamics during training
**Depends on**: Phase 1.2
**Requirements**: GRPO-01, GRPO-02, GRPO-03, GRPO-04, GRPO-05, GRPO-06, GRPO-07, GRPO-08, GRPO-09, GRPO-10
**Success Criteria** (what must be TRUE):
  1. GRPO training loop generates G candidate outputs per input, computes composite reward R = (1-lambda)(1-WER) + lambda(-|WER_g - WER_mean|), and updates only LoRA parameters
  2. Training with lambda=0.5 runs to completion and produces an adapter whose per-group WER spread is measurably different from the standard LoRA baseline
  3. Per-group WER is logged every N steps during training and shows interpretable dynamics (not flat or divergent)
  4. Dr. GRPO variant (no std normalization, length normalization) is selectable via config and produces a trained adapter
**Plans**: 2 plans
Plans:
- [x] 01.0-01-PLAN.md -- Create Fair-Speech train/eval splits and validation script
- [x] 01.0-02-PLAN.md -- Build PyTorch Dataset and DataLoader with demographic-stratified sampling

### Phase 3: Fairness Baselines
**Goal**: All three fairness baselines are trained and evaluated, producing results directly comparable to the standard LoRA and GRPO methods
**Depends on**: Phase 1.2
**Requirements**: BASE-02, BASE-03, BASE-04, EVAL-05, EVAL-06
**Success Criteria** (what must be TRUE):
  1. FairLoRA adapter is trained with group-variance regularization and evaluated with full fairness metrics on both datasets
  2. Group-DRO adapter is trained with smoothed exponential group weight updates and evaluated with full fairness metrics on both datasets
  3. ICASSP fairness-prompted adapter is trained with the four-term fusion loss and evaluated with full fairness metrics on both datasets
  4. Perturbation robustness evaluation runs under existing 12 conditions for all trained adapters (standard LoRA, GRPO, and all baselines)
  5. Error decomposition (sub/ins/del rates) by demographic group is computed for all trained models
**Plans**: 2 plans
Plans:
- [x] 01.0-01-PLAN.md -- Create Fair-Speech train/eval splits and validation script
- [ ] 01.0-02-PLAN.md -- Build PyTorch Dataset and DataLoader with demographic-stratified sampling

### Phase 4: Lambda Sweep and Pareto Frontier
**Goal**: The central experimental result is produced: a Pareto frontier plotting accuracy vs. fairness across all methods and lambda values, with statistical backing
**Depends on**: Phase 2, Phase 3
**Requirements**: EXPT-01, EXPT-02, EXPT-03, EXPT-04
**Success Criteria** (what must be TRUE):
  1. Lambda sweep across 6+ values produces trained GRPO adapters that trace a visible accuracy-fairness tradeoff curve
  2. Pareto frontier scatter plot shows all 5 methods (Standard LoRA, GRPO variants, FairLoRA, Group-DRO, ICASSP) with accuracy on one axis and fairness metric on the other
  3. Bootstrap pairwise significance tests determine which method differences are statistically significant
  4. Hyperparameter sensitivity ablation over G, learning rate, and beta (KL coefficient) is documented with results
**Plans**: 2 plans
Plans:
- [ ] 01.0-01-PLAN.md -- Create Fair-Speech train/eval splits and validation script
- [ ] 01.0-02-PLAN.md -- Build PyTorch Dataset and DataLoader with demographic-stratified sampling

## Progress

**Execution Order:**
Phases execute in numeric order: 1.0 -> 1.1 -> 1.2 -> 2 -> 3 -> 4
Note: Phase 2 and Phase 3 depend only on Phase 1.2 and could run in parallel.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1.0 Prepare Fine-Tuning Dataset | 0/TBD | Not started | - |
| 1.1 LoRA Prototype | 0/2 | Planning complete | - |
| 1.2 Standard LoRA Baseline | 0/TBD | Not started | - |
| 2. GRPO Training Loop | 0/TBD | Not started | - |
| 3. Fairness Baselines | 0/TBD | Not started | - |
| 4. Lambda Sweep and Pareto Frontier | 0/TBD | Not started | - |
