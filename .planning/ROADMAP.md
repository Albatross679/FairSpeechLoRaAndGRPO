# Roadmap: GRPO Fairness-Aware ASR Fine-Tuning

## Overview

This roadmap delivers a GRPO fairness-aware fine-tuning pipeline for Qwen3-ASR-1.7B. Phase 1 validates the critical risk (LoRA on Qwen3-ASR decoder) and proves the full adapter lifecycle with a standard LoRA baseline. Phase 2 builds the core GRPO training loop with composite fairness reward. Phase 3 implements the three fairness baselines for comparison. Phase 4 runs the lambda sweep, produces the Pareto frontier, and completes strengthening experiments. Every phase builds on proven infrastructure from the phase before it.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: LoRA Foundation and Standard Baseline** - Attach LoRA to Qwen3-ASR decoder, validate VRAM budget, train standard LoRA baseline, build evaluation bridge
- [ ] **Phase 2: GRPO Training Loop** - Implement custom GRPO loop with composite accuracy-fairness reward and demographic-stratified batching
- [ ] **Phase 3: Fairness Baselines** - Implement FairLoRA, Group-DRO, and ICASSP fairness-prompted baselines using shared adapter infrastructure
- [ ] **Phase 4: Lambda Sweep and Pareto Frontier** - Run lambda sweep, produce Pareto frontier visualization, complete statistical testing and ablations

## Phase Details

### Phase 1: LoRA Foundation and Standard Baseline
**Goal**: Users can train a standard LoRA adapter on Qwen3-ASR-1.7B and evaluate it with fairness metrics on both datasets
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, BASE-01, EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. LoRA adapter attaches to Qwen3-ASR-1.7B decoder and forward pass produces valid transcriptions (not garbage text)
  2. Standard LoRA training (lambda=0) runs to completion without OOM on RTX A4000 with peak VRAM under 14GB
  3. Trained adapter can be saved, loaded on a fresh process, and produce identical transcriptions to the in-memory version
  4. Evaluation produces WER and fairness metrics (max-min ratio, gap%, std, bootstrap CIs) for both ethnicity and accent axes on Fair-Speech and Common Voice 24
**Plans**: TBD

### Phase 2: GRPO Training Loop
**Goal**: Users can train Qwen3-ASR with GRPO using a composite fairness reward and observe per-group WER dynamics during training
**Depends on**: Phase 1
**Requirements**: GRPO-01, GRPO-02, GRPO-03, GRPO-04, GRPO-05, GRPO-06, GRPO-07, GRPO-08, GRPO-09, GRPO-10
**Success Criteria** (what must be TRUE):
  1. GRPO training loop generates G candidate outputs per input, computes composite reward R = (1-lambda)(1-WER) + lambda(-|WER_g - WER_mean|), and updates only LoRA parameters
  2. Training with lambda=0.5 runs to completion and produces an adapter whose per-group WER spread is measurably different from the standard LoRA baseline
  3. Per-group WER is logged every N steps during training and shows interpretable dynamics (not flat or divergent)
  4. Dr. GRPO variant (no std normalization, length normalization) is selectable via config and produces a trained adapter
**Plans**: TBD

### Phase 3: Fairness Baselines
**Goal**: All three fairness baselines are trained and evaluated, producing results directly comparable to the standard LoRA and GRPO methods
**Depends on**: Phase 1
**Requirements**: BASE-02, BASE-03, BASE-04, EVAL-05, EVAL-06
**Success Criteria** (what must be TRUE):
  1. FairLoRA adapter is trained with group-variance regularization and evaluated with full fairness metrics on both datasets
  2. Group-DRO adapter is trained with smoothed exponential group weight updates and evaluated with full fairness metrics on both datasets
  3. ICASSP fairness-prompted adapter is trained with the four-term fusion loss and evaluated with full fairness metrics on both datasets
  4. Perturbation robustness evaluation runs under existing 12 conditions for all trained adapters (standard LoRA, GRPO, and all baselines)
  5. Error decomposition (sub/ins/del rates) by demographic group is computed for all trained models
**Plans**: TBD

### Phase 4: Lambda Sweep and Pareto Frontier
**Goal**: The central experimental result is produced: a Pareto frontier plotting accuracy vs. fairness across all methods and lambda values, with statistical backing
**Depends on**: Phase 2, Phase 3
**Requirements**: EXPT-01, EXPT-02, EXPT-03, EXPT-04
**Success Criteria** (what must be TRUE):
  1. Lambda sweep across 6+ values produces trained GRPO adapters that trace a visible accuracy-fairness tradeoff curve
  2. Pareto frontier scatter plot shows all 5 methods (Standard LoRA, GRPO variants, FairLoRA, Group-DRO, ICASSP) with accuracy on one axis and fairness metric on the other
  3. Bootstrap pairwise significance tests determine which method differences are statistically significant
  4. Hyperparameter sensitivity ablation over G, learning rate, and beta (KL coefficient) is documented with results
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4
Note: Phase 2 and Phase 3 depend only on Phase 1 and could run in parallel.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. LoRA Foundation and Standard Baseline | 0/TBD | Not started | - |
| 2. GRPO Training Loop | 0/TBD | Not started | - |
| 3. Fairness Baselines | 0/TBD | Not started | - |
| 4. Lambda Sweep and Pareto Frontier | 0/TBD | Not started | - |
