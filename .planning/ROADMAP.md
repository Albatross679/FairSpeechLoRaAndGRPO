# Roadmap: GRPO Fairness-Aware ASR Fine-Tuning

## Overview

This roadmap delivers a GRPO fairness-aware fine-tuning pipeline for Qwen3-ASR-1.7B. Phase 1 validates the critical risk (LoRA on Qwen3-ASR decoder) and proves the full adapter lifecycle with a standard LoRA baseline. Phase 2 builds the core GRPO training loop with composite fairness reward. Phase 3 implements the three fairness baselines for comparison. Phase 4 runs the lambda sweep, produces the Pareto frontier, and completes strengthening experiments. Every phase builds on proven infrastructure from the phase before it.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

### Phase 2: Standard LoRA Baseline + Evaluation Bridge
- **Goal:** Train a standard LoRA adapter (lambda=0, no fairness reward) and build the evaluation bridge that produces fairness metrics — establishing the accuracy-only baseline all fairness methods compare against
- **Depends on:** Phase 1.0, Phase 1.1 (completed)
- **Requirements:** BASE-01, EVAL-01, EVAL-02, EVAL-03, EVAL-04
- **Canonical refs:** `.planning/research/SUMMARY.md`, `.planning/research/ARCHITECTURE.md`, `.planning/research/PITFALLS.md`

### Phase 2.1: Advanced HP Sweep — PLoRA + ASHA + RsLoRA [INSERTED]
- **Goal:** Upgrade HP sweep with three complementary techniques: PLoRA (parallel adapters), ASHA pruning (early stopping of bad trials), and RsLoRA (rank-stable scaling) — enabling 10×+ faster sweeps and fair exploration of higher ranks
- **Depends on:** Phase 2 (completed — sequential sweep infrastructure, evaluation bridge)
- **Requirements:** PLORA-01–05, ASHA-01–03, RSLORA-01–03

### Phase 3: Full SFT LoRA Training
- **Goal:** Run full-scale SFT LoRA training using the validated hyperparameters and configuration from Phase 2 prototyping — producing the production-quality baseline adapter
- **Depends on:** Phase 2 (validated config, evaluation bridge), Phase 2.1 (optional — improved sweep if available)
- **Requirements:** BASE-01, EVAL-01

## Phase Details

### Phase 2: Standard LoRA Baseline + Evaluation Bridge

**Delivers:** Trained standard LoRA adapter, evaluation bridge producing fairness metrics on Fair-Speech and Common Voice 24, baseline WER numbers by demographic group

**Rationale:** Standard SFT with LoRA is the simplest training method and validates the entire adapter-to-evaluation pipeline end-to-end. It also produces the accuracy-only baseline that all fairness methods are compared against.

**Avoids:** Pitfall #9 (evaluation methodology — validate speaker-disjoint splits here)

### Phase 2.1: Advanced HP Sweep — PLoRA + ASHA + RsLoRA [INSERTED]

**Delivers:** PLoRA training script (3-5 adapters per forward pass), ASHA-pruned Optuna sweep (kills bottom 2/3 at each rung), RsLoRA-enabled rank exploration ({4, 8, 16, 32} with stable gradients), validation against Phase 2 sequential results, speedup benchmarks

**Rationale:** Phase 2's sequential Optuna sweep (20 trials, ~1h 32min) has three limitations: (1) sequential execution wastes compute on the shared frozen model — PLoRA fixes this; (2) all 20 trials run to completion even when clearly losing — ASHA fixes this; (3) standard α/r scaling suppresses gradients at higher ranks, biasing the sweep toward rank-4 — RsLoRA fixes this. Combined, these enable 60-120+ configs in the same wall-clock time.

**Avoids:** Pitfall #3 (premature scaling — validates all three techniques against known Phase 2 results before using for production sweeps)

### Phase 3: Full SFT LoRA Training

**Delivers:** Production-quality SFT LoRA adapter trained with full dataset and validated hyperparameters from Phase 2, complete evaluation metrics across demographic groups

**Rationale:** Phase 2 prototypes and sweeps to find optimal config (learning rate, rank, epochs, etc.). Phase 3 takes that validated config and runs the full training — longer epochs, full dataset, proper checkpointing. This produces the definitive accuracy-only baseline adapter.

**Avoids:** Pitfall #3 (premature scaling — only runs full training after config is validated)

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 2 | 2/2 | Complete | 2026-04-07 |
| 2.1 | 2/2 | Ready | — |
| 3 | 2 | Planned | — |
