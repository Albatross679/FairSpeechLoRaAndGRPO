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

### Phase 4: RL Prototyping Before Scaling Up RL
- **Goal:** Prototype GRPO fairness-aware fine-tuning with a custom training loop (TRL incompatible), validate training signal exists on a 2K-sample subset with G=2 candidates, and run a 3-lambda sweep (0, 0.3, 0.7) to establish the accuracy-fairness tradeoff before full-scale Phase 5
- **Depends on:** Phase 3 (production SFT baseline adapter)
- **Requirements:** GRPO-01, GRPO-02, GRPO-03, GRPO-04, GRPO-05, GRPO-06, GRPO-07, GRPO-08, GRPO-09, EXPT-01 (prototype), EXPT-04 (prototype)
- **Plans:** 2 plans

Plans:
- [ ] 04-01-PLAN.md — Prototype validation (lambda=0.0, 200 steps, signal check)
- [ ] 04-02-PLAN.md — Lambda sweep (0, 0.3, 0.7) + evaluation + Pareto analysis

### Phase 5: Launch the Scaled-Up RL
- **Goal:** Launch the scaled-up RL training run using validated GRPO configuration from Phase 4 prototyping — producing the fairness-aware checkpoint at full scale
- **Depends on:** Phase 4 (RL prototyping validated)
- **Requirements:** TBD

### Phase 6: Rejection Sampling — Fairness-Aware SFT (Optional)
- **Goal:** Sample from best GRPO checkpoint, filter for low-gap + high-accuracy transcriptions, and use as data for a second SFT round — producing a model that is both fair and accurate without RL overhead at inference
- **Depends on:** Phase 5 (best GRPO checkpoint)
- **Requirements:** TBD

### Phase 7: FairLoRA + Group-DRO + ICASSP Comparisons
- **Goal:** Train the three comparison baselines: FairLoRA (supervised fairness regularizer), Group-DRO (distributionally robust optimization), and ICASSP 2026 fairness-prompted fine-tuning. All plotted on the same Pareto frontier.
- **Depends on:** Phase 3 (production SFT baseline adapter)
- **Requirements:** TBD

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

### Phase 4: RL Prototyping Before Scaling Up RL

**Delivers:** Custom GRPO training loop (train_grpo.py), composite reward function (reward.py), VRAM profile on T4, training signal validation (advantage std, zero-std fraction), 3-lambda sweep results (0, 0.3, 0.7) showing accuracy-fairness tradeoff on 2K subset

**Rationale:** GRPO for ASR is novel and risky — TRL doesn't support audio, VRAM is tight on T4, and the unique-answer problem may starve the learning signal. Prototyping at small scale validates that the algorithm works before committing to expensive full-scale training. The 3-lambda sweep establishes whether fairness reward produces a meaningful tradeoff.

**Avoids:** Pitfall #3 (premature scaling — validates signal before Phase 5), Pitfall #1 (unique-answer — monitors frac_zero_std), Pitfall #2 (VRAM — starts with G=2, batch_size=1)

### Phase 5: Launch the Scaled-Up RL

**Delivers:** Full-scale GRPO training run with validated hyperparameters from Phase 4, fairness-aware checkpoint ready for evaluation and Pareto frontier analysis

**Rationale:** Phase 4 prototypes and validates the GRPO reward design, training stability, and hyperparameters at small scale. Phase 5 takes that validated configuration and runs the full-scale training — longer training, full dataset, proper checkpointing — producing the definitive GRPO checkpoint that balances fairness and accuracy.

**Avoids:** Pitfall #3 (premature scaling — only runs full RL training after config is validated in Phase 4)

### Phase 6: Rejection Sampling — Fairness-Aware SFT (Optional)

**Delivers:** Rejection-sampled dataset from best GRPO checkpoint (filtered for low demographic gap + high accuracy), second-round SFT adapter that embeds fairness into the model without RL at inference time

**Rationale:** GRPO produces a checkpoint that balances fairness and accuracy, but RL inference overhead is expensive. Rejection sampling distills the GRPO policy into a standard SFT model: generate many transcriptions, keep only those that are both accurate and demographically fair, then fine-tune on this curated set. The result is a model with fairness baked in, deployable without RL.

**Avoids:** Pitfall #3 (premature scaling — only runs after GRPO checkpoint is validated and Pareto frontier is established)

### Phase 7: FairLoRA + Group-DRO + ICASSP Comparisons

**Delivers:** Three trained comparison baselines (FairLoRA, Group-DRO, ICASSP 2026 fairness-prompted) with fairness metrics, all plotted on the shared Pareto frontier alongside GRPO results

**Rationale:** The paper's contribution requires showing GRPO outperforms existing fairness methods. These three represent the state of the art: FairLoRA adds a supervised fairness regularizer during LoRA training, Group-DRO optimizes for worst-group performance, and ICASSP 2026 uses fairness-prompted fine-tuning. Training all three under identical conditions enables fair comparison on the Pareto frontier.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 2 | 2/2 | Complete | 2026-04-07 |
| 2.1 | 2/2 | Ready | — |
| 3 | 3/3 | Planned | — |
| 4 | 0/2 | Planned | — |
| 5 | 0 | Planned | — |
| 6 | 0 | Planned | — |
| 7 | 0 | Planned | — |
