# GRPO Fairness-Aware ASR Fine-Tuning

## What This Is

A fairness-aware fine-tuning pipeline for Qwen3-ASR-1.7B using Group Relative Policy Optimization (GRPO) with a composite accuracy-fairness reward. The system trains LoRA adapters on the frozen encoder, sweeps the fairness weight λ to trace a Pareto frontier of accuracy vs. demographic fairness, and compares against four baselines (standard LoRA, FairLoRA, Group-DRO, ICASSP 2026 fairness-prompted fine-tuning). This is the primary intervention planned in the CSE 5525 final project, building on midterm benchmarking that found Qwen3-ASR-1.7B has the best raw accuracy but the worst relative ethnicity gap (+203%).

## Core Value

Produce a working GRPO training pipeline that traces the accuracy-fairness Pareto frontier for Qwen3-ASR-1.7B, demonstrating whether fairness-aware RL fine-tuning can close demographic gaps without sacrificing accuracy.

## Requirements

### Validated

- ✓ ASR inference pipeline for 9 models across clean and perturbed conditions — existing
- ✓ Fairness metrics computation (max-min ratio, gap %, std, bootstrap CIs) — existing
- ✓ Common Voice 24 and Fair-Speech dataset preparation — existing
- ✓ Perturbation generation (noise, reverb, silence, masking) — existing
- ✓ Visualization pipeline for publication-ready figures — existing

### Active

- [ ] GRPO training loop with composite fairness reward R = (1-λ)(1-WER) + λ(-|WER_g - WER|)
- [ ] LoRA adapter integration (freeze encoder, train ~1-2% decoder parameters)
- [ ] λ sweep across multiple values to trace Pareto frontier
- [ ] Fairness reward computed over both ethnicity and accent demographic axes
- [ ] Standard LoRA baseline (λ=0, no fairness reward)
- [ ] FairLoRA baseline (supervised fairness regularizer adapted from vision)
- [ ] Group-DRO baseline (distributionally robust optimization over demographic groups)
- [ ] ICASSP 2026 fairness-prompted fine-tuning baseline
- [ ] Evaluation on both Fair-Speech and Common Voice 24
- [ ] Pareto frontier plot (accuracy vs. fairness for each method)

### Out of Scope

- Attention head surgery / hallucination mitigation — separate future work direction
- Intersectional analysis (accent × gender, ethnicity × age) — deferred to additional directions
- Demographic conditioning / group-specific LoRA adapters — deferred
- Multi-speaker diarization fairness — blocked by dataset labels
- Audio-visual ASR fairness — blocked by dataset availability
- Full paper writing — this milestone focuses on training pipeline and Pareto plot

## Context

- **Course:** CSE 5525 Speech and Language Processing
- **Team:** Srishti Ginjala, Qifan Wen
- **Midterm finding:** No single model is Pareto-optimal — Qwen3-1.7B dominates accuracy/accent fairness/robustness but has worst ethnicity fairness (+203% gap). This motivates GRPO fairness fine-tuning.
- **Existing codebase:** 28 Python scripts covering dataset prep → inference → metrics → visualization. Modular pipeline architecture with CSV manifests as data contracts.
- **GRPO advantage:** Eliminates the value model required by PPO, reducing compute. Validated for ASR with up to 18.4% relative WER improvement in prior work.
- **Hardware:** Single NVIDIA RTX A4000 (16GB VRAM). LoRA keeps fine-tuning feasible on this GPU.
- **Training data:** Fair-Speech (~26,470 utterances, controlled prompts, clean demographic labels) and Common Voice 24 (~16,400 utterances, crowd-sourced, noisier labels). Both have demographic annotations needed for the fairness reward.
- **Model:** Qwen3-ASR-1.7B — best raw accuracy from midterm benchmarks, worst ethnicity gap, making it the ideal target for fairness intervention.

## Constraints

- **Hardware**: Single RTX A4000 (16GB VRAM) — LoRA required, full fine-tuning infeasible
- **Tech stack**: Must integrate with existing Python/PyTorch/HuggingFace pipeline
- **Timeline**: Class project final deliverable
- **Baselines**: 4 comparison methods needed to contextualize GRPO results

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Target Qwen3-ASR-1.7B | Best accuracy but worst ethnicity gap — most room for fairness improvement | — Pending |
| GRPO over PPO | No value model needed, validated for ASR, lower compute | — Pending |
| LoRA adapters (freeze encoder) | 16GB VRAM constraint, train ~1-2% parameters | — Pending |
| Both ethnicity and accent axes | Ethnicity is primary gap, accent adds completeness | — Pending |
| Train on Fair-Speech + Common Voice | Fair-Speech for controlled evaluation, CV for diversity | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-05 after initialization*
