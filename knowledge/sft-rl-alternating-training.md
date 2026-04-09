---
fileClass: Knowledge
name: SFT+RL Alternating Training Pipeline
description: How supervised fine-tuning and reinforcement learning coordinate in modern LLM/ASR training — the alternating multi-stage pattern
created: 2026-04-06
updated: 2026-04-06
tags: [grpo, sft, rl, training-pipeline, deepseek-r1, qwen3, rejection-sampling]
aliases: [iterative SFT RL, alternating training]
---

# SFT + RL Alternating Training: The Modern Standard

## Core Insight: "SFT Expands, RL Squeezes"

A 2025 paper (arXiv:2509.21128) provides the clearest explanation:
- **SFT expands** the space of correct reasoning trajectories (teaches new solution strategies by imitation)
- **RL compresses** incorrect trajectories (eliminates bad behaviors through negative signal)
- They are fundamentally complementary — neither alone is sufficient

## The Standard Multi-Stage Pipeline

The field has converged on **4+ stage pipelines that alternate SFT and RL**:

### DeepSeek-R1 (Jan 2025)

| Stage | Type | Purpose |
|-------|------|---------|
| 1. Cold-Start SFT | SFT | Establish readable reasoning format, prevent language mixing |
| 2. Reasoning RL | GRPO | Large-scale RL on reasoning tasks with rule-based rewards |
| 3. Rejection Sampling + SFT | SFT | Sample from Stage 2, keep correct outputs (~600k), retrain from base |
| 4. General RL | RL | Helpfulness + harmlessness across all scenarios |

### Qwen3 (2025)

| Stage | Type | Purpose |
|-------|------|---------|
| 1. CoT Cold Start | SFT | Establish "thinking" ability |
| 2. Reasoning RL | GRPO | RL on reasoning tasks |
| 3. Thinking Mode Fusion | SFT | Merge thinking/non-thinking via rejection sampling from Stage 2 |
| 4. General RL | RL | Instruction-following, format compliance, agent abilities |

### Llama 3.1 / 4

- Multiple rounds of SFT + DPO (preferred over PPO for stability)
- Llama 4 finding: **SFT can over-constrain the model**, restricting RL exploration
- Solution: prune 50-95% of SFT data, do lightweight SFT, heavy RL, lightweight DPO

## Why Alternate? What Each Stage Solves

### What SFT does that RL cannot:
- Rapid initial learning (shows correct outputs directly)
- Establishes format, readability, structure
- Provides warm start for RL (avoids wasting compute)

### What RL does that SFT cannot:
- Discovers novel solutions not in training data
- Generalizes better (SFT tends to memorize)
- Better asymptotic performance
- Minimizes catastrophic forgetting vs SFT (MIT study, 2025)

### Without SFT (DeepSeek-R1-Zero):
- Model learned to reason via pure GRPO
- But: endless repetition, poor readability, language mixing, no formatting

### Without RL (SFT-only):
- Performance plateaus below SFT+RL
- Fails to suppress incorrect behaviors
- Cannot optimize for non-differentiable objectives (like fairness gaps)

## Rejection Sampling: The Bridge

Rejection sampling connects RL rounds back to SFT rounds:
1. Train model with RL
2. Generate multiple candidate responses per prompt (16-64 samples)
3. Filter: keep only correct/high-quality responses
4. Use filtered responses as SFT training data for next round

This creates a virtuous cycle: RL discovers better solutions → rejection sampling captures them → next SFT round has higher-quality data.

## GRPO vs PPO for This Pattern

| Aspect | PPO | GRPO |
|--------|-----|------|
| Critic model | Required (~2x memory) | Not needed (~1x memory) |
| Advantage estimation | Per-token, learned value function | Per-response, normalized within group |
| Typical use | RLHF (reward model) | RLVR (verifiable rewards) |
| Memory | Higher | ~50% less |

GRPO's lower memory makes the alternating pattern more practical, especially with LoRA on limited VRAM.

## LoRA Considerations (TRL "LoRA Without Regret")

- **SFT stage**: High rank (r=256) — needs capacity for diverse data
- **RL stage**: Low rank (r=1 to 32) — policy gradient extracts ~1 bit/episode
- **Target modules**: Always `target_modules="all-linear"` — all layers, not just attention
- **Between stages**: Merge SFT LoRA into base, then apply fresh LoRA for RL
- **Never stack** LoRA adapters (causes instability)

## ASR-Specific RL (arXiv:2509.18569)

GRPO applied to ASR with three rewards:
- R1: `1 - WER(prediction, reference)` — accuracy
- R2: `-1` on hallucinations — penalize confabulation
- R3: Keyword recall/precision — domain accuracy

## For Our Project: Proposed Pipeline

1. **SFT with LoRA** (r=256): Fine-tune Qwen3-ASR on Fair-Speech + CV transcription data
2. **GRPO with LoRA** (r=16): Fairness reward `R = (1-λ)(1-WER) + λ(-|WER_g - WER|)`
3. **Optional**: Rejection sampling from GRPO checkpoint → SFT round 2 (if time permits)
4. λ sweep across GRPO checkpoints for Pareto frontier

## Key Sources

- DeepSeek-R1: arXiv:2501.12948
- Qwen3 Technical Report: arXiv:2505.09388
- "RL Squeezes, SFT Expands": arXiv:2509.21128
- TRL LoRA guide: huggingface.co/docs/trl/main/en/lora_without_regret
- GRPO for ASR: arXiv:2509.18569
- Fair ASR: arXiv:2510.18374
- BRIDGE (cooperative SFT+RL): arXiv:2509.06948
