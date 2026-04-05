---
fileClass: Knowledge
name: LoRA Hyperparameter Sensitivity for ASR Fine-Tuning
description: Research summary on whether a LoRA hyperparameter sweep is worth adding before scaling to full ASR fine-tuning, with priority ordering and cost-benefit analysis
created: 2026-04-05
updated: 2026-04-05
tags:
  - lora
  - hyperparameter-tuning
  - asr
  - fine-tuning
  - whisper
  - qwen3-asr
aliases:
  - lora-sweep-decision
---

# LoRA Hyperparameter Sensitivity for ASR Fine-Tuning

## Context

Deciding whether to add a LoRA hyperparameter sweep phase between Phase 1.1 (feasibility prototype) and full-scale training of Qwen3-ASR-1.7B on FairSpeech + CommonVoice.

## Key Findings

### 1. LoRA IS sensitive to hyperparameters

Recent empirical work (Feb 2026, arxiv 2602.04998) shows that once learning rates are properly tuned, different LoRA variants achieve similar peak performance (within 1-2%). However, the **learning rate itself** is the critical variable -- getting it wrong can cause LoRA to barely outperform or even degrade below the base model.

The "default" settings (r=16, alpha=32, lr=3e-4) work for roughly 30% of use cases (Databricks guide). Task-specific tuning matters.

### 2. ASR-specific evidence

Multiple papers on LoRA for Whisper confirm:
- **Rank**: Ablation studies show optimal performance around r=32 for ASR, with no improvement at higher ranks (Springer, 2026). Values of r=8 to r=32 are the practical range.
- **Target modules**: Attention-only targeting is a reasonable starting point but applying LoRA to all linear layers is needed to match full fine-tuning performance.
- **Learning rate**: ASR-specific work recommends lr=5e-4 for distil-Whisper, lr=1e-4 for standard Whisper. The general rule is 10x the full fine-tuning LR.
- **Initialization**: DWT-based LoRA initialization (DWTLoRA) for speech models improves convergence under low-resource conditions.

### 3. Hyperparameter priority order

1. **Learning rate** (highest impact, can make or break results)
2. **Target modules** (attention-only vs all linear layers)
3. **Rank** (search powers of 2: 4, 8, 16, 32)
4. **Alpha** (set alpha = rank or 2x rank; less sensitive)
5. **Dropout** (least sensitive, 0.05-0.1 is standard)

### 4. Cost estimate for a small sweep on RTX A4000 (16GB)

Qwen3-ASR-1.7B with LoRA (r=16) likely uses ~8-10GB VRAM. A sweep of 8 configs at 500 steps each:
- ~4,000 total training steps
- At ~1-2 seconds/step on A4000: roughly 1-2 hours total
- This is negligible compared to full training (thousands of steps on full datasets)

### 5. Practitioner consensus

The dominant advice: do NOT skip hyperparameter validation entirely. But a full grid search is overkill. The recommended middle ground is a **targeted mini-sweep** on the 2-3 most impactful parameters.

## Recommendation

**Do a lightweight sweep (middle ground)**, structured as follows:

1. **Fix** alpha=2*rank, dropout=0.05 (low sensitivity)
2. **Sweep learning rate first** (3 values: 1e-4, 3e-4, 5e-4) with your current config (r=16, attention-only) -- this is the single highest-impact parameter
3. **Then sweep rank** (8, 16, 32) at the best LR
4. **Optionally test** target_modules: attention-only vs all linear layers (2 configs)

Total: ~8 configs, 500 steps each, ~1-2 hours on A4000. This is cheap insurance against picking a config that underperforms or degrades.

## Sources

- [Learning Rate Matters: Vanilla LoRA May Suffice (arxiv 2602.04998)](https://arxiv.org/html/2602.04998v1)
- [A Unified Study of LoRA Variants (arxiv 2601.22708)](https://arxiv.org/pdf/2601.22708)
- [Efficient LoRA for Amazigh ASR (Springer, 2026)](https://link.springer.com/article/10.1007/s11042-026-21336-0)
- [LoRA-Whisper: Multilingual ASR](https://arxiv.org/html/2406.06619v1)
- [Sparsely Shared LoRA on Whisper for Child Speech](https://ar5iv.labs.arxiv.org/html/2309.11756)
- [Databricks: LoRA Guide for LLMs](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
- [Unsloth LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Trelis Research: How to Choose LoRA Hyperparameters](https://trelis.substack.com/p/how-to-choose-lora-hyper-parameters)
- [Beware of the Batch Size (arxiv 2602.09492)](https://arxiv.org/html/2602.09492)
- [LoRA Without Regret (Thinking Machines Lab)](https://thinkingmachines.ai/blog/lora/)
