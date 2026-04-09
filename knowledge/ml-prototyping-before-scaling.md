---
fileClass: Knowledge
name: ML Prototyping Before Scaling Training Runs
description: Comprehensive guide on how practitioners prototype ML experiments before committing to expensive full-scale training runs. Covers data subsetting, model scaling, HP transfer, compute budgets, scaling laws, tools, and common pitfalls.
created: 2026-04-06
updated: 2026-04-06
tags:
  - prototyping
  - scaling
  - hyperparameter-tuning
  - compute-efficiency
  - training
  - best-practices
aliases:
  - ML prototyping guide
  - training at scale
  - experiment prototyping
---

# Prototyping Before Scaling ML Training Runs

A practical guide synthesized from Google's Deep Learning Tuning Playbook, Karpathy's "Recipe for Training Neural Networks", scaling laws literature, muP research, and practitioner experience.

---

## 1. The Standard Prototyping Workflow

The consensus workflow among experienced practitioners follows this sequence:

1. **Inspect your data** -- Before writing any model code. Understand distributions, edge cases, label quality, class balance. (Karpathy: "The first step is to not touch any neural net code at all.")
2. **Establish a baseline** -- Use the simplest known-good architecture from a related paper. Don't innovate yet.
3. **Overfit a single batch** -- Verify the model can memorize a tiny subset (1-10 examples). If it can't, you have a bug.
4. **Train on a small data subset** -- 1-10% of full data, for ~10% of full training steps.
5. **Hyperparameter sweep at small scale** -- Find promising HP regions using the small model/data.
6. **Validate on a medium-scale run** -- Confirm the best HPs hold at an intermediate scale.
7. **Final full-scale run** -- Commit to the expensive training with your best configuration.

**Key principle:** Build from simple to complex. Never introduce unverified complexity. Each step should have a concrete hypothesis you're testing.

---

## 2. Data Subsetting Strategies

### Rules of Thumb

| Purpose | Data Fraction | Notes |
|---------|--------------|-------|
| Debugging / sanity checks | 1-2% or a single batch | Verify the pipeline works end-to-end |
| Architecture comparison | 5-10% | Enough to see relative differences between approaches |
| Hyperparameter search | 5-20% | Relative HP rankings usually transfer |
| Medium-scale validation | 20-50% | Confirm small-scale findings hold |
| Final training | 100% | Only after everything else checks out |

### Important Considerations

- **Stratify your subset.** For fairness work especially, ensure demographic subgroups are represented proportionally (or oversampled if they're small).
- **Use a fixed random seed** for reproducible subsets across experiments.
- **Don't manually clean data in ways that won't scale** -- any cleaning must be automated.
- For **ASR/speech data**, subsetting by hours of audio is more meaningful than by number of utterances (utterance length varies enormously).
- For **LoRA fine-tuning** specifically, even 500-2000 examples can be enough for prototyping since LoRA's low parameter count means it converges quickly on small data.

---

## 3. Model Scaling Strategies

### The Practical Approach

1. **Start with the smallest variant** of your model family (e.g., whisper-tiny before whisper-large, wav2vec2-base before wav2vec2-large).
2. **Use LoRA/PEFT** -- Parameter-efficient fine-tuning is itself a form of prototyping. Start with low rank (r=4-8), then scale rank up if needed.
3. **Progressive resizing** (fast.ai approach) -- In vision, train on small images first, then scale up. The analog in speech/NLP is training on shorter sequences first.

### When to Scale Up

- Scale up when your small model shows **consistent improvement trends** (loss still decreasing, metrics still improving).
- Scale up when you've **exhausted easy gains** at the current scale.
- Don't scale up if your small model is overfitting -- adding data or regularization is cheaper than scaling the model.

### muP (Maximal Update Parameterization)

For those training models from scratch (not fine-tuning), muP enables rigorous HP transfer from small to large models:

- **Core idea:** Keep activations in a constant dynamic range across scales so that optimal HPs remain stable.
- **Practical recipe:** Train a proxy model with hidden size ~256 for ~20 tokens per parameter. Sweep HPs on the proxy. Transfer to the large model.
- **What transfers:** Learning rate, initialization scale, optimizer HPs (mostly).
- **What does NOT transfer:** Regularization HPs (weight decay, dropout), anything tied to the data regime rather than the model scale.
- **Demonstrated savings:** Transferring from 13M params outperformed published BERT-large (350M) results; transferring from 40M outperformed GPT-3 (6.7B) at only 7% of total pretraining cost for tuning.

---

## 4. Hyperparameter Search at Small Scale

### What Transfers from Small to Full Scale

| Transfers Well | Transfers Poorly |
|---------------|-----------------|
| Relative ranking of architectures | Exact optimal learning rate (needs adjustment) |
| Relative ranking of optimizers | Weight decay / dropout rates |
| Learning rate schedule shape | Batch size (often must increase) |
| LoRA rank relative to task difficulty | Exact number of training steps |
| Activation functions, normalization choices | Regularization strength |

### Google's Tuning Playbook Strategy

> "You can probably answer many questions while only training for approximately 10% of the production length."

- Run 1-3 rounds of tuning with **increasing per-trial step limits**.
- Round 1: 10% of production steps, broad sweep.
- Round 2: 30-50% of production steps, narrow sweep around best HPs.
- Round 3 (optional): Full-length confirmation run.

### Practical HP Search Configuration

- **Use quasi-random search** (e.g., Sobol sequences) rather than grid search. It covers the space better.
- **Start with Adam at lr=3e-4** (Karpathy's recommendation) as a safe default for prototyping.
- **Don't tune too many HPs at once.** Focus on 2-3 at a time: learning rate first, then batch size, then regularization.

---

## 5. Compute Budget Allocation

### The Data on How Teams Actually Spend Compute

Research on frontier models shows the **ratio of total compute to final-run compute ranges from 1.2x to 4x**, with a **median of 2.2x**. This means:

| Budget Phase | Fraction of Total | What It Covers |
|-------------|-------------------|----------------|
| Exploration / prototyping | ~20-30% | Architecture search, data experiments, debugging |
| Hyperparameter tuning | ~15-25% | HP sweeps at small and medium scale |
| Validation runs | ~5-10% | Medium-scale confirmation |
| Final training run | ~45-55% | The production run |

### Concrete Examples from Published Models

- **GPT-3:** At least 14% of compute was spent outside the main training run.
- **BLOOM:** About 63% of compute was used on smaller models (very thorough prototyping).

### Practical Rule

**Budget 2x your final run cost for the full project.** If your final LoRA fine-tuning run will take 4 GPU-hours, budget 8 GPU-hours total for the project including all experimentation.

---

## 6. Learning Rate and Batch Size Scaling Laws

### When Scaling Batch Size by Factor k

| Optimizer | Learning Rate Rule | Notes |
|-----------|-------------------|-------|
| SGD + Momentum | Scale LR by k (linear) | Works well in practice for moderate k |
| Adam / AdamW | Scale LR by sqrt(k) | More conservative; Adam is already adaptive |

### Important Caveats

- These are **starting points, not laws.** Always verify with a short run.
- **Warmup becomes more important** with larger batch sizes. Use linear warmup for the first 5-10% of training.
- Recent research (2024) shows the relationship is not monotonic for Adam: optimal LR **first rises then falls** as batch size increases ("surge phenomenon").
- **One Cycle Policy** (Leslie Smith / fast.ai): Use a LR range test to find max LR, set min LR to 1/5th-1/10th of max, cycle up then down. Works well as a prototyping tool since it eliminates LR schedule tuning.

### When Scaling Model Size

- Larger models generally need **lower learning rates**.
- Chinchilla-style scaling: when you double model size, also double the training data.
- The **20 tokens per parameter** rule from Chinchilla is a useful starting point (though recent work suggests even more data may be optimal).

---

## 7. Common Pitfalls

### Pitfalls That Cause Small-Scale Results to Not Generalize

1. **Overfitting the validation set through extensive experimentation.** The more HP comparisons you run, the more likely you select a config that got lucky on the val set. Use a held-out test set that you touch only once.

2. **Data subset not representative.** If your 5% subset doesn't capture the full distribution (especially tail cases, hard examples, minority subgroups), findings won't transfer. For ASR fairness work, this is critical -- ensure all demographic groups are in the subset.

3. **Regularization tuned at wrong scale.** Dropout and weight decay that work at small scale will usually be too strong at full scale (more data = less overfitting = less regularization needed).

4. **Ignoring training dynamics.** A model that converges in 1000 steps at small scale may need non-linearly more steps at full scale. Loss curves have different shapes at different scales.

5. **Batch normalization behavior changes.** BatchNorm statistics differ between small and large batches. If using BatchNorm, prototype with the same batch size you'll use at scale, or switch to LayerNorm/GroupNorm.

6. **Confusing "works" with "works well."** A model that achieves 30% WER on a 5% subset tells you the pipeline works, but says little about whether it'll reach 5% WER at full scale.

7. **Not accounting for data leakage in prototyping splits.** When creating fast subsets, ensure no overlap between train/val/test, especially with augmented data.

8. **Hardware-dependent results.** Mixed precision behavior, gradient accumulation artifacts, and multi-GPU synchronization can change results when moving from a single-GPU prototype to multi-GPU training.

---

## 8. Tools and Frameworks

### Hyperparameter Optimization

| Tool | Best For | Key Strength |
|------|----------|-------------|
| **Optuna** | Single-machine sweeps, rapid prototyping | Pruning (early stopping of bad trials), define-by-run API, TPE sampler |
| **Ray Tune** | Multi-GPU / multi-node distributed sweeps | Scales to clusters, integrates with Optuna/HyperOpt as backends |
| **W&B Sweeps** | Team collaboration, experiment tracking | Best visualization, easy to share results, Bayesian and random search |
| **Optuna + Ray Tune** | Best of both worlds | Optuna's algorithms + Ray's parallelization |

### Experiment Tracking

- **Weights & Biases (W&B):** Industry standard. Log metrics, compare runs, share dashboards.
- **MLflow:** Open-source alternative, good for on-prem.
- **TensorBoard:** Built into PyTorch/TF, good for quick local visualization.

### Efficient Training

- **Hugging Face PEFT:** LoRA, QLoRA, and other parameter-efficient methods. Perfect for prototyping fine-tuning with minimal compute.
- **DeepSpeed ZeRO:** Memory-efficient training for scaling up.
- **bitsandbytes:** 4-bit/8-bit quantization for fitting larger models on smaller GPUs during prototyping.

### For This Project Specifically (ASR Fairness)

- Use **PEFT/LoRA** for rapid prototyping of fine-tuning approaches.
- Use **W&B** for tracking fairness metrics (WER by subgroup) across experiments.
- Use **Optuna** for HP search since ASR fine-tuning is typically single-GPU.

---

## 9. Scaling Laws Literature

### Key Papers and Their Practical Implications

**Kaplan et al. (2020) -- "Scaling Laws for Neural Language Models"**
- Performance is a smooth power-law function of model size, dataset size, and compute.
- Larger models are more sample-efficient (need less data per parameter to reach a given loss).
- **Practical implication:** You can predict full-scale performance from small-scale runs by fitting power laws.

**Chinchilla / Hoffmann et al. (2022) -- "Training Compute-Optimal Large Language Models"**
- For a fixed compute budget, model size and training tokens should scale equally.
- Rule of thumb: ~20 tokens per parameter (though more recent work suggests higher ratios).
- **Practical implication:** Most models were undertrained. Smaller models trained on more data often beat larger models trained on less data, and are cheaper at inference too.

**muP / Yang et al. (2022) -- "Tensor Programs V"**
- Hyperparameters can transfer across model scales with the right parameterization.
- **Practical implication:** Tune HPs on a 13M-40M proxy model, transfer to billion-scale models. Saves enormous compute.

**Predictable Scale (2025)**
- Optimal learning rate follows a power-law with both model params and data size.
- Optimal batch size scales primarily with data size.
- Estimated values are within 0.07% of globally optimal found via exhaustive search.
- **Practical implication:** You can analytically predict optimal LR and batch size for large runs without searching.

---

## 10. Practical Heuristics (Cheat Sheet)

### Karpathy's Recipe (Condensed)

1. Inspect data thoroughly.
2. Set up end-to-end training/eval skeleton, verify loss at init.
3. Overfit a single batch.
4. Add data, regularize, tune. In that order.
5. Use Adam 3e-4 as default. Only switch if you have a reason.
6. Only introduce one change at a time.

### Google Tuning Playbook (Condensed)

1. Start with a well-understood model and dataset.
2. Separate scientific HPs (you're investigating) from nuisance HPs (you just want good enough).
3. Use quasi-random search, not grid search.
4. Run short experiments (10% of production length) to answer most questions.
5. Narrow the search space iteratively over 1-3 rounds.
6. Don't decay LR until you've tuned everything else.

### fast.ai Approach (Condensed)

1. Use transfer learning (pretrained models) whenever possible.
2. Use the 1-cycle LR policy to eliminate LR schedule tuning.
3. Progressive resizing: start small, scale up.
4. Use discriminative learning rates (lower LR for pretrained layers, higher for new heads).
5. Freeze pretrained layers first, train head, then unfreeze and fine-tune everything.

### For LoRA Fine-Tuning Specifically

1. Start with rank r=8, alpha=16 (2x rank), dropout=0.05.
2. Target attention layers first (q_proj, v_proj). Add more if needed.
3. Use 500-2000 examples for initial prototyping.
4. LR in range 1e-4 to 5e-4 for most LoRA setups.
5. If not converging, increase rank before increasing LR.
6. Sweep rank and LR together -- they interact strongly.

---

## Summary Decision Framework

```
Is pipeline working end-to-end?
  NO  --> Debug with 1 batch / 1% data
  YES --> Are you getting reasonable baseline metrics?
    NO  --> Check data, loss function, evaluation code
    YES --> Have you done a HP sweep?
      NO  --> Run sweep at 5-10% data, 10% steps (Optuna/W&B)
      YES --> Do results hold at 20-50% data?
        NO  --> Your subset was unrepresentative, fix sampling
        YES --> Run full-scale training with best config
```

---

## Sources

- [Google Deep Learning Tuning Playbook (GitHub)](https://github.com/google-research/tuning_playbook)
- [Google Deep Learning Tuning Playbook (Developers)](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook)
- [Karpathy -- A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)
- [Chinchilla -- Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [muP -- Cerebras Practitioner's Guide](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)
- [muP -- EleutherAI Guide](https://blog.eleuther.ai/mutransfer/)
- [muP -- Microsoft GitHub](https://github.com/microsoft/mup)
- [Predictable Scale -- Optimal HP Scaling Laws (2025)](https://arxiv.org/html/2503.04715v1)
- [Batch Size vs LR Scaling Rules -- Princeton](https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/)
- [Surge Phenomenon in LR/Batch Size Scaling](https://arxiv.org/abs/2405.14578)
- [Rising Costs of Training Frontier Models (compute budget data)](https://arxiv.org/html/2405.21015v1)
- [Databricks -- LoRA Fine-Tuning Guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
- [PLoRA -- Efficient LoRA HP Tuning](https://arxiv.org/html/2508.02932v1)
- [HDMoLE -- LoRA MoE for ASR](https://arxiv.org/html/2409.19878)
- [Optuna](https://optuna.org/)
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)
- [Speechmatics -- Reduce Model Tuning Costs with MuP](https://blog.speechmatics.com/mup)
