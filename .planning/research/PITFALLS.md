# Domain Pitfalls

**Domain:** GRPO fairness-aware ASR fine-tuning (Qwen3-ASR-1.7B, LoRA, RTX A4000 16GB)
**Researched:** 2026-04-05

---

## Critical Pitfalls

Mistakes that cause training failure, wasted GPU hours, or invalidated results.

### Pitfall 1: GRPO VRAM Memory Leak Leading to OOM

**What goes wrong:** GRPO training VRAM usage increases monotonically with each training step, eventually crashing with OOM. This is a widely-reported bug across TRL, Unsloth, and veRL frameworks in 2025. On a 16GB RTX A4000 with a 1.7B model, there is essentially zero headroom -- even a slow leak (100MB over 500 steps) will crash training.

**Why it happens:** GRPO internally runs multiple forward passes per step: the policy model generates `num_generations` completions per prompt, then the reference model computes log-probs for KL penalty, then reward is computed. KV cache, intermediate tensors, and computation graphs accumulate across steps if not explicitly freed. Some frameworks retain references to old rollout tensors across steps.

**Consequences:** Training crashes after N steps with no checkpoint saved. Hours of GPU time wasted. Subtle: if training completes on small data but fails on full dataset, you only discover the leak at scale.

**Prevention:**
1. Monitor VRAM every 50 steps with `torch.cuda.max_memory_allocated()` -- plot it. If non-flat, there is a leak.
2. Force garbage collection between steps: `torch.cuda.empty_cache(); gc.collect()` after each training step.
3. Use `batch_size=1` with `gradient_accumulation_steps=4` (not batch_size=4) to keep peak VRAM low.
4. Set `max_completion_length` conservatively (e.g., 256 tokens for ASR transcriptions which are short).
5. Use `num_generations=4` (not 8 or 16) -- each generation multiplies VRAM.
6. Pin TRL version and test on 100 steps before committing to full runs.

**Detection:** VRAM monitoring script that logs every N steps. If `max_memory_allocated` increases by >500MB over 200 steps, abort and investigate.

**Phase:** Infrastructure setup (Phase 1). Must be validated before any real training runs.

**Confidence:** HIGH -- multiple GitHub issues document this exact problem (TRL #2927, #3678; Unsloth #2470, #3512, #3864).

---

### Pitfall 2: Fairness Reward Hacking via Accuracy Collapse

**What goes wrong:** The composite reward `R = (1-lambda)(1-WER) + lambda(-|WER_g - WER|)` can be maximized by making ALL groups equally bad. A model that outputs empty transcriptions for every group achieves perfect fairness (all WER gaps = 0) while having catastrophic accuracy (WER = 1.0). At moderate lambda values, the optimizer can find local optima where it degrades majority-group accuracy to match minority-group accuracy rather than lifting minority groups.

**Why it happens:** The fairness term rewards equality of WER, not low WER. Without sufficient weight on the accuracy term, the path of least resistance is leveling down. GRPO's group-relative advantage computation may amplify this: if a batch happens to contain mostly one demographic, the advantage signal is computed relative to that group's baseline, creating noisy gradient directions.

**Consequences:** Model appears to "improve" fairness during training (reward goes up) while WER degrades across the board. Only visible if you track per-group absolute WER alongside the composite reward.

**Prevention:**
1. Log per-group WER at every evaluation step, not just the composite reward. Plot absolute WER alongside fairness gap.
2. Add a WER floor constraint: if WER > baseline_WER + 0.05, override the fairness bonus to zero. This prevents rewarding fairness improvements that come from degradation.
3. Start lambda sweep from 0.0 and increase gradually (0.0, 0.1, 0.2, ..., 0.9). If WER degrades monotonically with lambda, the reward is encouraging collapse.
4. Consider reformulating as `R = (1-WER) - lambda * max(0, |WER_g - WER| - tau)` with a tolerance threshold tau, so small fairness gaps are not penalized.
5. Validate on held-out data after each checkpoint -- if held-out WER rises by >5% relative to baseline, that lambda point is degenerate.

**Detection:** Divergence between composite reward (increasing) and absolute WER (also increasing or flat). Per-group WER converging toward the worst group's level rather than all groups improving.

**Phase:** Reward design (Phase 2). Must be resolved before lambda sweep.

**Confidence:** HIGH -- this is a well-documented failure mode in multi-objective RL and fairness-constrained optimization literature.

---

### Pitfall 3: GRPO Produces Minimal ASR Improvement Due to Unique-Answer Problem

**What goes wrong:** GRPO was designed for tasks with multiple valid outputs (reasoning, translation) where group-relative comparison is meaningful. ASR transcription has exactly ONE correct answer per utterance. When all `num_generations` outputs for a prompt are either correct or incorrect, the advantage computation yields near-zero gradients -- there is nothing to differentiate within the group.

**Why it happens:** GRPO computes advantages by comparing rewards within a group of completions for the same prompt. If all completions get similar WER (e.g., all get the same transcription right or wrong), the normalized advantage is ~0 for all, producing no learning signal. This was explicitly noted in the GRPO-for-ASR paper (arxiv:2509.01939): "preliminary experiments with ASR obtained only minor improvements" compared to speech translation and spoken QA.

**Consequences:** Training appears stable but produces negligible WER improvement. Lambda sweep shows a flat Pareto frontier because GRPO is not meaningfully optimizing the accuracy component.

**Prevention:**
1. Use a continuous WER reward (character-level or word-level edit distance), NOT binary correct/incorrect. This creates reward variance even when outputs are similar but not identical.
2. Increase sampling temperature during generation to 0.7-1.0 to create diversity in completions. If temperature is too low, all completions converge to the same output.
3. Use `num_generations >= 4` to ensure statistical variance in the group.
4. Include diverse difficulty levels in each batch -- mix easy and hard utterances so at least some prompts have reward variance.
5. Monitor the standard deviation of rewards within each group. If avg std < 0.01 for >50% of batches, GRPO is not getting useful signal.
6. Consider supplementary rewards beyond WER: penalize hallucinations (output much longer than expected), reward partial correctness.

**Detection:** Flat or near-zero advantage values across training. Reward std within groups near zero. WER improvement <1% relative after 500+ steps.

**Phase:** GRPO implementation (Phase 2). Reward function design must account for this.

**Confidence:** HIGH -- explicitly documented in published ASR-GRPO research.

---

### Pitfall 4: 16GB VRAM Budget Exhausted by GRPO's Multi-Model Architecture

**What goes wrong:** GRPO needs the policy model, reference model, and reward computation simultaneously in VRAM. Even a 1.7B model in fp16 takes ~3.4GB. With LoRA adapters, optimizer states (Adam has 2x model-size states), KV cache for `num_generations` completions, and activation memory for backprop, 16GB fills completely.

**Why it happens:** Naive GRPO implementation loads: policy model (~3.4GB) + reference model (~3.4GB) + LoRA optimizer states (~200MB) + KV cache (~1-2GB for 4 generations) + activations (~2-4GB with gradient checkpointing) + framework overhead (~1-2GB). Total: ~12-16GB before any batch processing. A single long utterance can push over the edge.

**Consequences:** Either OOM immediately, or OOM on longer utterances (variable-length audio creates variable VRAM usage).

**Prevention:**
1. **Merge reference model**: Use `ref_model=None` in TRL's GRPOTrainer -- it will use the initial policy weights as reference, avoiding a second model copy. The KL penalty is computed by storing the initial log-probs, not a separate model.
2. **8-bit quantization**: Load the reference model (if separate) in 8-bit via bitsandbytes. Policy model stays in fp16 for gradient accuracy.
3. **Gradient checkpointing**: Mandatory. Trades ~25% speed for ~40% VRAM savings on activations.
4. **AdamW 8-bit optimizer** (bitsandbytes): 25-30% optimizer memory savings with negligible quality impact.
5. **Cap sequence lengths**: ASR outputs are typically short. Set `max_completion_length=128` (sufficient for most utterances). Set `max_prompt_length=30s * 16kHz = 480000` samples, but process audio in chunks if needed.
6. **Profile before training**: Run one forward+backward pass with `torch.cuda.memory_summary()` and verify peak < 14GB (leave 2GB headroom for spikes).

**Detection:** Run `nvidia-smi` monitoring in background. Profile first batch. If peak > 14GB, reduce num_generations or sequence length before proceeding.

**Phase:** Infrastructure setup (Phase 1). Non-negotiable prerequisite.

**Confidence:** HIGH -- arithmetic on model sizes + verified VRAM reports from similar setups.

---

## Moderate Pitfalls

### Pitfall 5: FairLoRA Regularizer Does Not Transfer to ASR Domain

**What goes wrong:** FairLoRA was designed for vision classification (ViT, DiNO, CLIP) where per-group loss variance maps directly to accuracy disparity. In ASR with CTC/attention-based decoding, per-group loss variance is confounded by utterance length, linguistic complexity, and acoustic conditions -- not just demographic bias. Applying FairLoRA's loss-variance regularizer to ASR may penalize legitimate difficulty differences rather than demographic bias.

**Why it happens:** FairLoRA's regularizer `lambda * sum_g (L_g - L_mean)^2` assumes loss differences between groups reflect bias. In ASR, a group with longer utterances or more complex vocabulary will have higher loss even with zero bias. CTC loss specifically scales with input length, as documented in CTC-DRO (Bartelds et al., 2025).

**Prevention:**
1. Normalize per-group loss by utterance length before computing FairLoRA's variance penalty.
2. Use WER-based fairness penalty (as in the GRPO reward) rather than raw loss variance, since WER is length-normalized.
3. If implementing FairLoRA as a baseline, carefully document whether you use loss-based or WER-based variance, and explain the difference.
4. Consider CTC-DRO's approach: input length-matched batching + smoothed group weight updates, which was specifically designed for ASR fairness.

**Detection:** FairLoRA baseline shows fairness improvements on training loss but NOT on evaluation WER gaps. This signals the regularizer is optimizing a proxy, not the actual fairness objective.

**Phase:** Baseline implementation (Phase 3). Must validate the fairness metric FairLoRA optimizes vs. what we evaluate.

**Confidence:** MEDIUM -- FairLoRA has not been applied to ASR in published work. CTC-DRO documents exactly this loss-vs-metric mismatch.

---

### Pitfall 6: Group-DRO Collapses to Worst-Group Overfitting Without Strong Regularization

**What goes wrong:** Group-DRO minimizes worst-group loss, but in overparameterized neural networks (which 1.7B models are), the model can perfectly fit training data for all groups, making worst-group training loss vanish. At test time, the worst group may still perform poorly because DRO did not generalize. Alternatively, DRO overweights a small noisy group, causing the entire model to overfit to that group's idiosyncrasies.

**Why it happens:** Sagawa et al. (2020) showed that naive Group-DRO on overparameterized networks requires significantly stronger L2 regularization than standard training (10-40 percentage point improvement when regularization is added). Without it, worst-group test accuracy can be worse than standard ERM. Additionally, group weight updates can become unstable when one group has very few samples.

**Prevention:**
1. Apply stronger L2 regularization than you would for standard LoRA training (2-5x the default weight_decay).
2. Use early stopping based on worst-group validation WER, not average WER.
3. Smooth group weight updates (exponential moving average with momentum 0.1) to prevent oscillation on small groups.
4. Enforce minimum group weight: no group's weight should drop below 1/|G| * 0.1 to prevent ignoring any group.
5. Length-normalize CTC loss before group aggregation (following CTC-DRO).

**Detection:** Worst-group training loss decreasing but worst-group validation WER increasing or flat. Group weights oscillating wildly between steps.

**Phase:** Baseline implementation (Phase 3).

**Confidence:** HIGH -- Sagawa et al. ICLR 2020 + CTC-DRO 2025 both document this.

---

### Pitfall 7: Demographic Group Imbalance Corrupts Fairness Reward Signal

**What goes wrong:** Fair-Speech has ~26,470 utterances and Common Voice has ~16,400, but demographic groups within each dataset are highly imbalanced. If "African American" has 500 utterances and "White" has 8,000, the per-batch fairness reward will be computed over vastly different sample sizes. Small-group WER estimates are noisy, creating noisy fairness rewards that destabilize GRPO training.

**Why it happens:** Mini-batch sampling for GRPO is typically uniform over utterances, not stratified by demographic group. A batch of 32 may contain 0-2 utterances from small groups, making per-group WER in that batch meaningless. The fairness term in the reward becomes dominated by sampling noise rather than actual demographic gaps.

**Prevention:**
1. **Stratified batch sampling**: Ensure each training batch contains minimum N samples per demographic group (e.g., N >= 8). If a group has fewer total samples than needed, oversample it.
2. **Accumulate fairness signal**: Compute the fairness reward term over a rolling window of K batches (e.g., K=10) rather than per-batch, to smooth out sampling noise.
3. **Document group sizes**: Before training, print the demographic distribution of both datasets. If any group has < 100 utterances, flag it as potentially unreliable.
4. **Use MIN_GROUP_SIZE consistently**: The existing codebase uses MIN_GROUP_SIZE=50. Apply the same threshold during training data filtering.
5. **Weight fairness term by group confidence**: Scale each group's contribution to the fairness reward by sqrt(n_g) to downweight estimates from small groups.

**Detection:** High variance in fairness reward between consecutive batches (>2 std from mean). Groups with <50 samples in training set. Per-batch group counts frequently hitting 0 for any demographic.

**Phase:** Data preparation + reward design (Phase 1-2).

**Confidence:** HIGH -- basic statistics. The existing codebase already has MIN_GROUP_SIZE for evaluation; the same logic must apply to training.

---

### Pitfall 8: Entropy Collapse Kills GRPO Exploration

**What goes wrong:** GRPO's clipping mechanism constrains low-probability tokens more than high-probability ones. Over training, the model becomes increasingly deterministic -- entropy collapses. Once entropy is low, all `num_generations` completions are identical, advantages are zero, and learning stops entirely.

**Why it happens:** With symmetric clipping at epsilon=0.2, the probability ratio is clamped to [0.8, 1.2]. A token with p=0.01 can only increase to p=0.012 (20% increase), but a token with p=0.5 can increase to p=0.6 (20% increase = much larger absolute change). This systematically suppresses exploration of rare tokens, which is particularly harmful in ASR where rare words (names, technical terms) are disproportionately important for fairness.

**Prevention:**
1. Use asymmetric (DAPO-style) clipping: `epsilon_low=0.2, epsilon_high=0.28` to allow more exploration.
2. Monitor entropy at every evaluation step. If entropy drops below 50% of initial value, training is collapsing.
3. Add a small entropy bonus to the reward: `R += 0.01 * H(policy)`.
4. Use token-level loss aggregation (not sample-level) to prevent length bias that compounds entropy collapse.

**Detection:** Entropy plot showing monotonic decline. All `num_generations` outputs becoming identical (measure by average pairwise edit distance between completions).

**Phase:** GRPO training loop (Phase 2).

**Confidence:** HIGH -- documented in DAPO and Dr. GRPO papers (2025).

---

### Pitfall 9: Evaluation Methodology Invalidated by Overlapping Train/Eval Demographics

**What goes wrong:** If the same speakers appear in both training and evaluation sets (even if different utterances), fairness improvements may reflect speaker memorization rather than demographic generalization. Similarly, if Common Voice and Fair-Speech have overlapping speakers, cross-dataset evaluation is contaminated.

**Why it happens:** Neither Common Voice nor Fair-Speech guarantee speaker-disjoint splits. Common Voice's validated/test splits may share speakers with the training partition. Fair-Speech's controlled prompts mean all speakers read similar content, making speaker-specific patterns easier to memorize.

**Prevention:**
1. Verify speaker-disjoint splits: ensure no speaker ID appears in both training and evaluation sets.
2. Use Fair-Speech for evaluation only and Common Voice for training (or vice versa), creating a dataset-disjoint evaluation.
3. Report results on both in-distribution (same dataset) and out-of-distribution (cross-dataset) evaluation.
4. For Fair-Speech, verify that the controlled prompts do not create trivially memorizable patterns.

**Detection:** Suspiciously large fairness improvements on in-distribution data that do not transfer to cross-dataset evaluation.

**Phase:** Evaluation design (Phase 1 data preparation, Phase 4 evaluation).

**Confidence:** MEDIUM -- depends on actual speaker overlap in the specific dataset versions used.

---

### Pitfall 10: Lambda Sweep Produces Degenerate Pareto Frontier

**What goes wrong:** The Pareto frontier plot shows only 2-3 meaningful points because most lambda values produce either (a) negligible fairness change (low lambda) or (b) catastrophic accuracy loss (high lambda), with a very narrow band of useful lambda values.

**Why it happens:** The fairness reward gradient is much steeper than the accuracy gradient at the boundary where they conflict. Small lambda changes produce large shifts in behavior. A linear sweep (0.0, 0.1, ..., 1.0) misses the interesting region.

**Prevention:**
1. Do a coarse sweep first (0.0, 0.1, 0.2, ..., 1.0), identify the transition zone, then do a fine sweep in that zone (e.g., 0.15, 0.175, 0.2, 0.225, 0.25).
2. Use logarithmic spacing for lambda if the transition is near zero: (0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0).
3. Plot intermediate checkpoints, not just final models -- a single training run may trace part of the frontier as lambda is held fixed but training progresses.
4. Need minimum 5-6 non-dominated points for a convincing Pareto plot.

**Detection:** Fewer than 4 distinct points on the Pareto frontier. Most lambda values clustering at one of two extremes.

**Phase:** Lambda sweep (Phase 3). But sweep strategy must be designed in Phase 2.

**Confidence:** MEDIUM -- common in multi-objective optimization but severity depends on the specific reward landscape.

---

## Minor Pitfalls

### Pitfall 11: LoRA Rank Too High Wastes VRAM; Too Low Limits Expressiveness

**What goes wrong:** With only 16GB VRAM, every MB matters. LoRA rank of 64 (common default) may be unnecessarily high for a 1.7B model doing ASR. But rank 4 may be too low to learn fairness-relevant adjustments.

**Prevention:**
1. Start with rank 16, alpha 32 (alpha = 2x rank is standard).
2. Target `q_proj, k_proj, v_proj, o_proj` in decoder attention layers.
3. Compute trainable parameter count: rank 16 on a 1.7B model is ~0.5-1% of parameters, well within the 1-2% target.
4. If fairness improvement plateaus, try rank 32. If VRAM is tight, try rank 8.

**Phase:** LoRA configuration (Phase 1).

**Confidence:** HIGH -- standard LoRA practice.

---

### Pitfall 12: Existing Codebase Hardcoded Paths Block Training Integration

**What goes wrong:** The existing 28 scripts have hardcoded absolute paths (`/users/PAS2030/srishti/...`) that will break when GRPO training scripts try to reuse data loading, metric computation, or evaluation functions.

**Prevention:**
1. Before writing any GRPO code, extract shared utilities (WER computation, demographic grouping, data loading) into a `lib/` module with configurable paths.
2. At minimum, create a `config.yaml` with all paths and have both old and new scripts read from it.
3. This is already documented in CONCERNS.md -- address it as a prerequisite.

**Phase:** Prerequisite (Phase 0 / infrastructure setup).

**Confidence:** HIGH -- directly observed in codebase analysis.

---

### Pitfall 13: Bootstrap CI Computation During Training Is Too Slow

**What goes wrong:** The existing evaluation pipeline uses bootstrap confidence intervals (1000 resamples), which is appropriate for final evaluation but far too slow to run every N training steps as a monitoring signal.

**Prevention:**
1. Use a fast fairness proxy during training: simple max-min WER gap across groups (no bootstrap).
2. Reserve full bootstrap CI evaluation for checkpointed models (every 500 steps or end of epoch).
3. Log the fast proxy every 50 steps for monitoring, full evaluation every 500 steps.

**Phase:** Training loop design (Phase 2).

**Confidence:** HIGH -- computational cost is straightforward to estimate.

---

### Pitfall 14: Mixed Precision (fp16) Instability with Small LoRA Gradients

**What goes wrong:** LoRA adapters have small gradient magnitudes because they modify a small subspace. In fp16, these gradients can underflow to zero, causing dead adapters that never update.

**Prevention:**
1. Use bf16 if supported (A4000 supports bf16). bf16 has larger dynamic range than fp16.
2. If using fp16, use a gradient scaler (torch.cuda.amp.GradScaler) -- this is standard but easy to forget when implementing custom GRPO loops.
3. Monitor LoRA adapter weight norms every 100 steps. If weights stop changing, gradients are underflowing.

**Phase:** Training infrastructure (Phase 1-2).

**Confidence:** MEDIUM -- depends on gradient magnitudes in practice. A4000 (Ampere) supports bf16.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Data preparation | Demographic group imbalance (#7) | Compute and log group distributions before any training. Implement stratified sampling. |
| Data preparation | Train/eval speaker overlap (#9) | Verify speaker-disjoint splits upfront. |
| Infrastructure setup | VRAM budget (#4) | Profile one forward+backward pass. Pin max sequence lengths. |
| Infrastructure setup | VRAM memory leak (#1) | Monitor VRAM per step from first training run. |
| Reward design | Fairness reward hacking (#2) | Add WER floor constraint. Log per-group absolute WER. |
| Reward design | Unique-answer problem (#3) | Use continuous WER reward, increase temperature, monitor advantage std. |
| GRPO training | Entropy collapse (#8) | Asymmetric clipping, entropy monitoring, entropy bonus. |
| GRPO training | fp16 gradient underflow (#14) | Use bf16 on A4000. Monitor adapter weight norms. |
| Lambda sweep | Degenerate frontier (#10) | Coarse-then-fine sweep strategy. Logarithmic spacing. |
| Baseline: FairLoRA | Domain transfer failure (#5) | Length-normalize loss. Use WER-based variance. |
| Baseline: Group-DRO | Worst-group overfitting (#6) | Strong regularization. Smooth group weights. Early stopping on worst-group val. |
| Evaluation | Statistical validity with small groups (#7) | MIN_GROUP_SIZE threshold. Report group sizes alongside results. |
| Codebase integration | Hardcoded paths (#12) | Extract shared config before writing GRPO code. |

---

## Sources

### GRPO Training
- [GRPO++: Tricks for Making RL Actually Work](https://cameronrwolfe.substack.com/p/grpo-tricks) -- DAPO improvements, Dr. GRPO, entropy collapse, token-level loss
- [GRPO VRAM Requirements For the GPU Poor](https://ghost.oxen.ai/grpo-vram-requirements-for-the-gpu-poor/) -- 16GB training feasibility, memory optimization
- [TRL Issue #2927: GRPO memory usage](https://github.com/huggingface/trl/issues/2927) -- VRAM leak reports
- [TRL Issue #3678: OOM on 20GB VRAM](https://github.com/huggingface/trl/issues/3678) -- Qwen + GRPO OOM
- [Unsloth Issue #3864: VRAM increases per step](https://github.com/unslothai/unsloth/issues/3864) -- Memory leak documentation

### GRPO for ASR
- [Group Relative Policy Optimization for Speech Recognition](https://arxiv.org/pdf/2509.01939) -- Limited ASR improvement, unique-answer problem
- [Advancing Speech Understanding with GRPO](https://arxiv.org/pdf/2509.16990) -- GRPO for speech-aware LMs

### Fairness and ASR
- [CTC-DRO: Robust Optimization for Reducing Language Disparities](https://arxiv.org/abs/2502.01777) -- Group-DRO failures in ASR, CTC loss scaling
- [FairLoRA: Bias Mitigation in Vision Models](https://arxiv.org/html/2410.17358) -- Regularizer formula, domain-specific limitations
- [ASR-FAIRBENCH: Measuring and Benchmarking Equity](https://arxiv.org/html/2505.11572) -- Fairness evaluation methodology
- [Fairness in ASR Isn't One-Size-Fits-All](https://aclanthology.org/2025.findings-emnlp.1044.pdf) -- Metric selection

### Group-DRO
- [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731) -- Regularization requirement (Sagawa et al., 2020)

### Reward Hacking
- [MO-GRPO: Mitigating Reward Hacking on Multi-Objective Problems](https://arxiv.org/html/2509.22047) -- Multi-objective reward hacking in GRPO
- [Reward Shaping to Mitigate Reward Hacking in RLHF](https://arxiv.org/abs/2502.18770) -- Bounded rewards, PAR approach
