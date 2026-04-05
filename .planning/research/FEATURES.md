# Feature Landscape

**Domain:** GRPO fairness-aware ASR fine-tuning pipeline
**Researched:** 2026-04-05

## Table Stakes

Features the pipeline must have to produce valid, publishable results. Missing any of these means the experiment cannot run or the paper lacks credibility.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| GRPO training loop with composite reward | Core contribution of the paper; without it there is no experiment | High | Must generate G candidate outputs per input, score each with R = (1-lambda)(1-WER) + lambda(-\|WER_g - WER\|), compute group-relative advantages, update policy with clipped surrogate loss. Reference: arxiv 2509.01939 validates GRPO for ASR with up to 18.4% relative WER improvement. |
| WER-based reward computation | WER is the standard ASR metric; reward must be computed per-sample at training time | Medium | Use JiWER (already in pipeline) to compute WER on each candidate output. Negated WER (-WER) is the accuracy reward. Must normalize by reference word count. |
| Fairness reward term per demographic group | The entire thesis depends on penalizing demographic disparity during training | Medium | For each training batch, compute group-mean WER, then per-sample penalty -\|WER_g - WER_mean\|. Requires demographic labels available at training time, which both Fair-Speech and CV24 provide. |
| Lambda sweep (Pareto frontier) | Central experimental result; traces accuracy-fairness tradeoff | Low | Train separate models at lambda in {0, 0.1, 0.25, 0.5, 0.75, 1.0} (or similar grid). Each is an independent training run. Low complexity per run but high total compute. |
| LoRA adapter on decoder (frozen encoder) | 16GB VRAM constraint makes full fine-tuning infeasible; LoRA is standard for parameter-efficient fine-tuning of LLM-ASR models | Medium | Freeze Qwen3-ASR encoder, apply LoRA to decoder/LLM layers. Target rank 8-64 (start with 16). The GRPO-for-ASR paper freezes the acoustic encoder and updates projection + LLM; LoRA constrains this further. |
| Standard LoRA baseline (lambda=0) | Control condition; proves GRPO adds value beyond vanilla fine-tuning | Low | Identical to GRPO pipeline with lambda=0 (no fairness reward). Effectively supervised fine-tuning via RL with pure accuracy reward. |
| FairLoRA baseline | Comparison method from the vision fairness literature adapted to ASR | Medium | Loss = L_standard + lambda * Var(L_g across groups). Minimizes variance of per-group losses. Implementation: compute group-stratified losses each batch, add variance penalty. Reference: arxiv 2410.17358. No existing ASR implementation -- must adapt from vision. |
| Group-DRO baseline | Standard fairness optimization method; reviewers will expect it | Medium | Maintain per-group weights, upweight worst-performing group each step. Use CTC-DRO variant's smoothing insight (exponential moving average on group weights) to prevent oscillation. Reference implementation: github.com/kohpangwei/group_DRO (vision) and github.com/Bartelds/ctc-dro (speech). Since Qwen3-ASR uses cross-entropy not CTC, adapt vanilla Group-DRO with smoothed weights. |
| ICASSP 2026 fairness-prompted baseline | Specific comparison target from the project requirements | Medium | Combined loss: L_total = lambda_e * L_ERM + lambda_s * L_SD + lambda_d * L_DRO + lambda_i * L_IRM. Reference: arxiv 2510.18374. Must implement Spectral Decoupling (penalize squared logit magnitude), Group-DRO component, and IRM (invariant risk minimization). Default hyperparams from paper: lambda_e=1, lambda_d=1, lambda_s=0.06, lambda_i=0.01. |
| Evaluation on Fair-Speech and Common Voice 24 | Both datasets already prepared; evaluation on both is required for paper completeness | Low | Reuse existing inference + metrics pipeline (Stage 3-4 in current architecture). Run each trained adapter through run_inference.py, then compute_fairness_metrics.py. |
| Fairness metrics: max-min ratio, gap%, std, bootstrap CIs | Already implemented in pipeline; must be applied to all trained models | Low | Existing compute_fairness_metrics.py handles this. Just run on new prediction CSVs. |
| Pareto frontier visualization | Core figure of the paper; plots accuracy vs. fairness for all methods | Low | Plot (mean_WER, fairness_metric) for each method/lambda. Existing visualization pipeline can be extended. Use matplotlib scatter + Pareto front line. |
| Ethnicity and accent demographic axes | Project requires both; ethnicity is primary gap (203%), accent adds breadth | Low | Both axes have labels in Fair-Speech and CV24 manifests. Fairness reward and evaluation must stratify by both independently. |

## Differentiators

Features that strengthen the paper's contribution but are not strictly required for the pipeline to function. These distinguish the work from a straightforward application of GRPO.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Dr. GRPO / DAPO modifications | Standard GRPO has known issues: length bias and advantage normalization instability. Dr. GRPO removes std normalization in advantage computation; DAPO removes KL divergence and adds token-level gradients. Both are validated improvements from arxiv 2509.01939. | Medium | Implement as configurable options: `--grpo-variant {standard, dr_grpo, dapo}`. Dr. GRPO is the safer bet (simpler change, validated for ASR). |
| Multiple reward functions (WER, exact match, edit distance) | The GRPO-ASR paper found WER rewards best overall but exact match better for clean data. Showing reward function ablation strengthens the paper. | Low | Already computing WER; exact match is trivial (binary). Total edit distance is unnormalized WER numerator. Add as config option. |
| Beam search vs. multinomial sampling for GRPO generation | GRPO-ASR paper found beam search better for noisy data, multinomial better for clean. Ablation adds depth to analysis. | Low | Both decoding strategies supported by Qwen3-ASR. Configure in generation step of GRPO loop. |
| Perturbation robustness evaluation | Existing pipeline already has 12 perturbation conditions. Showing GRPO-fairness models maintain robustness under noise/reverb is a strong result. | Low | Reuse generate_perturbations.py + inference pipeline. Just run each adapter through perturbed conditions. High value, very low marginal cost. |
| Error decomposition by demographic group | Existing error_decomposition.py breaks errors into sub/ins/del. Showing GRPO reduces specific error types for disadvantaged groups adds interpretability. | Low | Already built. Run on new prediction CSVs. |
| Adaptive lambda scheduling | Instead of fixed lambda, anneal from 0 to target lambda during training (accuracy-first, then fairness). Motivated by curriculum learning: model first learns to transcribe, then learns to be fair. | Medium | Not validated in literature for this specific setting. Novel contribution potential but risky. Implement as optional: `--lambda-schedule {fixed, linear_warmup, cosine}`. |
| Per-group WER tracking during training (not just evaluation) | Live monitoring of demographic gaps during training enables early stopping if a run diverges. Also produces training dynamics plots for the paper. | Medium | Log per-group WER every N steps on a held-out validation set. Adds compute overhead but provides rich analysis. |
| Statistical significance testing across methods | Mann-Whitney U or paired bootstrap tests between GRPO and each baseline. Reviewers expect significance tests. | Low | Existing pipeline has bootstrap CIs. Extend with pairwise comparisons. |
| Hyperparameter sensitivity analysis for GRPO | Show that results are robust to group size G, learning rate, beta (KL coefficient). Standard in RL papers. | Medium | Run 2-3 values of each key hyperparameter. Adds compute but strengthens claims. Reference: GRPO-ASR paper tested G in {6, 10} with "insignificant" impact. |

## Anti-Features

Features to deliberately NOT build. Building these would waste time, add complexity without value, or violate project constraints.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Full fine-tuning (no LoRA) | 16GB VRAM makes this infeasible. Even if possible, it would not be a fair comparison since baselines also use parameter-efficient methods. | Use LoRA consistently across all methods. Rank 8-64. |
| PPO with separate value model | GRPO's entire advantage is eliminating the critic/value model. PPO would double memory requirements and negate the compute efficiency argument. | Use GRPO which estimates advantages from group-relative rewards. |
| Custom GRPO training framework from scratch | TRL library has GRPOTrainer. Writing from scratch risks bugs and wastes time. However, TRL's GRPOTrainer is designed for text LLMs, not audio. | Use TRL as reference architecture but implement a custom training loop that handles audio inputs. Borrow the clipped surrogate loss and advantage computation; customize the generation and reward steps for ASR. |
| Intersectional fairness (ethnicity x accent x gender) | Explicitly out of scope per PROJECT.md. Group sizes become too small for meaningful statistics. | Evaluate ethnicity and accent axes independently. Note intersectional analysis as future work. |
| Group-specific LoRA adapters (one adapter per demographic) | Explicitly deferred in PROJECT.md. Fundamentally different approach (personalization vs. fairness). Multiplies parameter count and inference complexity. | Single shared adapter that performs well across all groups -- that is the fairness goal. |
| Real-time / streaming inference | Not relevant for a research pipeline evaluating on fixed test sets. Adds massive complexity. | Batch offline inference, which is what the existing pipeline does. |
| Curriculum learning over demographic groups | Ordering training data by demographic difficulty is theoretically interesting but unvalidated for this setting. Risk of introducing confounders. Recent literature (2025) shows reverse curricula sometimes outperform easy-to-hard, making the design space ambiguous. | Fixed random sampling with demographic-balanced batches. If time permits, test as ablation under Differentiators (adaptive lambda scheduling is the simpler version of this idea). |
| Multi-task learning (ASR + fairness classification) | Auxiliary fairness classification heads add architectural complexity. The reward-based approach (GRPO) already encodes fairness signal without auxiliary tasks. | Keep the architecture clean: single ASR objective, fairness enters only through the reward function. |
| Automated hyperparameter optimization (Optuna, Ray Tune) | Compute budget is too limited for HPO sweeps. The lambda sweep itself is already 6+ training runs. | Manual hyperparameter selection based on GRPO-ASR paper defaults. Grid search only for lambda. |
| Web UI or interactive dashboard | Research pipeline, not a product. | CLI scripts + matplotlib figures, consistent with existing pipeline. |

## Feature Dependencies

```
LoRA adapter integration
  |
  +---> Standard LoRA baseline (lambda=0)
  |
  +---> GRPO training loop
  |       |
  |       +---> WER-based reward computation
  |       |
  |       +---> Fairness reward term
  |       |       |
  |       |       +---> Lambda sweep (Pareto frontier)
  |       |
  |       +---> [Optional] Dr. GRPO / DAPO modifications
  |       |
  |       +---> [Optional] Multiple reward functions
  |       |
  |       +---> [Optional] Beam search vs. multinomial sampling
  |
  +---> FairLoRA baseline (independent of GRPO)
  |
  +---> Group-DRO baseline (independent of GRPO)
  |
  +---> ICASSP fairness-prompted baseline (independent of GRPO)

Evaluation (reuses existing pipeline):
  All trained adapters
    |
    +---> Inference on Fair-Speech + CV24
    |       |
    |       +---> Fairness metrics computation
    |       |
    |       +---> Pareto frontier visualization
    |       |
    |       +---> [Optional] Perturbation robustness evaluation
    |       |
    |       +---> [Optional] Error decomposition
    |       |
    |       +---> [Optional] Statistical significance testing
    |
    +---> [Optional] Per-group WER tracking during training
```

## MVP Recommendation

**Phase 1 -- Core pipeline (must complete first):**
1. LoRA adapter integration for Qwen3-ASR-1.7B (all methods depend on this)
2. GRPO training loop with composite reward (core contribution)
3. Standard LoRA baseline (lambda=0, simplest comparison)

**Phase 2 -- Baselines (can parallelize across team members):**
4. FairLoRA baseline
5. Group-DRO baseline
6. ICASSP fairness-prompted baseline

**Phase 3 -- Evaluation and Pareto frontier:**
7. Lambda sweep (6+ GRPO runs at different lambda values)
8. Evaluation on both datasets with fairness metrics
9. Pareto frontier visualization

**Phase 4 -- Strengthen (if time permits):**
10. Dr. GRPO variant (strongest differentiator for lowest effort)
11. Perturbation robustness evaluation (near-zero cost, reuses pipeline)
12. Statistical significance testing

**Defer:** Adaptive lambda scheduling, hyperparameter sensitivity, curriculum approaches. These are interesting but risky given compute and time constraints.

**Critical path:** LoRA integration -> GRPO loop -> lambda sweep -> evaluation -> Pareto plot. Baselines can proceed in parallel once LoRA integration is done.

## Baseline Implementation Reference

| Baseline | Reference Paper | Reference Code | Adaptation Needed |
|----------|----------------|---------------|-------------------|
| Standard LoRA | N/A (standard practice) | HuggingFace PEFT | Minimal -- apply to Qwen3-ASR decoder layers |
| FairLoRA | arxiv 2410.17358 (vision) | No public code found | Significant -- adapt from vision classification to ASR sequence generation. Core idea (variance of group losses) transfers directly. |
| Group-DRO | Sagawa et al. 2020 (arxiv 1911.08731) | github.com/kohpangwei/group_DRO | Moderate -- vanilla Group-DRO is well-understood. Adapt loss weighting to cross-entropy ASR loss. Consider CTC-DRO smoothing insights (github.com/Bartelds/ctc-dro). |
| ICASSP fairness-prompted | arxiv 2510.18374 | No public code found | Significant -- must implement SD + Group-DRO + IRM combined loss. Individual components are well-documented but combination requires careful tuning. |

## Sources

- [GRPO for Speech Recognition](https://arxiv.org/html/2509.01939v1) -- Primary reference for GRPO-ASR implementation
- [FairLoRA: Fairness-Regularized LoRA](https://arxiv.org/html/2410.17358v1) -- FairLoRA method for vision, to be adapted
- [Fairness-Prompted Finetuning for ASR](https://arxiv.org/html/2510.18374) -- ICASSP baseline (SD + DRO + IRM)
- [CTC-DRO: Robust Optimization for Speech](https://arxiv.org/abs/2502.01777) -- Group-DRO adaptation for speech; code at [GitHub](https://github.com/Bartelds/ctc-dro)
- [Group-DRO Reference Implementation](https://github.com/kohpangwei/group_DRO) -- Original Group-DRO code (vision)
- [TRL GRPOTrainer](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl) -- HuggingFace GRPO reference for LLMs
- [Controllable Pareto Trade-off](https://arxiv.org/abs/2509.13651) -- Multi-objective fairness-accuracy optimization
- [ASR-FAIRBENCH](https://www.isca-archive.org/interspeech_2025/rai25_interspeech.pdf) -- Fairness evaluation framework for ASR (Interspeech 2025)
- [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v2) -- Model architecture and training details
