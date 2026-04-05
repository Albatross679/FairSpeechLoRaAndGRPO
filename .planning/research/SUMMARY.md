# Project Research Summary

**Project:** GRPO Fairness-Aware ASR Fine-Tuning for Qwen3-ASR-1.7B
**Domain:** Reinforcement learning for fairness-constrained automatic speech recognition
**Researched:** 2026-04-05
**Confidence:** MEDIUM-HIGH

## Executive Summary

This project applies Group Relative Policy Optimization (GRPO) to fine-tune Qwen3-ASR-1.7B with a composite reward that balances transcription accuracy (WER) against demographic fairness (WER parity across ethnic and accent groups). The recommended approach is a custom ~200-line GRPO training loop built on PyTorch + PEFT (LoRA), not TRL's GRPOTrainer, because Qwen3-ASR is an audio-in/text-out encoder-decoder model that does not fit TRL's text-only API. LoRA adapters target only the LLM decoder's attention layers (~3.5M trainable params out of 2B), keeping the audio encoder frozen. This fits comfortably within the 16GB RTX A4000 VRAM budget. Four baselines (Standard LoRA, FairLoRA, Group-DRO, ICASSP fairness-prompted) share the same adapter infrastructure and evaluation pipeline.

The central experimental deliverable is a Pareto frontier plotting accuracy vs. fairness across methods and lambda values. The existing 28-script benchmarking pipeline is treated as read-only; all new training code lives in a separate `scripts/training/` directory and reuses manifest CSVs and evaluation patterns without modifying existing scripts. This clean separation reduces integration risk.

The three highest-severity risks are: (1) GRPO VRAM memory leaks causing OOM on the tight 16GB budget, (2) reward hacking where the model achieves fairness by degrading all groups equally rather than lifting underperforming ones, and (3) the "unique-answer problem" where ASR's single correct answer per utterance produces near-zero advantage variance, starving GRPO of learning signal. All three have documented mitigations: VRAM monitoring + aggressive cache clearing, WER floor constraints + per-group absolute WER logging, and continuous WER rewards + increased sampling temperature.

## Key Findings

### Recommended Stack

The stack is almost entirely composed of existing dependencies. Only PEFT (for LoRA), optionally bitsandbytes (for QLoRA fallback), and optionally wandb (for experiment tracking) are new. The GRPO algorithm is implemented in project code, not imported from any RL library. See [STACK.md](./STACK.md) for full details.

**Core technologies:**
- **PyTorch 2.2+ (existing):** Custom GRPO training loop with raw autograd control
- **PEFT 0.18.x (new):** LoRA adapter injection on decoder attention (q/k/v/o_proj), ~3.5M trainable params
- **Transformers 4.48+ (existing):** Model loading, tokenizer, from_pretrained ecosystem
- **jiwer (existing):** Per-utterance WER computation for the reward function
- **bitsandbytes 0.45+ (new, optional):** 4-bit NF4 quantization fallback if FP16 LoRA hits OOM

**Key decision:** Start with FP16 LoRA (simpler, ~8-12GB VRAM). Fall back to QLoRA only if OOM with G=4, batch_size=2. Do NOT use TRL GRPOTrainer, OpenRLHF, Unsloth, or DeepSpeed -- none support Qwen3-ASR's audio encoder architecture.

### Expected Features

See [FEATURES.md](./FEATURES.md) for the full landscape and dependency graph.

**Must have (table stakes):**
- GRPO training loop with composite reward R = (1-lambda)(1-WER) + lambda(-|WER_g - WER_mean|)
- WER-based reward and fairness reward per demographic group
- LoRA adapter on decoder with frozen encoder
- Lambda sweep producing Pareto frontier (6+ values)
- All four baselines: Standard LoRA, FairLoRA, Group-DRO, ICASSP fairness-prompted
- Evaluation on Fair-Speech and Common Voice 24 with full fairness metrics
- Ethnicity and accent demographic axes

**Should have (differentiators):**
- Dr. GRPO variant (removes std normalization, validated for ASR) -- lowest effort, highest value
- Perturbation robustness evaluation (reuses existing pipeline, near-zero marginal cost)
- Statistical significance testing (extends existing bootstrap CIs)
- Per-group WER tracking during training for dynamics plots

**Defer:**
- Adaptive lambda scheduling, hyperparameter sensitivity analysis, intersectional fairness, group-specific adapters

### Architecture Approach

The architecture separates cleanly into a Training Subsystem (new) and an Evaluation Bridge (wrapping existing code). All methods share FairnessDataLoader, AdapterManager, and EvaluationBridge components. They differ only in their training objective/loss function. The existing pipeline's manifest-as-contract pattern is preserved: training reads the same CSV manifests already produced by prepare scripts. See [ARCHITECTURE.md](./ARCHITECTURE.md) for component diagrams and data flow.

**Major components:**
1. **FairnessDataLoader** -- Demographic-stratified batch sampling ensuring each batch has samples from multiple groups
2. **RewardComputer** -- Composite accuracy-fairness reward with group normalization
3. **GRPOTrainer** -- Custom policy gradient loop with clipped ratio, KL penalty, LoRA-only updates
4. **BaselineTrainers** -- Standard LoRA (SFT), FairLoRA (variance regularizer), Group-DRO (worst-group), ICASSP (fusion loss)
5. **AdapterManager** -- LoRA config/save/load/merge for Qwen3-ASR via PEFT
6. **EvaluationBridge** -- Adapter loading, inference, fairness metrics computation, Pareto plotting

### Critical Pitfalls

See [PITFALLS.md](./PITFALLS.md) for the full list of 14 pitfalls with detection and prevention strategies.

1. **GRPO VRAM memory leak** -- Monitor `torch.cuda.max_memory_allocated()` every 50 steps; force `empty_cache() + gc.collect()` each step; abort if VRAM grows >500MB over 200 steps
2. **Fairness reward hacking via accuracy collapse** -- Log per-group absolute WER (not just composite reward); add WER floor constraint; validate held-out WER at each checkpoint
3. **Unique-answer problem kills GRPO signal** -- Use continuous WER reward (not binary); increase sampling temperature to 0.7-1.0; monitor advantage std within groups
4. **16GB VRAM exhaustion** -- Use PEFT's `disable_adapter_layers()` for reference policy (no second model copy); gradient checkpointing mandatory; start G=2 then try G=4
5. **Demographic group imbalance corrupts fairness signal** -- Stratified batch sampling with minimum N samples per group; rolling-window fairness reward computation

## Implications for Roadmap

### Phase 1: Foundation and Infrastructure
**Rationale:** Every subsequent phase depends on LoRA adapter integration working with Qwen3-ASR and the data pipeline being demographic-aware. This phase also validates the VRAM budget before committing to long training runs.
**Delivers:** Working LoRA adapter for Qwen3-ASR, demographic-stratified data loader, shared config, VRAM profiling
**Addresses:** LoRA adapter integration, FairnessDataLoader, config extraction
**Avoids:** Pitfall #4 (VRAM budget), Pitfall #12 (hardcoded paths), Pitfall #7 (group imbalance)

### Phase 2: Standard LoRA Baseline + Evaluation Bridge
**Rationale:** Standard SFT with LoRA is the simplest training method and validates the entire adapter-to-evaluation pipeline end-to-end. It also produces the accuracy-only baseline that all fairness methods are compared against.
**Delivers:** Trained standard LoRA adapter, evaluation bridge producing fairness metrics, baseline WER numbers
**Addresses:** Standard LoRA baseline (lambda=0), evaluation on both datasets, fairness metrics
**Avoids:** Pitfall #9 (evaluation methodology -- validate speaker-disjoint splits here)

### Phase 3: GRPO Training Loop + Reward Design
**Rationale:** The core contribution. Depends on Phase 1 (adapter, data loader) and Phase 2 (evaluation bridge for validation). Reward design is the most research-intensive component and must be validated before running expensive lambda sweeps.
**Delivers:** Working GRPO loop with composite fairness reward, initial lambda=0.5 training run, reward monitoring dashboards
**Addresses:** GRPO training loop, WER reward, fairness reward, per-group tracking
**Avoids:** Pitfall #1 (memory leak), Pitfall #2 (reward hacking), Pitfall #3 (unique-answer problem), Pitfall #8 (entropy collapse)

### Phase 4: Fairness Baselines
**Rationale:** Can proceed in parallel with GRPO tuning once Phase 2 infrastructure is validated. All three baselines share the same data loader and evaluation bridge but differ in loss function. FairLoRA and Group-DRO are moderate complexity; ICASSP is the most complex.
**Delivers:** Trained adapters for FairLoRA, Group-DRO, ICASSP fairness-prompted baselines
**Addresses:** All three fairness baselines from FEATURES.md
**Avoids:** Pitfall #5 (FairLoRA domain transfer -- use WER-based variance), Pitfall #6 (Group-DRO overfitting -- strong regularization + early stopping)

### Phase 5: Lambda Sweep + Pareto Frontier
**Rationale:** Requires a validated GRPO loop (Phase 3) and all baselines (Phase 4) to be complete. This is the central experimental result. Use coarse-then-fine sweep strategy.
**Delivers:** Pareto frontier figure with all 5 methods, analysis of accuracy-fairness tradeoff
**Addresses:** Lambda sweep, Pareto visualization, cross-method comparison
**Avoids:** Pitfall #10 (degenerate frontier -- coarse sweep first, then fine-grained in transition zone)

### Phase 6: Strengthening and Paper Figures
**Rationale:** Only after core results exist. Low-cost differentiators that reuse existing infrastructure.
**Delivers:** Dr. GRPO ablation, perturbation robustness results, statistical significance tests, error decomposition
**Addresses:** Differentiator features from FEATURES.md
**Avoids:** Over-engineering before core results are in hand

### Phase Ordering Rationale

- **Foundation first (Phase 1-2):** LoRA-on-Qwen3-ASR integration is the highest technical risk and blocks everything. Validating it early with a simple SFT baseline catches adapter/VRAM issues before they become expensive.
- **GRPO before baselines (Phase 3 before 4):** The GRPO reward function and training loop are the core contribution and most novel. If GRPO fundamentally does not work for ASR fairness, the project pivots. Baselines are straightforward adaptations of known methods.
- **Lambda sweep last (Phase 5):** Each lambda value is an independent training run. This is compute-expensive and should only run after the loop is validated and reward hacking is prevented.
- **Baselines parallelizable with GRPO tuning:** Once Phase 2 infrastructure works, Phase 3 (GRPO) and Phase 4 (baselines) can proceed in parallel if two people are working.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Qwen3-ASR LoRA integration -- the `qwen-asr` package is an inference wrapper; accessing the underlying transformers model for gradient-based training is underdocumented. Key risk to validate early.
- **Phase 3:** GRPO reward design for ASR -- the unique-answer problem is documented but mitigations are not yet validated for fairness-composite rewards. Reward shaping (continuous WER, temperature, floor constraints) needs empirical testing.
- **Phase 4 (ICASSP baseline):** The fairness-prompted fusion loss (ERM + SD + IRM + DRO) has 4 interacting hyperparameters and no public code. Implementation requires careful reading of the source paper.

Phases with standard patterns (skip deeper research):
- **Phase 2:** Standard LoRA SFT is extremely well-documented with established PEFT patterns.
- **Phase 5:** Lambda sweep is mechanically straightforward once the training loop works.
- **Phase 6:** All strengthening items reuse existing pipeline infrastructure.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Minimal new dependencies. PyTorch + PEFT is battle-tested. TRL exclusion well-justified by audio-in architecture mismatch. |
| Features | HIGH | Table stakes are clearly defined by project requirements. Feature dependency graph is clean. MVP phasing is straightforward. |
| Architecture | MEDIUM-HIGH | Component boundaries are clean. Key uncertainty: whether `qwen-asr` package exposes the transformers model cleanly for LoRA injection. |
| Pitfalls | HIGH | Critical pitfalls (VRAM leak, reward hacking, unique-answer) are well-documented in published literature with specific mitigations. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Qwen3-ASR LoRA integration path:** The `qwen-asr` package may not expose the HuggingFace model in a way that supports PEFT LoRA injection. Must validate in Phase 1 by loading the model and inspecting its module hierarchy. Fallback: load directly via transformers `AutoModelForCausalLM` and reconstruct the audio pipeline manually.
- **GRPO effectiveness for ASR fairness:** Published GRPO-ASR work shows "only minor improvements" for transcription accuracy. The fairness reward component is entirely novel and unvalidated. Phase 3 must include early checkpoints and go/no-go decision points.
- **FairLoRA domain transfer:** FairLoRA has never been applied to ASR. The loss-variance regularizer may not map to WER fairness. Must decide during Phase 4 whether to use loss-based or WER-based variance and document the choice.
- **Exact LoRA target modules for Qwen3-ASR decoder:** The recommended targets (q/k/v/o_proj) follow Qwen3 conventions but are not confirmed for the ASR variant. Validate by inspecting `model.named_modules()` in Phase 1.
- **Reference policy memory:** Plan to use PEFT's `disable_adapter_layers()` for zero-copy reference policy. If this introduces numerical issues with the reward, may need CPU offload of a frozen copy.

## Sources

### Primary (HIGH confidence)
- [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1) -- model architecture (AuT 300M + Qwen3-1.7B decoder)
- [GRPO for Speech Recognition (ASRU 2025)](https://arxiv.org/abs/2509.01939) -- GRPO-ASR validation, unique-answer problem
- [PEFT GitHub](https://github.com/huggingface/peft) -- v0.18.x, LoRA adapter API
- [TRL GRPO Trainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) -- confirmed text-only API limitation
- [Sagawa et al. (ICLR 2020)](https://arxiv.org/abs/1911.08731) -- Group-DRO regularization requirements

### Secondary (MEDIUM confidence)
- [FairLoRA (Sukumaran et al., 2024)](https://arxiv.org/abs/2410.17358) -- fairness regularizer, vision domain only
- [Fairness-Prompted Finetuning (ICASSP 2026)](https://arxiv.org/html/2510.18374) -- fusion loss baseline, no public code
- [CTC-DRO (Bartelds et al., 2025)](https://arxiv.org/abs/2502.01777) -- ASR fairness, loss-length confound
- [GRPO VRAM Requirements](https://ghost.oxen.ai/grpo-vram-requirements-for-the-gpu-poor/) -- memory estimation
- [FairGRPO for Clinical Reasoning](https://arxiv.org/abs/2510.19893) -- adaptive importance weighting for demographic fairness in GRPO

### Tertiary (LOW confidence)
- [Qwen3-ASR GitHub Issue #68](https://github.com/QwenLM/Qwen3-ASR/issues/68) -- community discussion on fine-tuning challenges, unresolved
- VRAM budget estimates -- calculated, not measured; actual values depend on Qwen3-ASR memory allocation patterns

---
*Research completed: 2026-04-05*
*Ready for roadmap: yes*
