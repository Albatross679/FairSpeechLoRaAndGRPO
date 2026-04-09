# Phase 4: RL Prototyping Before Scaling Up RL - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Prototype GRPO (Group Relative Policy Optimization) for fairness-aware ASR fine-tuning on Qwen3-ASR-1.7B before committing to expensive full-scale training in Phase 5. Uses a custom ~300-line training loop (TRL GRPOTrainer does NOT support audio/ASR). Validates training signal exists, profiles VRAM, and runs a 3-lambda sweep on a small subset (~2K samples) to establish the accuracy-fairness tradeoff.

</domain>

<decisions>
## Implementation Decisions

### Algorithm
- **D-01:** Custom GRPO training loop on raw PyTorch + PEFT. TRL GRPOTrainer is NOT viable (no audio data_collator, incompatible with encoder-decoder architecture — confirmed via TRL docs + GitHub issue #2807).
- **D-02:** Use Dr. GRPO variant by default (advantage = reward - mean, no std division). Removes difficulty bias per arxiv:2503.20783.
- **D-03:** Continuous WER reward (`1 - jiwer.wer()`) not binary correct/incorrect. Critical for ASR where the unique-answer problem starves binary rewards.
- **D-04:** Composite reward: R = (1-lambda)(1-WER) + lambda(-|WER_g - WER_mean|). Lambda configurable via CLI.
- **D-05:** Clipped surrogate policy loss (PPO-style), epsilon=0.2, single iteration per batch (mu=1 for prototype).

### VRAM strategy
- **D-06:** GPU is Tesla T4 with 15GB VRAM. Budget 13GB (2GB headroom).
- **D-07:** Start with G=2 candidates, batch_size=1, max_new_tokens=128. Generate candidates sequentially, not batched.
- **D-08:** Reference model via PEFT `disable_adapter_layers()` — no second model copy. Verified PEFT 0.18.1 supports this.
- **D-09:** Gradient accumulation steps=4 to simulate effective batch of 4.
- **D-10:** Aggressive `torch.cuda.empty_cache()` between candidate generations.

### Training configuration
- **D-11:** LoRA config from Phase 2.1 locked_config: rank=4, alpha=4, dropout=0.05, target_mlp=true.
- **D-12:** Learning rate: 1e-5 (between ASRU's 1e-6 full-model and SFT's 3.5e-4 LoRA).
- **D-13:** KL penalty beta=0.04 (per Amazon ASRU 2025 paper).
- **D-14:** Sampling temperature=0.8 for candidate diversity.
- **D-15:** Max steps: 200 for prototype, 200 per lambda for sweep.

### Fairness reward
- **D-16:** Rolling window of last 10 batches for fairness penalty computation. Mitigates Pitfall #5 (single-batch fairness = 0 with batch_size=1).
- **D-17:** WER floor check: if mean_WER > baseline_WER * 1.1, zero out fairness bonus. Prevents reward hacking (Pitfall #3).
- **D-18:** Prototype with 3 lambda values: 0.0, 0.3, 0.7. Full sweep deferred to Phase 5.

### Prototype scope
- **D-19:** Subset size: 2K samples (equal FS/CV split, stratified by demographics).
- **D-20:** Two modes: `prototype` (single lambda) and `sweep` (3 lambdas).
- **D-21:** Signal validation: monitor `frac_reward_zero_std` and `advantage_std`. If frac_zero_std > 0.5, GRPO is not learning.
- **D-22:** Adaptive lambda scheduling deferred to Phase 5. Prototype uses fixed lambda only.

### Claude's Discretion
- Exact warmup schedule (linear warmup over 10 steps)
- W&B project naming and run organization
- Checkpoint frequency (every 100 steps)
- Whether to increase G to 4 if VRAM permits

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 4 infrastructure (NEW)
- `scripts/training/train_grpo.py` — Custom GRPO training loop (prototype + sweep modes)
- `scripts/training/reward.py` — RewardComputer (WER accuracy + fairness composite)
- `.planning/phases/04-rl-prototyping-before-scaling-up-rl/04-RESEARCH.md` — Full research document

### Phase 2.1 outputs (prerequisites)
- `outputs/standard-lora/locked_config.json` — Validated LoRA HP config (rank=4, alpha=4, lr=3.47e-4, target_mlp=true)

### Existing infrastructure (reuse)
- `scripts/training/train_standard_lora.py` — Model loading, LoRA injection, data subset creation
- `scripts/training/data_loader.py` — ASRFairnessDataset, DemographicStratifiedSampler
- `scripts/training/data_collator.py` — DataCollatorForQwen3ASR
- `scripts/training/evaluate_adapter.py` — Evaluation bridge (reuse for checkpoint evaluation)
- `outputs/manifests/fs_train.csv` — Fair-Speech manifest (26,471 rows)
- `outputs/manifests/cv_train.csv` — Common Voice train manifest (18,876 rows)

### Research
- `.planning/research/PITFALLS.md` — Pitfall #3 (premature scaling), Pitfall #9 (speaker-disjoint splits)
- Amazon ASRU 2025 paper (arxiv:2509.01939) — GRPO for ASR, lr=1e-6, G=6-10, beta=0.04
- Dr. GRPO (arxiv:2503.20783) — Remove std normalization for ASR

### Requirements
- `.planning/REQUIREMENTS.md` — GRPO-01 through GRPO-10, EXPT-01, EXPT-04

</canonical_refs>

<code_context>
## Existing Code Insights

### Infrastructure reuse
- `load_model_and_processor()`, `apply_lora()`, `patch_outer_forward()`: Imported from train_standard_lora.py
- `create_stratified_subset()`, `create_speaker_disjoint_split()`: Imported from train_standard_lora.py
- `ASRFairnessDataset`, `DemographicStratifiedSampler`, `collate_fn`: From data_loader.py
- `EnglishTextNormalizer`: Same normalization as evaluate_adapter.py

### New files created
- `reward.py`: RewardComputer with rolling-window fairness, WER floor, per-group diagnostics
- `train_grpo.py`: Full GRPO loop with VRAMMonitor, per-token log-prob computation, candidate generation, clipped surrogate loss

### Open questions (resolve empirically in prototype)
1. Does Qwen3-ASR `generate()` support `do_sample=True` + `temperature`?
2. Exact VRAM for G=2 GRPO step on T4?
3. Appropriate learning rate for LoRA GRPO (starting at 1e-5)?
4. Can fairness reward work with batch_size=1 accumulated over gradient_accumulation?

</code_context>

<specifics>
## Specific Ideas

- Start with `--mode prototype --lambda_ 0.0` to validate pure accuracy GRPO works
- Then `--mode prototype --lambda_ 0.3` to test fairness signal
- Then `--mode sweep` for the full 3-lambda comparison
- Use evaluate_adapter.py after each sweep to get per-group WER comparison

</specifics>

<deferred>
## Deferred Ideas

- Adaptive lambda scheduling (linear warmup, cosine anneal) — Phase 5
- G=4 or higher — only if G=2 VRAM permits
- DAPO asymmetric clipping — Phase 5 if needed
- Full lambda sweep (more than 3 values) — Phase 5
- EXPT-02 (Pareto frontier visualization) — Phase 5
- EXPT-03 (statistical significance testing) — Phase 5

</deferred>

---

*Phase: 04-rl-prototyping-before-scaling-up-rl*
*Context gathered: 2026-04-08*
