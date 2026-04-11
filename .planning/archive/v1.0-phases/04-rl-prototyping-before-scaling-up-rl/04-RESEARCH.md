# Phase 4: RL Prototyping Before Scaling Up RL - Research

**Researched:** 2026-04-08
**Domain:** GRPO (Group Relative Policy Optimization) for fairness-aware ASR fine-tuning
**Confidence:** MEDIUM-HIGH

## Summary

Phase 4 prototypes GRPO for fairness-aware ASR fine-tuning of Qwen3-ASR-1.7B before committing to expensive full-scale training in Phase 5. The core challenge is implementing a custom GRPO training loop that generates G candidate transcriptions per audio input, computes composite accuracy+fairness rewards, and performs group-relative policy gradient updates -- all within the 15GB VRAM budget of a Tesla T4.

TRL's GRPOTrainer does NOT support audio/ASR models -- it lacks audio data_collator support and cannot handle Qwen3-ASR's encoder-decoder architecture. The correct approach is a custom ~300-line GRPO training loop built on PyTorch + PEFT, reusing the existing model loading, LoRA injection, and data collation infrastructure from Phases 1-2. The Amazon ASRU 2025 paper (arxiv:2509.01939) validates GRPO for ASR with up to 18.4% relative WER improvement, using WER-based rewards with G=6-10 and lr=1e-6.

The three critical risks are: (1) the unique-answer problem where ASR has mostly one correct transcription, starving GRPO of advantage variance -- mitigated by continuous WER rewards and higher temperature sampling; (2) VRAM exhaustion from G=4 generations + reference model KL computation on a 15GB T4 -- mitigated by CPU-offloaded reference log-probs and starting with G=2; (3) fairness reward hacking where the model achieves equality by degrading all groups -- mitigated by per-group absolute WER monitoring and WER floor constraints.

**Primary recommendation:** Implement a custom GRPO training loop (not TRL) with continuous WER reward, start G=2 on a small subset (~2K samples), validate training signal exists (advantage std > 0.01), then scale to G=4 if VRAM permits. Prototype with 3 lambda values (0, 0.3, 0.7) before the full sweep in Phase 5.

## Project Constraints (from CLAUDE.md)

- GPU Safety: Run `nvidia-smi` before GPU tasks, do not launch if >80% utilization
- Documentation: Log every significant change immediately to `logs/`, `experiments/` etc. with proper fileClass frontmatter
- Architecture: Script-oriented pipeline, all training code in `scripts/training/`
- Existing pipeline is read-only -- do not modify existing inference/evaluation scripts

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GRPO-01 | Custom GRPO training loop generates G candidate outputs per input audio | Custom loop architecture documented in Architecture Patterns; TRL not viable for audio |
| GRPO-02 | WER-based accuracy reward: R_acc = (1-lambda)(1-WER) | RewardComputer pattern; jiwer + whisper_normalizer verified available |
| GRPO-03 | Fairness penalty: R_fair = lambda(-\|WER_g - WER_mean\|) | Requires demographic-aware batch accumulation, not per-sample |
| GRPO-04 | Composite reward R = R_acc + R_fair with configurable lambda | Lambda as CLI arg; prototype with 3 values |
| GRPO-05 | Group-relative advantage normalization across G candidates | Standard GRPO advantage; Dr. GRPO variant removes std division |
| GRPO-06 | Clipped surrogate policy loss for stable updates | Epsilon=0.2, single iteration (mu=1) for prototype |
| GRPO-07 | Dr. GRPO variant: removes std normalization, adds length normalization | Flag in config; set scale_rewards=False equivalent |
| GRPO-08 | Stratified demographic batching | Existing DemographicStratifiedSampler reusable |
| GRPO-09 | Per-group WER tracked and logged every N steps | W&B logging with per-group metrics |
| GRPO-10 | Adaptive lambda scheduling option | Prototype with fixed lambda only; scheduling deferred to Phase 5 |
| EXPT-01 | Lambda sweep traces Pareto frontier | 3 values for prototype (0, 0.3, 0.7); full sweep in Phase 5 |
| EXPT-04 | HP sensitivity analysis: G, lr, beta | Prototype establishes baseline values; ablation in Phase 5 |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.8.0+cu128 | Training loop, autograd, CUDA | Already installed, custom GRPO loop requires raw PyTorch [VERIFIED: pip show] |
| peft | 0.18.1 | LoRA adapter injection/save/load | Already installed, proven in Phases 1-2 [VERIFIED: pip show] |
| transformers | 4.57.6 | Model loading, tokenizer, generation | Already installed [VERIFIED: pip show] |
| jiwer | 4.0.0 | WER computation for reward function | Already installed, critical for R_acc [VERIFIED: pip show] |
| wandb | 0.25.1 | Experiment tracking, per-group metrics | Already installed [VERIFIED: pip show] |
| qwen-asr | 0.0.6 | Model loading wrapper | Already installed, used for base model access [VERIFIED: pip show] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| whisper-normalizer | (installed) | Text normalization for WER | Must match evaluation pipeline normalization [VERIFIED: used in evaluate_adapter.py] |
| soundfile | (installed) | Audio loading | Used by existing data_loader.py [VERIFIED: codebase] |

### NOT Using

| Library | Reason |
|---------|--------|
| TRL GRPOTrainer | Does NOT support audio modality, no data_collator for ASR, Qwen3-ASR architecture incompatible [VERIFIED: TRL docs + GitHub issue #2807] |
| OpenRLHF | Text-only RL framework, no audio encoder support [ASSUMED] |
| Unsloth | Optimizes text LLMs only, no multimodal audio support [ASSUMED] |
| DeepSpeed | Overkill for single-GPU prototype; adds complexity without benefit at this scale [ASSUMED] |
| bitsandbytes | Not installed; QLoRA fallback only if FP16 LoRA OOMs -- start without it [VERIFIED: pip show returned nothing] |

**Installation:**
```bash
# No new packages needed -- all dependencies already installed
# If VRAM issues force QLoRA fallback:
pip install bitsandbytes
```

## Architecture Patterns

### Recommended Project Structure

```
scripts/training/
├── train_grpo.py              # NEW: Custom GRPO training loop (primary deliverable)
├── reward.py                  # NEW: RewardComputer (WER + fairness composite)
├── train_standard_lora.py     # EXISTING: reuse model loading, LoRA injection patterns
├── data_loader.py             # EXISTING: ASRFairnessDataset, DemographicStratifiedSampler
├── data_collator.py           # EXISTING: DataCollatorForQwen3ASR (reuse for input preparation)
├── evaluate_adapter.py        # EXISTING: evaluation bridge (reuse as-is for checkpoint eval)
└── ...

outputs/
├── grpo-prototype/            # NEW: Phase 4 outputs
│   ├── lambda_0.0/            # Per-lambda run outputs
│   │   ├── adapter/           # LoRA adapter weights
│   │   ├── training_log.json  # Step-by-step metrics
│   │   └── checkpoints/       # Intermediate checkpoints
│   ├── lambda_0.3/
│   ├── lambda_0.7/
│   └── eval/                  # Evaluation results per lambda
```

### Pattern 1: Custom GRPO Training Loop (Core Algorithm)

**What:** A manual training loop that handles generation, reward computation, advantage normalization, and policy gradient updates without TRL.

**When to use:** Always -- TRL cannot handle Qwen3-ASR's audio encoder architecture.

**Example:**
```python
# Source: GRPO algorithm (DeepSeekMath paper) + ASR adaptation (arxiv:2509.01939)

def grpo_step(model, ref_log_probs_fn, batch, config):
    """Single GRPO training step.
    
    Args:
        model: PeftModel (Qwen3-ASR with LoRA, trainable)
        ref_log_probs_fn: Function computing log-probs under reference policy
        batch: Dict with audio, transcripts, demographic_groups
        config: GRPOConfig (G, lambda_, beta, epsilon, temperature)
    
    Returns:
        loss, metrics_dict
    """
    # Step 1: Generate G candidate transcriptions per input
    model.eval()
    with torch.no_grad():
        candidates = []  # List of G transcription strings per input
        candidate_log_probs = []  # Per-token log-probs for each candidate
        for g in range(config.G):
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=config.temperature,  # 0.7-1.0 for diversity
                return_dict_in_generate=True,
                output_scores=True,
            )
            # Extract generated tokens and compute log-probs
            ...
            candidates.append(transcriptions)
            candidate_log_probs.append(log_probs)
    
    # Step 2: Compute rewards for each candidate
    rewards = compute_composite_reward(
        candidates, batch["transcripts"], batch["demographic_groups"],
        lambda_=config.lambda_
    )  # shape: (batch_size, G)
    
    # Step 3: Compute advantages (group-relative normalization)
    if config.dr_grpo:
        # Dr. GRPO: no std division
        advantages = rewards - rewards.mean(dim=1, keepdim=True)
    else:
        # Standard GRPO
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = (rewards - mean) / std
    
    # Step 4: Compute policy loss with clipped ratio
    model.train()
    # Re-compute log-probs under current policy (forward pass with grad)
    current_log_probs = compute_log_probs(model, batch_inputs, candidate_tokens)
    
    # Reference log-probs (computed on CPU or cached)
    ref_lp = ref_log_probs_fn(batch_inputs, candidate_tokens)
    
    # Importance ratio
    ratio = torch.exp(current_log_probs - old_log_probs)
    
    # Clipped surrogate objective
    clipped_ratio = ratio.clamp(1 - config.epsilon, 1 + config.epsilon)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    # Optional KL penalty
    if config.beta > 0:
        kl = (current_log_probs - ref_lp).mean()
        loss = loss + config.beta * kl
    
    return loss, {"reward_mean": rewards.mean().item(), ...}
```

### Pattern 2: Reference Model via Frozen LoRA Disable

**What:** Instead of loading a separate reference model (doubles VRAM), disable LoRA adapters to get reference policy log-probs.

**When to use:** Always on single-GPU setups with tight VRAM.

**Example:**
```python
# Source: PEFT docs -- disable_adapter_layers()
# [VERIFIED: PEFT 0.18 supports disable_adapter_layers]

def compute_ref_log_probs(model, inputs, generated_ids):
    """Compute log-probs under reference policy by disabling LoRA."""
    model.eval()
    with torch.no_grad():
        # Temporarily disable LoRA -- model reverts to base weights
        model.disable_adapter_layers()
        outputs = model(**inputs, labels=generated_ids)
        ref_log_probs = -outputs.loss  # Or compute per-token log-probs
        model.enable_adapter_layers()
    return ref_log_probs
```

### Pattern 3: Demographic-Aware Reward Computation

**What:** Compute fairness reward at the GROUP level, not per-sample. Accumulate WER within demographic groups across the batch.

**When to use:** Every GRPO step that includes a fairness penalty (lambda > 0).

**Example:**
```python
# Source: Project REQUIREMENTS.md GRPO-03

def compute_composite_reward(candidates, references, demographics, lambda_):
    """Compute R = (1-lambda)(1-WER_i) + lambda(-|WER_g - WER_mean|).
    
    Args:
        candidates: List[List[str]] -- G transcriptions per input
        references: List[str] -- ground truth transcriptions
        demographics: List[str] -- demographic group per input
        lambda_: float -- fairness weight (0=accuracy only, 1=fairness only)
    """
    normalizer = EnglishTextNormalizer()
    batch_size = len(references)
    G = len(candidates)
    
    rewards = torch.zeros(batch_size, G)
    
    for g in range(G):
        # Per-utterance WER
        wers = []
        for i in range(batch_size):
            ref = normalizer(references[i])
            hyp = normalizer(candidates[g][i])
            wer = jiwer.wer(ref, hyp) if ref else 1.0
            wers.append(wer)
        
        wers_tensor = torch.tensor(wers)
        
        # Accuracy reward: (1-lambda)(1-WER)
        r_acc = (1 - lambda_) * (1 - wers_tensor)
        
        # Fairness penalty: lambda(-|WER_g - WER_mean|)
        if lambda_ > 0:
            # Group WER within this batch
            group_wers = {}
            for i, demo in enumerate(demographics):
                if demo:  # Skip unlabeled
                    group_wers.setdefault(demo, []).append(wers[i])
            
            wer_mean = wers_tensor.mean().item()
            r_fair = torch.zeros(batch_size)
            for i, demo in enumerate(demographics):
                if demo and demo in group_wers and len(group_wers[demo]) >= 2:
                    group_mean = np.mean(group_wers[demo])
                    r_fair[i] = -abs(group_mean - wer_mean)
                # else: r_fair[i] = 0 (no fairness penalty for unlabeled)
            
            rewards[:, g] = r_acc + lambda_ * r_fair
        else:
            rewards[:, g] = r_acc
    
    return rewards
```

### Pattern 4: VRAM-Safe Generation Loop

**What:** Generate G candidates sequentially (not batched) to keep peak VRAM bounded, with explicit cache clearing between generations.

**When to use:** On 15GB T4 with G >= 2.

**Example:**
```python
# Source: Pitfall #1 from project PITFALLS.md

def generate_candidates_safe(model, batch_inputs, G, temperature, max_tokens=128):
    """Generate G candidates per input with aggressive VRAM management."""
    candidates = []
    all_scores = []
    
    model.eval()
    for g in range(G):
        with torch.no_grad():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
            )
        candidates.append(outputs.sequences)
        all_scores.append(outputs.scores)
        
        # Aggressive cleanup between generations
        torch.cuda.empty_cache()
    
    return candidates, all_scores
```

### Anti-Patterns to Avoid

- **Using TRL GRPOTrainer:** Does not support audio data_collator, will fail on Qwen3-ASR. Build custom loop.
- **Loading separate reference model on GPU:** Doubles VRAM from ~3.4GB to ~6.8GB. Use `disable_adapter_layers()` instead.
- **Computing fairness reward per-sample:** Fairness is a group-level property. Per-sample computation yields meaningless signal.
- **Batching all G generations simultaneously:** G=4 with batch_size=2 means 8 concurrent sequences in KV cache. Generate sequentially.
- **Using `model.generate()` for log-prob computation:** Generation and log-prob computation are separate operations. Use `model.forward()` with labels for log-probs, `model.generate()` for candidate text.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| WER computation | Custom edit distance | `jiwer.wer()` | Edge cases in normalization, insertion/deletion counting [VERIFIED: jiwer 4.0 installed] |
| Text normalization | Custom lowercasing/stripping | `whisper_normalizer.english.EnglishTextNormalizer` | Must match evaluation pipeline exactly [VERIFIED: used in evaluate_adapter.py] |
| LoRA injection | Manual weight surgery | `peft.get_peft_model()` + `LoraConfig` | Proven in Phases 1-2, handles save/load/merge lifecycle [VERIFIED: peft 0.18.1] |
| Demographic batching | Custom batch sampler | `DemographicStratifiedSampler` from `data_loader.py` | Already implemented and validated [VERIFIED: codebase] |
| Audio preprocessing | Custom mel spectrogram | `Qwen3ASRProcessor` via qwen-asr | Model-specific preprocessing, tokenization, chat template [VERIFIED: codebase] |
| Gradient checkpointing | Manual recomputation | `model.gradient_checkpointing_enable()` | Already used in existing training scripts [VERIFIED: codebase] |
| KL divergence estimator | Custom formula | Schulman (2020) approximator: `r_ref/r_policy - log(r_ref/r_policy) - 1` | Numerically stable, standard in GRPO literature [CITED: TRL docs] |

**Key insight:** The GRPO algorithm itself (~200-300 lines) MUST be hand-rolled because no existing library supports Qwen3-ASR's audio encoder architecture. But everything around it (WER, LoRA, data loading, evaluation) already exists in the codebase.

## Common Pitfalls

### Pitfall 1: Unique-Answer Problem Starves GRPO Learning Signal

**What goes wrong:** ASR has exactly one correct transcription per utterance. When all G candidates produce similar WER (either all correct or all wrong), advantage normalization yields near-zero gradients. The Amazon ASRU 2025 paper explicitly noted "preliminary experiments with ASR obtained only minor improvements."
**Why it happens:** GRPO computes advantages by comparing rewards within a group. If all candidates get the same reward, advantage = 0 for all.
**How to avoid:**
1. Use continuous WER reward (`1 - jiwer.wer()`) NOT binary correct/incorrect
2. Set temperature >= 0.7 during generation for candidate diversity
3. Use G >= 4 (Amazon used G=6-10; they found impact "insignificant" between 6 and 10)
4. Monitor `frac_reward_zero_std` -- fraction of prompts where all G candidates get identical reward. If > 50%, GRPO is not getting signal.
**Warning signs:** Advantage std near zero across batches; WER improvement < 0.5% after 200+ steps.

### Pitfall 2: VRAM Exhaustion During GRPO Generation

**What goes wrong:** Generating G candidates requires G forward passes with KV cache allocation. On a 15GB T4, G=4 with batch_size=2 can exhaust memory.
**Why it happens:** Each generation allocates KV cache (~0.5-1GB per sequence for 1.7B model). G=4 means 4x cache. Add optimizer states and activation memory.
**How to avoid:**
1. Generate candidates sequentially (one at a time), clearing cache between generations
2. Start with G=2, validate VRAM, then try G=4
3. Use `batch_size=1, gradient_accumulation_steps=4` for GRPO (not batch_size=4)
4. Set `max_new_tokens=128` (ASR transcriptions are short -- most < 50 tokens)
5. Monitor with `torch.cuda.max_memory_allocated()` every 10 steps
**Warning signs:** VRAM monotonically increasing over steps; OOM after N steps.

### Pitfall 3: Fairness Reward Hacking via Accuracy Collapse

**What goes wrong:** Model maximizes fairness reward by making ALL groups equally bad (WER gap = 0 when all groups have WER = 1.0).
**Why it happens:** The fairness term rewards equality of WER, not low WER. With high lambda, degrading majority group to match minority group is easier than lifting minority.
**How to avoid:**
1. Log per-group absolute WER alongside composite reward at every eval step
2. Add WER floor: if mean_WER > baseline_WER * 1.1, override fairness bonus to zero
3. Start lambda sweep from 0 and increase gradually
4. Prototype with lambda = {0, 0.3, 0.7} to see if there's a useful operating point
**Warning signs:** Composite reward increasing while per-group WER also increasing; groups converging toward worst-group WER.

### Pitfall 4: Log-Probability Computation Mismatch

**What goes wrong:** Log-probs computed during generation (with KV cache, autoregressive) differ from log-probs computed in training forward pass (teacher-forced). This creates biased importance ratios.
**Why it happens:** Qwen3-ASR's `generate()` uses its own KV cache and sampling logic. The training forward pass uses teacher-forcing. Numerical differences accumulate.
**How to avoid:**
1. Compute "old" log-probs via a separate teacher-forced forward pass AFTER generation, not from generation scores
2. Use the same forward function for both old and current log-probs
3. The importance ratio `pi_theta / pi_old` should start at exactly 1.0 at the beginning of each iteration
**Warning signs:** Initial ratio values != 1.0; clip_ratio metrics showing > 50% clipping from the first step.

### Pitfall 5: Incorrect Reward Normalization Across Demographic Groups

**What goes wrong:** The fairness penalty `|WER_g - WER_mean|` is computed within a single batch. With small batch sizes (1-2), a batch may contain only 1 demographic group, making the fairness signal trivially zero.
**Why it happens:** Small effective batch size + demographic stratification doesn't guarantee multiple groups per batch at batch_size=1.
**How to avoid:**
1. Use a rolling window of recent batches (e.g., last 10) for fairness reward computation, not just current batch
2. Ensure batch_size >= 4 for fairness reward computation (even if gradient_accumulation > 1)
3. Or: accumulate per-group WER across the full gradient accumulation window before computing fairness penalty
**Warning signs:** Fairness reward = 0 for most steps; only 1 demographic group per batch.

## Code Examples

### Computing Per-Token Log-Probabilities

```python
# Source: Standard transformer log-prob computation pattern
# [VERIFIED: works with transformers 4.57.6 + Qwen3-ASR forward]

def compute_per_token_log_probs(model, input_ids, attention_mask,
                                  input_features, feature_attention_mask,
                                  generated_ids):
    """Compute per-token log-probs for generated tokens via teacher forcing.
    
    Uses the same forward path as training (not generate()) to ensure
    consistency between old_log_probs and current_log_probs.
    """
    # Concatenate input + generated tokens
    full_ids = torch.cat([input_ids, generated_ids], dim=1)
    full_mask = torch.cat([
        attention_mask,
        torch.ones_like(generated_ids)
    ], dim=1)
    
    # Forward pass (teacher-forced)
    outputs = model(
        input_ids=full_ids,
        attention_mask=full_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
    )
    
    # Extract log-probs for generated positions only
    logits = outputs.logits  # (B, seq_len, vocab_size)
    gen_start = input_ids.shape[1]
    gen_logits = logits[:, gen_start-1:-1, :]  # Shifted for next-token prediction
    
    log_probs = torch.log_softmax(gen_logits, dim=-1)
    
    # Gather log-probs for actual generated tokens
    token_log_probs = log_probs.gather(
        2, generated_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, gen_len)
    
    return token_log_probs
```

### VRAM Monitoring Utility

```python
# Source: Project convention from train_standard_lora.py + PITFALLS.md

import gc

class VRAMMonitor:
    """Track VRAM usage across training steps, abort if leak detected."""
    
    def __init__(self, budget_gb=13.0, leak_threshold_mb=500, window=200):
        self.budget_gb = budget_gb
        self.leak_threshold_mb = leak_threshold_mb
        self.window = window
        self.history = []
    
    def check(self, step):
        """Call every N steps. Returns (ok, peak_gb)."""
        if not torch.cuda.is_available():
            return True, 0.0
        
        peak = torch.cuda.max_memory_allocated() / 1024**3
        self.history.append((step, peak))
        
        # Budget check
        if peak > self.budget_gb:
            print(f"WARNING: Step {step}: Peak VRAM {peak:.2f}GB > budget {self.budget_gb}GB")
            return False, peak
        
        # Leak detection
        if len(self.history) >= 2:
            old_step, old_peak = self.history[-min(self.window, len(self.history))]
            if (peak - old_peak) * 1024 > self.leak_threshold_mb:
                print(f"WARNING: VRAM leak detected: {old_peak:.2f}GB -> {peak:.2f}GB "
                      f"over {step - old_step} steps")
                return False, peak
        
        return True, peak
    
    @staticmethod
    def cleanup():
        """Aggressive cleanup between GRPO steps."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### Prototype Lambda Sweep Runner

```python
# Source: Project pattern from phase2_hp_sweep.py adapted for GRPO

PROTOTYPE_LAMBDAS = [0.0, 0.3, 0.7]

def run_prototype_sweep(config):
    """Run GRPO with 3 lambda values on small subset to validate pipeline."""
    results = []
    
    for lambda_ in PROTOTYPE_LAMBDAS:
        print(f"\n{'='*60}")
        print(f"GRPO Prototype: lambda={lambda_}")
        print(f"{'='*60}")
        
        output_dir = f"outputs/grpo-prototype/lambda_{lambda_}"
        
        # Train GRPO with this lambda
        metrics = train_grpo(
            lambda_=lambda_,
            G=config.G,  # Start with 2
            lr=1e-6,     # Per Amazon ASRU 2025 paper
            beta=0.04,   # KL coefficient
            epsilon=0.2, # Clip range
            temperature=0.8,
            max_steps=config.max_steps,  # 200 for prototype
            subset_size=config.subset_size,  # 2000 for prototype
            output_dir=output_dir,
        )
        
        # Evaluate checkpoint
        eval_results = evaluate_checkpoint(output_dir)
        results.append({
            "lambda": lambda_,
            "metrics": metrics,
            "eval": eval_results,
        })
    
    return results
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PPO with value model for ASR | GRPO (value-free, group-relative) | 2024-2025 | Eliminates critic model, halves VRAM for RL |
| Binary correct/incorrect reward | Continuous WER-based reward | 2025 (ASRU) | Critical for ASR where GRPO signal is weak |
| Standard GRPO with std normalization | Dr. GRPO: remove std, fixed normalization | 2025 (arxiv:2503.20783) | Eliminates difficulty bias, prevents length inflation |
| KL penalty mandatory | KL penalty optional (beta=0 common) | 2025 | Multiple studies show KL not essential for GRPO stability |
| Symmetric clipping [1-e, 1+e] | Asymmetric (DAPO): [1-e_low, 1+e_high] | 2025 | Prevents entropy collapse, allows more exploration |

**Key insight for ASR:** The Amazon ASRU 2025 paper (the only published GRPO-for-ASR work) used lr=1e-6, G=6-10, beta=0.04, and achieved up to 18.4% relative WER improvement. They used WER as the reward (not binary EM), which is critical for overcoming the unique-answer problem.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | OpenRLHF and Unsloth do not support audio/ASR models | Standard Stack (NOT Using) | Low -- would need to evaluate as alternative to custom loop |
| A2 | G=2 will fit within 15GB VRAM on T4 with batch_size=1 | Architecture Patterns | Medium -- if G=2 OOMs, must use QLoRA fallback |
| A3 | Qwen3-ASR `generate()` supports `do_sample=True, temperature` kwargs | Architecture Patterns | High -- if not, must implement custom sampling loop |
| A4 | `disable_adapter_layers()` in PEFT 0.18 correctly reverts to base weights | Pattern 2 | Medium -- if broken, must cache reference log-probs at initialization |
| A5 | 200 steps on 2K subset is sufficient for prototype validation | Code Examples | Low -- can extend if needed |
| A6 | Continuous WER reward provides enough advantage variance with G=4 | Pitfall 1 | High -- if variance still near-zero, GRPO may not work for this ASR task |

## Open Questions

1. **Does Qwen3-ASR's `model.generate()` support temperature sampling?**
   - What we know: The existing evaluation code uses `do_sample=False`. Phase 1 only tested greedy decoding.
   - What's unclear: Whether Qwen3-ASR-1.7B's generation supports `temperature`, `top_p`, and `return_dict_in_generate` with `output_scores=True`.
   - Recommendation: Test in prototype Step 1 before building the full GRPO loop. If not supported, implement custom autoregressive sampling.

2. **What is the actual VRAM budget for GRPO with G=2 on T4?**
   - What we know: SFT training peaks at ~10GB with batch_size=2. Generation adds KV cache overhead.
   - What's unclear: Exact VRAM for generation + forward pass + backward pass in the same step.
   - Recommendation: Profile a single GRPO step with `torch.cuda.memory_summary()` before training.

3. **Can per-batch fairness reward work with batch_size=1?**
   - What we know: Fairness reward requires multiple demographic groups in the computation window.
   - What's unclear: Whether accumulating across gradient accumulation steps (4 steps of batch_size=1) provides enough groups.
   - Recommendation: Use batch_size=2 minimum for GRPO, or implement rolling-window fairness computation across last N batches.

4. **What learning rate is appropriate for GRPO on a LoRA-adapted model?**
   - What we know: Amazon ASRU used lr=1e-6 for full-model GRPO. Phase 2 SFT used lr=3.5e-4 for LoRA.
   - What's unclear: LoRA GRPO likely needs a different lr than full-model GRPO or LoRA SFT.
   - Recommendation: Start with lr=1e-5 (between ASRU's 1e-6 and SFT's 3.5e-4), adjust based on reward curve stability.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| CUDA GPU | Training | Yes | Tesla T4 15GB | None -- required |
| torch | Core | Yes | 2.8.0+cu128 | -- |
| peft | LoRA | Yes | 0.18.1 | -- |
| transformers | Model | Yes | 4.57.6 | -- |
| jiwer | WER reward | Yes | 4.0.0 | -- |
| wandb | Tracking | Yes | 0.25.1 | Console logging |
| qwen-asr | Model loading | Yes | 0.0.6 | -- |
| bitsandbytes | QLoRA fallback | No | -- | Start with FP16; install only if OOM |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:**
- bitsandbytes: Only needed if G=2 with batch_size=1 still OOMs. Install `pip install bitsandbytes` and switch to 4-bit base model.

**Hardware note:** The actual GPU is a **Tesla T4 with 15GB VRAM** (not RTX A4000 16GB as stated in some requirements docs). This is 1GB less than originally assumed. VRAM budget should be set to 13GB (leaving 2GB headroom).

## Sources

### Primary (HIGH confidence)
- [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/grpo_trainer) - Confirmed no audio/ASR support, documented Dr. GRPO config (`scale_rewards=False`, `loss_type="dr_grpo"`)
- [TRL GitHub Issue #2807](https://github.com/huggingface/trl/issues/2807) - Confirmed data_collator not supported for multimodal GRPO
- [GRPO for Speech Recognition (Amazon, ASRU 2025)](https://arxiv.org/abs/2509.01939) - WER reward, G=6-10, lr=1e-6, beta=0.04, up to 18.4% relative WER improvement
- [Understanding R1-Zero Training (arxiv:2503.20783)](https://arxiv.org/abs/2503.20783) - Dr. GRPO removes std normalization to eliminate difficulty bias
- [GRPO-Zero implementation](https://github.com/policy-gradient/GRPO-Zero) - Reference implementation removing KL, per-token advantage
- pip show output for all installed packages - Version verification

### Secondary (MEDIUM confidence)
- [GRPO++ Tricks (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/grpo-tricks) - Practical training stability tips, DAPO variants, failure modes
- [FairGRPO (arxiv:2510.19893)](https://arxiv.org/abs/2510.19893) - Adaptive importance weighting for demographic fairness in GRPO
- Project codebase (train_standard_lora.py, data_loader.py, evaluate_adapter.py) - Existing patterns to reuse

### Tertiary (LOW confidence)
- Assumed claims about OpenRLHF and Unsloth audio support (not verified)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all packages verified installed, TRL non-viability confirmed
- Architecture: MEDIUM-HIGH - custom GRPO loop is validated approach per Amazon ASRU paper, but Qwen3-ASR-specific generation patterns need prototype validation
- Pitfalls: HIGH - well-documented in GRPO literature and project's own PITFALLS.md
- VRAM budget: MEDIUM - T4 15GB confirmed, but exact GRPO VRAM profile needs empirical measurement

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (30 days -- GRPO ecosystem is fast-moving but core algorithm is stable)
