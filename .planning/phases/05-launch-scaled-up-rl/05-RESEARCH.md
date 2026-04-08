# Phase 5: Launch the Scaled-Up RL - Research

**Researched:** 2026-04-08
**Domain:** GRPO reinforcement learning for fairness-aware ASR fine-tuning at full scale
**Confidence:** MEDIUM-HIGH

## Summary

Phase 5 takes the validated GRPO configuration from Phase 4 prototyping and runs full-scale training on the complete dataset (~45.3K utterances: 26,471 Fair-Speech + 18,876 Common Voice). The core task is executing sequential lambda-sweep GRPO training runs, each producing a fairness-aware LoRA adapter checkpoint, then evaluating all checkpoints to trace the accuracy-vs-fairness Pareto frontier. The hardware is a Tesla T4 with ~15 GB VRAM (not the originally-specified RTX A4000 with 16 GB), which is tighter but still feasible given Phase 2 profiling showed ~10 GB peak for single-adapter SFT with target_mlp=true.

The custom GRPO training loop (not TRL's GRPOTrainer) generates G candidate transcriptions per input, computes composite rewards R = (1-lambda)(1-WER) + lambda*(-|WER_g - WER_mean|), normalizes advantages group-relative, and applies a clipped surrogate policy gradient loss to LoRA parameters only. The critical scaling challenges are: (1) VRAM management with G=4 completions adding KV cache memory on top of the base model, (2) training stability over long runs (GRPO is known to collapse after ~1500 steps in some ASR settings), (3) reward hacking where the model achieves fairness by degrading all groups equally, and (4) lambda-sweep budgeting across 6-8 sequential training runs that each take many hours.

**Primary recommendation:** Run the lambda sweep sequentially (one lambda at a time) using the full dataset with the custom GRPO loop validated in Phase 4. Save checkpoints every 500 steps, select the best checkpoint per lambda by composite evaluation metric (weighted combination of mean WER and fairness gap), and budget approximately 1000-2000 training steps per lambda value based on GRPO-ASR literature showing convergence within this range.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GRPO-01 | Custom GRPO training loop generates G candidate outputs per input audio | Architecture pattern: custom loop with generate-then-score; VRAM budget analysis for G=4 on T4 |
| GRPO-02 | WER-based accuracy reward computed per candidate using jiwer | Reward design pattern using continuous WER (not binary); jiwer already installed |
| GRPO-03 | Fairness penalty computed per candidate over demographic groups | Group-level fairness reward aggregation pattern; rolling-window smoothing for stability |
| GRPO-04 | Composite reward R = R_acc + R_fair with configurable lambda | Lambda sweep strategy: 7-8 values with coarse-then-fine approach |
| GRPO-05 | Group-relative advantage normalization across G candidates | Dr. GRPO variant (no std normalization) recommended per research; scale_rewards=False |
| GRPO-06 | Clipped surrogate policy loss for stable updates | Asymmetric clipping (DAPO-style) recommended: epsilon_low=0.2, epsilon_high=0.28 |
| GRPO-07 | Dr. GRPO variant available | Remove std normalization from advantage, use fixed-constant token-level loss normalization |
| GRPO-08 | Stratified demographic batching | Existing DemographicStratifiedSampler in data_loader.py; extend for GRPO batch construction |
| GRPO-09 | Per-group WER tracked and logged every N steps | W&B logging pattern with per-group WER, entropy, reward stats; fast proxy every 50 steps |
| GRPO-10 | Adaptive lambda scheduling option | Fixed lambda per run (sweep strategy); adaptive scheduling deferred unless Phase 4 validates it |
| EXPT-01 | Lambda sweep traces Pareto frontier | 7-8 lambda values: {0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0} with possible fine-grained follow-up |
| EXPT-02 | Pareto frontier visualization | Scatter plot pattern: (mean_WER, fairness_metric) for all methods/lambdas |
| EXPT-03 | Statistical significance testing between GRPO and each baseline | Bootstrap pairwise testing using existing bootstrap CI infrastructure |
| EXPT-04 | Hyperparameter sensitivity analysis: ablation over G, learning rate, beta | Requires additional training runs beyond the lambda sweep; budget 3-5 extra runs |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **GPU Safety:** Must run `nvidia-smi` before any GPU task; abort if GPU memory > 80%
- **Documentation:** Log every code change immediately in `logs/` with proper fileClass frontmatter
- **Architecture:** Script-oriented pipeline in `scripts/training/`; manifest CSVs as data contracts
- **Existing pipeline is read-only:** Never modify scripts outside `scripts/training/`
- **Hardware:** Tesla T4 with ~15 GB VRAM (not RTX A4000 as originally planned -- verified via nvidia-smi)

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.8.0+cu128 | Custom GRPO training loop, autograd, GPU compute | Already installed; custom loop needs raw autograd control [VERIFIED: pip show] |
| PEFT | 0.18.1 | LoRA adapter injection, save/load/merge, reference policy toggle | Already installed; `disable_adapter_layers()` for zero-copy ref policy [VERIFIED: pip show] |
| transformers | 4.57.6 | Model loading, tokenizer, from_pretrained ecosystem | Already installed [VERIFIED: pip show] |
| jiwer | 4.0.0 | Per-utterance WER computation for reward function | Already installed; used throughout project [VERIFIED: pip show] |
| wandb | 0.25.1 | Experiment tracking, lambda sweep comparison, training curves | Already installed; standard for RL training monitoring [VERIFIED: pip show] |
| accelerate | 1.12.0 | Mixed-precision context managers, gradient accumulation helpers | Already installed [VERIFIED: pip show] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| qwen-asr | 0.0.6 | Load base Qwen3-ASR model and processor | Model loading (same pattern as existing scripts) [VERIFIED: pip show] |
| optuna | 4.8.0 | Only if adaptive lambda selection needed | Deferred -- manual lambda sweep is sufficient per REQUIREMENTS.md |
| bitsandbytes | NOT INSTALLED | QLoRA fallback if FP16 LoRA hits OOM | Install only if OOM with G=4; `pip install bitsandbytes` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom GRPO loop | TRL GRPOTrainer | TRL expects text-only input; Qwen3-ASR is audio-in model requiring custom generation pipeline. TRL now supports VLMs but NOT audio models. Custom loop is ~200 lines and fully controllable. [VERIFIED: TRL docs confirm no audio model support] |
| Manual lambda sweep | Optuna/Ray Tune | REQUIREMENTS.md explicitly marks automated HPO as out of scope. Sequential manual sweep is sufficient. |
| bitsandbytes QLoRA | FP16 LoRA | Phase 2 showed ~10 GB peak for single-adapter SFT. GRPO adds KV cache for G completions but should fit in 15 GB T4. QLoRA only if proven necessary. |

**Installation:**
```bash
# All required packages already installed. No new dependencies for Phase 5.
# If OOM fallback needed:
pip install bitsandbytes>=0.45.0
```

## Architecture Patterns

### Recommended Project Structure

```
scripts/training/
    grpo_trainer.py          # Custom GRPO training loop (Phase 4 creates, Phase 5 uses)
    reward.py                # Composite reward: WER accuracy + fairness penalty
    grpo_sweep.py            # Lambda sweep orchestrator: iterates lambda values, calls grpo_trainer
    grpo_evaluate.py         # Post-training evaluation: load adapter, run eval bridge, compute Pareto
outputs/
    grpo/
        lambda_0.00/         # Adapter + training logs for lambda=0.0
            adapter/         # LoRA adapter weights
            training_log.json
            checkpoints/     # step-500/, step-1000/, etc.
        lambda_0.10/
        lambda_0.20/
        ...
        pareto/              # Pareto frontier analysis
            pareto_results.json
            pareto_frontier.png
```

### Pattern 1: Custom GRPO Training Loop

**What:** A self-contained training loop that handles generation, reward computation, advantage normalization, and policy gradient updates -- all without TRL or any RL framework.

**When to use:** Always for this project. TRL GRPOTrainer does not support audio-in models.

**Example:**
```python
# Source: GRPO-ASR paper (arxiv:2509.01939) + project STACK.md
# One training step of the custom GRPO loop

def grpo_step(model, processor, batch, ref_logprobs_fn, config):
    """Execute one GRPO training step.
    
    Args:
        model: Qwen3-ASR with LoRA adapters (policy)
        processor: Qwen3ASRProcessor
        batch: Dict with audio, references, demographics
        ref_logprobs_fn: Function to compute reference log-probs (disable_adapter_layers)
        config: GRPOConfig with lambda_, G, epsilon_low, epsilon_high, beta, max_new_tokens
    """
    audio_list = batch["audio"]
    references = batch["references"]
    demographics = batch["demographics"]
    
    # 1. Generate G completions per audio sample (no grad)
    model.eval()
    completions_per_sample = []
    with torch.no_grad():
        for audio in audio_list:
            completions = []
            for _ in range(config.G):
                # Use model.generate() with temperature > 0 for diversity
                output = model.generate(
                    **prepare_input(processor, audio),
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=config.temperature,  # 0.7-1.0
                    top_p=1.0,
                )
                completions.append(output)
            completions_per_sample.append(completions)
    
    # 2. Compute rewards per completion
    rewards = compute_composite_rewards(
        completions_per_sample, references, demographics,
        lambda_=config.lambda_
    )  # shape: (batch_size, G)
    
    # 3. Normalize advantages (Dr. GRPO: no std division)
    advantages = rewards - rewards.mean(dim=-1, keepdim=True)
    # If using standard GRPO: advantages /= (rewards.std(dim=-1, keepdim=True) + 1e-8)
    
    # 4. Compute policy log-probs for each completion (with grad)
    model.train()
    policy_logprobs = compute_completion_logprobs(model, processor, audio_list, completions_per_sample)
    
    # 5. Compute reference log-probs (disable adapters)
    with torch.no_grad():
        ref_logprobs = ref_logprobs_fn(model, processor, audio_list, completions_per_sample)
    
    # 6. Clipped surrogate loss (DAPO asymmetric clipping)
    ratio = (policy_logprobs - ref_logprobs).exp()
    clipped = torch.clamp(ratio, 1 - config.epsilon_low, 1 + config.epsilon_high)
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    
    # 7. Optional KL penalty
    if config.beta > 0:
        kl = (policy_logprobs - ref_logprobs).mean()
        loss += config.beta * kl
    
    return loss, {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "advantage_std": advantages.std().item(),
        "kl": (policy_logprobs - ref_logprobs).mean().item(),
        "loss": loss.item(),
    }
```

### Pattern 2: Reference Policy via PEFT Adapter Toggle

**What:** Use `disable_adapter_layers()` / `enable_adapter_layers()` to switch between policy and reference model without loading a second copy.

**When to use:** Always for KL computation. Saves ~3.4 GB VRAM.

**Example:**
```python
# Source: PEFT documentation + project STACK.md
def compute_ref_logprobs(model, processor, audio_list, completions):
    """Compute log-probs under the reference (base) model."""
    model.disable_adapter_layers()  # temporarily remove LoRA
    with torch.no_grad():
        ref_logprobs = compute_completion_logprobs(model, processor, audio_list, completions)
    model.enable_adapter_layers()  # restore LoRA
    return ref_logprobs
```

### Pattern 3: Lambda Sweep Orchestrator

**What:** Sequential training runs across lambda values, reusing the same base model but training fresh LoRA adapters for each lambda.

**When to use:** Phase 5 main workflow.

**Example:**
```python
# Source: Project architecture pattern
LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

def run_lambda_sweep(base_config, lambda_values):
    """Run GRPO training for each lambda value sequentially."""
    results = []
    for lam in lambda_values:
        print(f"\n{'='*60}")
        print(f"Lambda = {lam}")
        print(f"{'='*60}")
        
        config = base_config.copy()
        config["lambda_"] = lam
        config["output_dir"] = f"outputs/grpo/lambda_{lam:.2f}"
        
        # Load fresh model + apply LoRA (start from scratch for each lambda)
        model, processor = load_model_and_processor()
        model = apply_lora(model, **locked_lora_config)
        
        # Train
        trainer = GRPOTrainer(model, processor, config)
        result = trainer.train()
        
        # Evaluate best checkpoint
        eval_result = evaluate_adapter(
            adapter_path=result["best_checkpoint_path"],
            model_name=f"grpo-lambda-{lam:.2f}",
        )
        results.append({"lambda": lam, **eval_result})
        
        # Free GPU memory before next run
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()
    
    return results
```

### Pattern 4: Checkpoint Strategy for Long RL Runs

**What:** Save adapter checkpoints at regular intervals; select best checkpoint by composite metric (not final step).

**When to use:** Every GRPO training run.

**Example:**
```python
# Source: Common RL training best practice + GRPO ASR paper conventions
CHECKPOINT_EVERY = 500  # steps
EVAL_EVERY = 250       # steps (fast proxy: per-group WER gap)
FULL_EVAL_EVERY = 500  # steps (full fairness metrics on eval split)

def select_best_checkpoint(checkpoints, eval_results):
    """Select best checkpoint by composite metric.
    
    Metric: alpha * (1 - mean_WER) + (1 - alpha) * (1 - fairness_gap)
    where alpha balances accuracy vs fairness in selection.
    """
    alpha = 0.5  # equal weight to accuracy and fairness
    scores = []
    for ckpt, result in zip(checkpoints, eval_results):
        score = alpha * (1 - result["mean_wer"]) + (1 - alpha) * (1 - result["max_min_gap"])
        scores.append(score)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return checkpoints[best_idx]
```

### Pattern 5: VRAM Monitoring for RL Training

**What:** Continuous VRAM tracking to detect memory leaks before OOM.

**When to use:** Every GRPO training run, checked every 50 steps.

**Example:**
```python
# Source: PITFALLS.md Pitfall #1
class VRAMMonitor:
    def __init__(self, budget_gb=13.0, leak_threshold_mb=500, window=200):
        self.budget_gb = budget_gb
        self.leak_threshold_mb = leak_threshold_mb
        self.window = window
        self.history = []
    
    def check(self, step):
        peak = torch.cuda.max_memory_allocated() / 1024**3
        self.history.append((step, peak))
        
        # Check absolute budget
        if peak > self.budget_gb:
            raise RuntimeError(f"Step {step}: VRAM {peak:.2f}GB exceeds budget {self.budget_gb}GB")
        
        # Check for leak: compare current window to previous
        if len(self.history) >= self.window:
            early = [h[1] for h in self.history[-self.window:-self.window//2]]
            late = [h[1] for h in self.history[-self.window//2:]]
            growth_mb = (sum(late)/len(late) - sum(early)/len(early)) * 1024
            if growth_mb > self.leak_threshold_mb:
                raise RuntimeError(
                    f"Step {step}: VRAM leak detected! Growth: {growth_mb:.0f}MB "
                    f"over {self.window} steps"
                )
        
        return peak
```

### Anti-Patterns to Avoid

- **Loading two copies of the model for reference policy:** Use PEFT `disable_adapter_layers()` instead. Saves ~3.4 GB.
- **Computing fairness reward per-sample instead of per-group:** The fairness penalty |WER_g - WER_mean| requires group-level aggregation across the batch or a rolling window. Per-sample computation yields meaningless signal.
- **Using low temperature (< 0.5) for GRPO generation:** Low temperature makes all G completions identical, producing zero advantage variance and no learning signal. Use temperature 0.7-1.0.
- **Skipping GC between GRPO steps:** GRPO generates intermediate tensors (KV cache, completion tokens, log-probs) that must be freed each step. Always call `torch.cuda.empty_cache(); gc.collect()` after each step.
- **Running all lambda values in parallel:** T4 has 15 GB VRAM -- only one training run at a time. Lambda sweep must be sequential.
- **Using TRL GRPOTrainer:** Does not support audio-in models. Do not attempt.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LoRA adapter management | Custom LoRA injection | PEFT `get_peft_model()`, `PeftModel.from_pretrained()` | Save/load/merge lifecycle, adapter toggling, all handled |
| Reference policy KL computation | Second model copy on CPU | PEFT `disable_adapter_layers()` / `enable_adapter_layers()` | Zero-copy reference, saves 3.4 GB VRAM |
| WER computation | Custom edit distance | `jiwer.wer()` | Handles tokenization, normalization edge cases |
| Text normalization | Custom regex | `whisper_normalizer.english.EnglishTextNormalizer` | Must match evaluation pipeline exactly |
| Experiment tracking | CSV-only logging | `wandb` with per-lambda-value runs grouped under a sweep | Comparison across lambda values, training curve overlays |
| VRAM profiling | Manual nvidia-smi | `torch.cuda.max_memory_allocated()` in Python + structured logging | Integrated with training loop, programmatic abort on leak |
| Checkpoint saving | Manual torch.save | PEFT `model.save_pretrained()` + `torch.save()` for optimizer state | Consistent adapter format, resume-friendly |
| Demographic-stratified batching | Custom sampler from scratch | Existing `DemographicStratifiedSampler` in `data_loader.py` | Already validated in Phase 2 |

**Key insight:** The GRPO algorithm itself (~150-200 lines) is the only thing that needs to be custom. Everything around it (model management, data loading, evaluation, experiment tracking) uses existing project infrastructure or established libraries.

## Common Pitfalls

### Pitfall 1: GRPO Training Collapse After Extended Steps

**What goes wrong:** GRPO training reward rises initially then crashes, accompanied by policy entropy collapse and accuracy degradation. Published research reports instability after ~1500 steps for ASR GRPO. [CITED: arxiv.org/abs/2509.18569]

**Why it happens:** The "Lazy Likelihood Displacement Death Spiral" -- even as rewards increase, the likelihood of correct responses enters monotonic decline, inflating gradients and causing collapse. [CITED: arxiv.org/abs/2512.04220]

**How to avoid:**
1. Cap training at 1000-2000 steps per lambda value (GRPO-ASR paper used max 5000 steps with lr=1e-6; our LoRA setup converges faster)
2. Monitor entropy every 50 steps -- if it drops below 50% of initial value, training is collapsing
3. Use early stopping: if eval WER increases for 3 consecutive evaluations, stop
4. Consider momentum-anchored approach (M-GRPO) if collapse occurs: exponential moving average of policy parameters with momentum=0.99 [CITED: arxiv.org/abs/2512.13070]

**Warning signs:** Entropy monotonically declining; all G completions becoming identical; reward suddenly crashing after initial increase.

### Pitfall 2: Fairness Reward Hacking via Accuracy Collapse

**What goes wrong:** Model maximizes fairness reward by making ALL groups equally bad (empty transcriptions, repeated tokens). [ASSUMED -- well-documented in multi-objective RL literature]

**Why it happens:** The fairness term rewards equality of WER, not low WER. Path of least resistance is leveling down.

**How to avoid:**
1. Log per-group absolute WER alongside composite reward at every eval step
2. Add WER floor constraint: if mean_WER > baseline_WER * 1.2, override fairness bonus to zero
3. Validate on held-out data after each checkpoint -- if WER degrades > 5% relative, that checkpoint is degenerate
4. Lambda=0.0 run serves as control: its WER should match SFT baseline closely

**Warning signs:** Composite reward increasing while absolute WER also increasing. Per-group WER converging toward the worst group's level.

### Pitfall 3: VRAM Exhaustion from KV Cache During Generation

**What goes wrong:** GRPO generates G completions per audio sample. Each completion requires a KV cache during autoregressive decoding. With G=4 and batch_size=2, this means 8 concurrent KV caches on top of the model weights and LoRA adapters. [VERIFIED: Phase 2 peak VRAM was ~10 GB for SFT; GRPO adds generation overhead]

**Why it happens:** Unlike SFT where forward pass is teacher-forced (no KV cache accumulation), GRPO generation accumulates KV cache tokens autoregressively up to max_new_tokens.

**How to avoid:**
1. Generate completions one at a time (not batched) to minimize peak KV cache: loop over G completions sequentially per audio sample
2. Set max_new_tokens=200 (ASR transcriptions are short; 200 tokens covers virtually all utterances)
3. Profile VRAM during first 10 steps before committing to full run
4. Start with G=2, only increase to G=4 if VRAM permits
5. Clear KV cache explicitly after generation: `torch.cuda.empty_cache()` between generation and training phases

**Warning signs:** OOM on first generation step; VRAM > 13 GB during generation phase.

### Pitfall 4: Unique-Answer Problem Starving Learning Signal

**What goes wrong:** ASR has exactly one correct answer per utterance. When all G completions get similar WER, advantages are near-zero, producing no learning signal. [CITED: arxiv.org/abs/2509.01939 -- "preliminary experiments with ASR obtained only minor improvements"]

**Why it happens:** GRPO advantage = (reward_i - mean_reward) / std_reward. If all completions are similarly good or bad, std is near-zero, advantage is near-zero.

**How to avoid:**
1. Use continuous WER reward (1 - WER), not binary correct/incorrect
2. Sampling temperature 0.7-1.0 to create diversity in completions
3. Monitor `frac_reward_zero_std` -- fraction of samples where all G completions get identical reward. If > 50%, learning signal is insufficient
4. Mix difficulty levels in each batch (easy + hard utterances)
5. Consider Dr. GRPO (remove std normalization) to prevent zero-std from zeroing out advantages entirely

**Warning signs:** Advantage std near zero; WER improvement < 0.5% after 500 steps.

### Pitfall 5: Lambda Sweep Produces Degenerate Pareto Frontier

**What goes wrong:** Most lambda values produce either negligible fairness change or catastrophic accuracy loss, with only 2-3 meaningful points on the frontier. [ASSUMED -- common in multi-objective optimization]

**Why it happens:** The fairness-accuracy tradeoff gradient is steep near the transition zone. Linear lambda spacing misses the interesting region.

**How to avoid:**
1. Use non-linear lambda spacing: {0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0}
2. Run coarse sweep first (0.0, 0.2, 0.5, 0.8, 1.0), identify transition zone, then add fine-grained values
3. Need minimum 5-6 non-dominated points for a convincing Pareto plot
4. Plot intermediate checkpoints -- a single training run traces part of the frontier as training progresses

**Warning signs:** Fewer than 4 distinct points on the frontier; most lambdas clustering at extremes.

### Pitfall 6: Demographic Group Imbalance Corrupts Fairness Signal

**What goes wrong:** Small demographic groups (< 50 utterances per batch) produce noisy per-group WER estimates, creating noisy fairness rewards that destabilize training. [CITED: PITFALLS.md Pitfall #7]

**Why it happens:** Fair-Speech and Common Voice have highly imbalanced demographic distributions. Without stratified sampling, some groups may have 0-2 samples per batch.

**How to avoid:**
1. Use existing `DemographicStratifiedSampler` to ensure minimum N samples per group per batch
2. Accumulate fairness signal over a rolling window of K batches (e.g., K=10) rather than per-batch
3. Weight fairness term by sqrt(n_g) to downweight estimates from small groups

**Warning signs:** High variance in fairness reward between consecutive batches (> 2 std from mean).

## Code Examples

### Composite Reward Computation

```python
# Source: Project ARCHITECTURE.md + REQUIREMENTS.md GRPO-02/03/04
import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()

def compute_composite_rewards(completions_per_sample, references, demographics, lambda_):
    """Compute composite GRPO reward for a batch.
    
    Args:
        completions_per_sample: List[List[str]] -- G completions per sample
        references: List[str] -- reference transcripts
        demographics: List[str] -- demographic group labels
        lambda_: float -- fairness weight (0=accuracy only, 1=fairness only)
    
    Returns:
        rewards: torch.Tensor of shape (batch_size, G)
    """
    batch_size = len(references)
    G = len(completions_per_sample[0])
    rewards = torch.zeros(batch_size, G)
    
    for i in range(batch_size):
        ref = normalizer(references[i])
        for g in range(G):
            hyp = normalizer(completions_per_sample[i][g])
            if ref and hyp:
                wer_val = jiwer.wer(ref, hyp)
            else:
                wer_val = 1.0 if ref else 0.0
            
            # Accuracy reward (per-completion)
            r_acc = (1 - lambda_) * (1 - wer_val)
            rewards[i, g] = r_acc  # fairness added below
    
    # Fairness penalty (per-group, applied to all completions in that group)
    # Computed over the entire batch, not per-completion
    unique_groups = list(set(d for d in demographics if d))
    if len(unique_groups) >= 2 and lambda_ > 0:
        for g_idx in range(G):
            # Compute per-group mean WER for this completion index
            group_wers = {}
            for i in range(batch_size):
                grp = demographics[i]
                if grp:
                    if grp not in group_wers:
                        group_wers[grp] = []
                    ref = normalizer(references[i])
                    hyp = normalizer(completions_per_sample[i][g_idx])
                    wer_val = jiwer.wer(ref, hyp) if (ref and hyp) else 1.0
                    group_wers[grp].append(wer_val)
            
            overall_wer = sum(sum(v) for v in group_wers.values()) / sum(len(v) for v in group_wers.values())
            
            for i in range(batch_size):
                grp = demographics[i]
                if grp and grp in group_wers:
                    group_mean = sum(group_wers[grp]) / len(group_wers[grp])
                    fairness_penalty = -abs(group_mean - overall_wer)
                    rewards[i, g_idx] += lambda_ * fairness_penalty
    
    return rewards
```

### W&B Logging for GRPO Training

```python
# Source: TRL logging documentation + wandb best practices for RL
import wandb

def init_wandb_grpo(lambda_val, config):
    """Initialize W&B run for a single lambda value in the sweep."""
    wandb.init(
        project="grpo-fairness-asr",
        name=f"grpo-lambda-{lambda_val:.2f}",
        group="lambda-sweep",
        config={
            "lambda": lambda_val,
            "G": config["G"],
            "lr": config["lr"],
            "epsilon_low": config["epsilon_low"],
            "epsilon_high": config["epsilon_high"],
            "beta": config["beta"],
            "max_steps": config["max_steps"],
            "temperature": config["temperature"],
        },
        tags=["phase-5", "grpo", "fairness"],
    )

def log_grpo_step(step, metrics, per_group_wer=None):
    """Log one training step to W&B."""
    log_dict = {
        "step": step,
        "reward/mean": metrics["reward_mean"],
        "reward/std": metrics["reward_std"],
        "advantage/std": metrics["advantage_std"],
        "policy/kl": metrics["kl"],
        "policy/loss": metrics["loss"],
        "policy/entropy": metrics.get("entropy", None),
        "vram/peak_gb": metrics.get("vram_peak_gb", None),
        "training/frac_zero_std": metrics.get("frac_zero_std", None),
    }
    
    # Per-group WER tracking (GRPO-09)
    if per_group_wer:
        for group, wer in per_group_wer.items():
            log_dict[f"wer/{group}"] = wer
        log_dict["fairness/max_min_gap"] = max(per_group_wer.values()) - min(per_group_wer.values())
        log_dict["fairness/wer_std"] = torch.tensor(list(per_group_wer.values())).std().item()
    
    wandb.log(log_dict, step=step)
```

### Dr. GRPO Advantage Normalization

```python
# Source: cameronrwolfe.substack.com/p/grpo-tricks (Dr. GRPO)
def compute_advantages_dr_grpo(rewards):
    """Dr. GRPO: remove std normalization, use mean-only baseline.
    
    Standard GRPO: A_i = (R_i - mean(R)) / std(R)
    Dr. GRPO:      A_i = R_i - mean(R)
    
    This prevents the difficulty bias where easy/hard batches get
    artificially inflated advantages. For ASR where reward variance
    is naturally low (unique-answer problem), this avoids division
    by near-zero std.
    """
    mean_reward = rewards.mean(dim=-1, keepdim=True)
    advantages = rewards - mean_reward
    return advantages
```

### Token-Level Loss with Fixed Constant Normalization

```python
# Source: cameronrwolfe.substack.com/p/grpo-tricks (Dr. GRPO + DAPO)
def compute_grpo_loss(policy_logprobs, ref_logprobs, advantages, 
                       epsilon_low=0.2, epsilon_high=0.28,
                       max_tokens=200):
    """GRPO loss with DAPO asymmetric clipping and Dr. GRPO normalization.
    
    Args:
        policy_logprobs: (batch, G, seq_len) log-probs under current policy
        ref_logprobs: (batch, G, seq_len) log-probs under reference
        advantages: (batch, G) advantages per completion
        epsilon_low: lower clip bound (standard)
        epsilon_high: upper clip bound (DAPO: higher for exploration)
        max_tokens: fixed constant for Dr. GRPO normalization
    """
    # Token-level importance ratios
    ratio = (policy_logprobs - ref_logprobs).exp()
    
    # Expand advantages to token level
    advantages_expanded = advantages.unsqueeze(-1)  # (batch, G, 1)
    
    # DAPO asymmetric clipping
    clipped = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)
    
    # PPO-style clipped surrogate
    surr1 = ratio * advantages_expanded
    surr2 = clipped * advantages_expanded
    token_loss = -torch.min(surr1, surr2)
    
    # Dr. GRPO: normalize by fixed constant, not sequence length
    # This prevents shorter sequences from getting disproportionate gradients
    loss = token_loss.sum() / max_tokens
    
    return loss
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| GRPO with std normalization | Dr. GRPO: mean-only advantage | 2025 (Liu et al.) | Removes length bias and difficulty bias [CITED: arxiv Dr. GRPO paper] |
| Symmetric PPO clipping (eps=0.2) | DAPO asymmetric clipping (eps_low=0.2, eps_high=0.28) | 2025 (DAPO paper) | Prevents entropy collapse by allowing more exploration [CITED: DAPO paper] |
| beta=0.04 for KL penalty | beta=0.0 (no KL) as default | 2025 (multiple studies) | KL penalty not essential for GRPO training; simplifies implementation [CITED: TRL docs] |
| TRL GRPOTrainer for everything | Custom loop for non-text models | Current | TRL now supports VLMs but still not audio models [VERIFIED: TRL docs 2026] |
| Sequence-level loss normalization | Token-level loss (DAPO) | 2025 | Prevents short responses from dominating gradients [CITED: cameronrwolfe substack] |
| GRPO for ASR (basic) | GRPO + DiffRO combination (FunRL) | 2025 | 5.3% relative WER improvement; GRPO alone unstable after ~1500 steps [CITED: arxiv 2509.18569] |

**Deprecated/outdated:**
- **beta=0.04 as standard KL coefficient**: Multiple 2025 studies show KL term unnecessary for GRPO. Default is now beta=0.0 in TRL. However, for fairness-constrained training where policy drift is a concern, a small beta (0.01-0.04) may still be beneficial to prevent mode collapse. [CITED: TRL docs, Open-Reasoner-Zero]
- **num_generations=16 or higher**: Memory-prohibitive on single GPU. G=4-8 is the practical range. GRPO-ASR paper found impact of G "insignificant" between 6 and 10. [CITED: arxiv 2509.01939]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | GRPO training will converge in 1000-2000 steps per lambda value | Architecture Patterns / Pattern 3 | If convergence requires more steps, lambda sweep takes significantly longer; may need to reduce number of lambda values |
| A2 | G=4 completions fit in T4 15GB VRAM with BF16 LoRA | Architecture Patterns / VRAM | If G=4 causes OOM, must reduce to G=2 (less diverse advantage estimates) or use QLoRA |
| A3 | Temperature 0.7-1.0 produces sufficient diversity in ASR completions | Pitfall 4 prevention | If diversity is still low, advantage variance will be near-zero; may need to try temperature > 1.0 |
| A4 | Rolling-window fairness reward (K=10 batches) is sufficient to smooth group imbalance noise | Pitfall 6 prevention | If fairness signal is still noisy, may need per-epoch fairness computation instead of per-batch |
| A5 | Phase 4 will validate GRPO config that works at prototype scale | Summary | If Phase 4 fails to find a working config, Phase 5 cannot proceed as described |
| A6 | Dr. GRPO (no std normalization) will outperform standard GRPO for ASR fairness | Code Examples | If standard GRPO works better, need to switch; low risk since Phase 4 should test both |
| A7 | Lambda sweep needs 7-8 values for a convincing Pareto frontier | Pattern 3 | If fewer values suffice (transition zone is narrow), can save compute; if more needed, budget increases |
| A8 | Reward hacking via accuracy collapse is the primary failure mode (not other forms) | Pitfall 2 | Other failure modes (entropy collapse, gradient explosion) may dominate; monitoring should cover all |

## Open Questions

1. **What specific GRPO hyperparameters will Phase 4 validate?**
   - What we know: Phase 4 will prototype GRPO at small scale (subset of data, fewer steps)
   - What's unclear: Exact values for G, temperature, lr, beta, epsilon, max_steps that Phase 4 will lock
   - Recommendation: Phase 5 should accept a `grpo_locked_config.json` from Phase 4, analogous to `locked_config.json` from Phase 2

2. **Will Phase 4's LoRA config match Phase 3's locked config or use different settings?**
   - What we know: Phase 3 trains SFT LoRA with rank=4, alpha=4, target_mlp=true, lr=3.47e-4
   - What's unclear: Whether GRPO LoRA config should reuse these exact settings or Phase 4 might find different optimal LoRA params for RL training
   - Recommendation: Start with Phase 3's locked LoRA config; Phase 4 can override if needed

3. **How many total GPU-hours does the lambda sweep require?**
   - What we know: Phase 2 SFT took ~1768 sec for 500 steps on 2K samples. Full dataset is ~45K samples. GRPO is slower per step (G forward passes for generation).
   - Estimated per-lambda: 1000 steps * (2x SFT step time for generation overhead) * (45K/2K dataset ratio factor for data loading, but step count is fixed) = roughly 4-8 hours per lambda
   - For 8 lambdas: 32-64 GPU-hours total
   - Recommendation: Budget for 3-5 day wall clock time; monitor first lambda run closely to calibrate estimate

4. **Should lambda=0.0 use GRPO or SFT?**
   - What we know: GRPO with lambda=0.0 is purely accuracy-reward RL, which should approximate SFT but with more overhead
   - What's unclear: Whether there's meaningful difference vs. using Phase 3's SFT baseline directly
   - Recommendation: Run GRPO lambda=0.0 as a sanity check (should match SFT baseline); if identical, can use SFT baseline and skip this run

5. **Dr. GRPO vs standard GRPO for fairness-aware ASR?**
   - What we know: Dr. GRPO removes std normalization (good for unique-answer problem); DAPO adds asymmetric clipping (good for entropy preservation)
   - What's unclear: Whether these modifications help or hurt when combined with a fairness reward that already has low variance
   - Recommendation: Phase 4 should test both variants; Phase 5 uses whichever Phase 4 validates

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| CUDA GPU (T4) | All training | Yes | compute 7.5 | None -- GPU required |
| PyTorch | Training loop | Yes | 2.8.0+cu128 | -- |
| PEFT | LoRA management | Yes | 0.18.1 | -- |
| transformers | Model loading | Yes | 4.57.6 | -- |
| jiwer | WER computation | Yes | 4.0.0 | -- |
| wandb | Experiment tracking | Yes | 0.25.1 | CSV logging fallback |
| qwen-asr | Model loading | Yes | 0.0.6 | -- |
| bitsandbytes | QLoRA fallback | No | -- | FP16 LoRA (primary); install if OOM |
| TRL | NOT USED | No | -- | Custom loop (by design) |

**Missing dependencies with no fallback:**
- None -- all required packages are installed

**Missing dependencies with fallback:**
- bitsandbytes: Not installed, but only needed if FP16 LoRA causes OOM with G=4. Fallback is reducing G to 2.

## Security Domain

> security_enforcement: Not applicable for this research/academic project. No user-facing services, authentication, or sensitive data handling beyond standard GPU access.

## Sources

### Primary (HIGH confidence)
- [TRL GRPO Trainer documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer) -- GRPOConfig defaults, logging metrics, custom reward function API, loss types, beta=0.0 default
- [GRPO for Speech Recognition (ASRU 2025)](https://arxiv.org/abs/2509.01939) -- GRPO-ASR training: lr=1e-6, batch=64, max_steps=5000, G={6,10}, WER/EM/ED rewards
- [GRPO++ Tricks (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/grpo-tricks) -- Dr. GRPO, DAPO improvements, entropy collapse prevention, token-level loss
- [Project research/STACK.md](../../research/STACK.md) -- Custom GRPO loop decision, VRAM budget analysis
- [Project research/PITFALLS.md](../../research/PITFALLS.md) -- 14 pitfalls with mitigations
- [Project research/ARCHITECTURE.md](../../research/ARCHITECTURE.md) -- Component boundaries, data flow, GRPO pseudocode
- pip show outputs on installed packages -- verified versions of all core dependencies

### Secondary (MEDIUM confidence)
- [RL for LLM-based ASR and TTS (FunRL)](https://arxiv.org/abs/2509.18569) -- GRPO instability after ~1500 steps in ASR, G=8, temperature=1.0, 5.3% relative WER improvement
- [M-GRPO: Stabilizing Self-Supervised RL](https://arxiv.org/abs/2512.13070) -- Momentum-anchored policy for GRPO stability, IQR-based filtering
- [On GRPO Collapse (LLD Death Spiral)](https://arxiv.org/abs/2512.04220) -- Lazy Likelihood Displacement as root cause of GRPO collapse
- [FairGRPO for Clinical Reasoning](https://arxiv.org/abs/2510.19893) -- Adaptive importance weighting for demographic fairness in GRPO, 27.2% reduction in predictive parity gap
- [Phase 3 CONTEXT.md](../03-full-sft-lora-training/03-CONTEXT.md) -- Scale-up patterns, locked config, VRAM estimates

### Tertiary (LOW confidence)
- GRPO-ASR instability threshold of "~1500 steps" -- referenced in FunRL paper abstract search results but exact details not extractable from PDF; needs validation in Phase 4
- VRAM estimates for GRPO generation phase on T4 -- calculated, not measured; Phase 4 must profile

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages already installed and verified; no new dependencies needed
- Architecture: MEDIUM-HIGH -- custom GRPO loop pattern well-documented in project research; specific generation/scoring implementation for Qwen3-ASR needs Phase 4 validation
- Pitfalls: HIGH -- extensively documented in project PITFALLS.md and 2025 GRPO literature; specific ASR stability issues confirmed by GRPO-ASR paper
- Lambda sweep strategy: MEDIUM -- coarse-then-fine approach is standard, but optimal lambda values unknown until Phase 4 calibrates reward scales

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (30 days -- stable domain; core libraries already pinned)
