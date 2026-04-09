"""
Custom GRPO training loop for fairness-aware ASR fine-tuning of Qwen3-ASR-1.7B.

TRL's GRPOTrainer does NOT support audio/ASR models (no audio data_collator,
incompatible with encoder-decoder architecture). This implements the GRPO
algorithm directly on PyTorch + PEFT, reusing existing model loading, LoRA
injection, and data infrastructure from Phases 1-3.

Algorithm: GRPO (DeepSeekMath) adapted for ASR (Amazon ASRU 2025):
  1. Generate G candidate transcriptions per audio input (with sampling)
  2. Compute composite reward: R = (1-lambda)(1-WER) + lambda(-|WER_g - WER_mean|)
  3. Normalize advantages: A = (R - mean(R)) / std(R)  [or Dr. GRPO: A = R - mean(R)]
  4. Compute clipped surrogate policy loss (PPO-style)
  5. Optional KL penalty against reference policy (via disable_adapter_layers)

Modes:
  prototype  — Small subset (~2K), G=2, 200 steps. Validates training signal.
  sweep      — 3 lambda values (0, 0.3, 0.7) on prototype scale.

Usage:
    # Prototype: validate GRPO training signal exists
    python scripts/training/train_grpo.py \
        --mode prototype \
        --fs_manifest outputs/manifests/fs_train.csv \
        --cv_manifest outputs/manifests/cv_train.csv \
        --output_dir outputs/grpo-prototype \
        --lambda_ 0.0 --G 2 --max_steps 200 --subset_size 2000

    # Lambda sweep: 3 values on prototype scale
    python scripts/training/train_grpo.py \
        --mode sweep \
        --fs_manifest outputs/manifests/fs_train.csv \
        --cv_manifest outputs/manifests/cv_train.csv \
        --output_dir outputs/grpo-prototype \
        --G 2 --max_steps 200 --subset_size 2000
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jiwer
import numpy as np
import pandas as pd
import torch
import wandb

from scripts.training.data_loader import ASRFairnessDataset, DemographicStratifiedSampler, collate_fn
from scripts.training.reward import RewardComputer
from scripts.training.train_standard_lora import (
    load_model_and_processor,
    patch_outer_forward,
    apply_lora,
    print_gpu_memory,
    create_stratified_subset,
    create_speaker_disjoint_split,
    MODEL_ID,
    SEED,
)

# -- Constants ----------------------------------------------------------------

PROTOTYPE_LAMBDAS = [0.0, 0.3, 0.7]
VRAM_BUDGET_GB = 13.0  # T4 15GB - 2GB headroom


# -- GRPO Config --------------------------------------------------------------

class GRPOConfig:
    """Configuration for GRPO training.

    Supports both prototype (small-scale) and full-scale training modes.
    DAPO asymmetric clipping: use epsilon_low != epsilon_high for asymmetric
    surrogate clipping (Pitfall #1: training collapse mitigation).
    """

    def __init__(
        self,
        G: int = 2,
        lambda_: float = 0.0,
        lr: float = 1e-5,
        beta: float = 0.04,
        epsilon: float = 0.2,
        epsilon_low: float | None = None,
        epsilon_high: float | None = None,
        temperature: float = 0.8,
        max_new_tokens: int = 128,
        max_steps: int = 200,
        gradient_accumulation_steps: int = 4,
        dr_grpo: bool = True,
        log_every: int = 10,
        eval_every: int = 50,
        save_every: int = 100,
        warmup_steps: int = 10,
        max_grad_norm: float = 1.0,
        baseline_wer: float | None = None,
        # -- Early stopping (Phase 5 full-scale) --------------------------------
        early_stop_patience: int = 3,
        early_stop_threshold: float = 0.05,
        # -- Checkpoint selection ------------------------------------------------
        checkpoint_select_alpha: float = 0.5,
    ):
        self.G = G
        self.lambda_ = lambda_
        self.lr = lr
        self.beta = beta
        # DAPO asymmetric clipping: epsilon_low/high override symmetric epsilon
        self.epsilon_low = epsilon_low if epsilon_low is not None else epsilon
        self.epsilon_high = epsilon_high if epsilon_high is not None else epsilon
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.dr_grpo = dr_grpo
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.baseline_wer = baseline_wer
        # Early stopping: stop if eval WER degrades for patience consecutive evals
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold  # relative WER degradation
        # Checkpoint selection: alpha * (1 - mean_WER) + (1 - alpha) * (1 - fairness_gap)
        self.checkpoint_select_alpha = checkpoint_select_alpha

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# -- VRAM Monitor --------------------------------------------------------------

class VRAMMonitor:
    """Track VRAM usage, abort if leak detected."""

    def __init__(self, budget_gb: float = VRAM_BUDGET_GB, leak_threshold_mb: float = 500, window: int = 200):
        self.budget_gb = budget_gb
        self.leak_threshold_mb = leak_threshold_mb
        self.window = window
        self.history: list[tuple[int, float]] = []

    def check(self, step: int) -> tuple[bool, float]:
        if not torch.cuda.is_available():
            return True, 0.0
        peak = torch.cuda.max_memory_allocated() / 1024**3
        self.history.append((step, peak))
        if peak > self.budget_gb:
            print(f"WARNING: Step {step}: Peak VRAM {peak:.2f}GB > budget {self.budget_gb}GB")
            return False, peak
        if len(self.history) >= 2:
            old_step, old_peak = self.history[-min(self.window, len(self.history))]
            if (peak - old_peak) * 1024 > self.leak_threshold_mb:
                print(f"WARNING: VRAM leak: {old_peak:.2f}GB -> {peak:.2f}GB over {step - old_step} steps")
                return False, peak
        return True, peak

    @staticmethod
    def cleanup():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -- Per-Token Log-Probability Computation -------------------------------------

def compute_per_token_log_probs(model, processor, audio_np_list, generated_ids_list, device):
    """Compute per-token log-probs via teacher-forced forward pass.

    Uses the same forward path as training to ensure consistency between
    old_log_probs and current_log_probs (Pitfall #4).

    Args:
        model: PeftModel (or base model with adapters disabled).
        processor: Qwen3ASRProcessor.
        audio_np_list: List of numpy audio arrays (one per batch item).
        generated_ids_list: List of 1D tensors of generated token IDs.
        device: torch device.

    Returns:
        List of 1D tensors, each containing per-token log-probs for the
        generated portion of one sample.
    """
    all_log_probs = []

    for audio_np, gen_ids in zip(audio_np_list, generated_ids_list):
        # Build the same chat-format input used during generation
        conversation = [
            {"role": "user", "content": [{"type": "audio", "audio": audio_np}]}
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        text += "language English<asr_text>"

        # Process input
        inputs = processor(
            text=text, audio=[audio_np], return_tensors="pt", padding=False
        )
        model_dtype = next(model.parameters()).dtype
        inputs_gpu = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                if v.is_floating_point():
                    v = v.to(model_dtype)
            inputs_gpu[k] = v

        prompt_len = inputs_gpu["input_ids"].shape[-1]

        # Concatenate prompt + generated tokens
        gen_ids_device = gen_ids.to(device).unsqueeze(0)  # (1, gen_len)
        full_ids = torch.cat([inputs_gpu["input_ids"], gen_ids_device], dim=1)
        full_mask = torch.cat([
            inputs_gpu["attention_mask"],
            torch.ones(1, gen_ids.shape[0], dtype=inputs_gpu["attention_mask"].dtype, device=device),
        ], dim=1)

        # Teacher-forced forward pass
        outputs = model(
            input_ids=full_ids,
            attention_mask=full_mask,
            input_features=inputs_gpu["input_features"],
            feature_attention_mask=inputs_gpu["feature_attention_mask"],
        )

        # Extract log-probs for generated tokens only
        logits = outputs.logits  # (1, seq_len, vocab_size)
        # Shifted: logits at position t predict token at position t+1
        gen_logits = logits[0, prompt_len - 1:-1, :]  # (gen_len, vocab_size)
        log_probs = torch.log_softmax(gen_logits.float(), dim=-1)

        # Gather log-probs for actual generated tokens
        token_log_probs = log_probs.gather(
            1, gen_ids.to(device).unsqueeze(-1)
        ).squeeze(-1)  # (gen_len,)

        all_log_probs.append(token_log_probs)

    return all_log_probs


# -- Candidate Generation -----------------------------------------------------

def generate_candidates(model, processor, audio_np_list, G, temperature, max_new_tokens, device):
    """Generate G candidate transcriptions per audio input.

    Generates sequentially (one candidate at a time) to keep VRAM bounded
    on the T4. Clears cache between generations.

    Args:
        model: PeftModel in eval mode.
        processor: Qwen3ASRProcessor.
        audio_np_list: List of numpy audio arrays.
        G: Number of candidates per input.
        temperature: Sampling temperature (>= 0.7 for diversity).
        max_new_tokens: Max tokens to generate per candidate.
        device: torch device.

    Returns:
        candidates: List of G lists, each containing batch_size transcription strings.
        generated_ids: List of G lists, each containing batch_size 1D token ID tensors.
    """
    candidates = []  # [G][batch_size] -> str
    generated_ids = []  # [G][batch_size] -> tensor

    model.eval()
    model_dtype = next(model.parameters()).dtype

    for g in range(G):
        g_texts = []
        g_ids = []

        for audio_np in audio_np_list:
            # Build chat-format input with assistant prefix
            conversation = [
                {"role": "user", "content": [{"type": "audio", "audio": audio_np}]}
            ]
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            text += "language English<asr_text>"

            inputs = processor(
                text=text, audio=[audio_np], return_tensors="pt", padding=False
            )
            inputs_gpu = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    if v.is_floating_point():
                        v = v.to(model_dtype)
                inputs_gpu[k] = v

            prompt_len = inputs_gpu["input_ids"].shape[-1]

            with torch.no_grad():
                gen_output = model.generate(
                    **inputs_gpu,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )

            # Extract generated tokens
            if hasattr(gen_output, "sequences"):
                output_ids = gen_output.sequences
            elif isinstance(gen_output, torch.Tensor):
                output_ids = gen_output
            else:
                output_ids = gen_output[0]

            new_token_ids = output_ids[0, prompt_len:]  # 1D tensor
            transcript = processor.tokenizer.decode(
                new_token_ids, skip_special_tokens=True
            )

            g_texts.append(transcript)
            g_ids.append(new_token_ids.cpu())

        candidates.append(g_texts)
        generated_ids.append(g_ids)

        # Aggressive cleanup between candidates
        VRAMMonitor.cleanup()

    return candidates, generated_ids


# -- Single GRPO Step ----------------------------------------------------------

def grpo_step(
    model,
    processor,
    batch,
    config: GRPOConfig,
    reward_computer: RewardComputer,
    device,
):
    """Single GRPO training step.

    1. Generate G candidates (no grad)
    2. Compute rewards
    3. Compute advantages
    4. Compute old log-probs (teacher-forced, no grad)
    5. Compute current log-probs (teacher-forced, with grad)
    6. Compute clipped surrogate loss + optional KL penalty

    Args:
        model: PeftModel (trainable).
        processor: Qwen3ASRProcessor.
        batch: Dict from data_loader.collate_fn.
        config: GRPOConfig.
        reward_computer: RewardComputer instance.
        device: torch device.

    Returns:
        loss: Scalar tensor (backward-ready).
        metrics: Dict of diagnostic values.
    """
    audio_np_list = [batch["audio"][i].numpy() for i in range(batch["audio"].shape[0])]
    references = batch["transcripts"]
    demographics = batch["demographic_groups"]
    batch_size = len(references)

    # Step 1: Generate G candidates
    candidates, generated_ids = generate_candidates(
        model, processor, audio_np_list,
        config.G, config.temperature, config.max_new_tokens, device,
    )

    # Step 2: Compute rewards
    rewards, reward_metrics = reward_computer(candidates, references, demographics)
    rewards = rewards.to(device)  # (batch_size, G)

    # Step 3: Compute advantages (group-relative normalization)
    if config.dr_grpo:
        # Dr. GRPO: no std division (eliminates difficulty bias)
        advantages = rewards - rewards.mean(dim=1, keepdim=True)
    else:
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = (rewards - mean) / std

    # Step 4: Compute old log-probs (no grad, teacher-forced)
    model.eval()
    with torch.no_grad():
        old_log_probs_list = []  # [G][batch_size] -> per-token log-probs
        for g in range(config.G):
            old_lp = compute_per_token_log_probs(
                model, processor, audio_np_list, generated_ids[g], device
            )
            old_log_probs_list.append(old_lp)

    # Step 4b: Compute reference log-probs (for KL penalty)
    ref_log_probs_list = None
    if config.beta > 0:
        with torch.no_grad():
            model.disable_adapter_layers()
            ref_log_probs_list = []
            for g in range(config.G):
                ref_lp = compute_per_token_log_probs(
                    model, processor, audio_np_list, generated_ids[g], device
                )
                ref_log_probs_list.append(ref_lp)
            model.enable_adapter_layers()

    # Step 5: Compute current log-probs (with grad)
    model.train()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_kl = 0.0
    total_clip_frac = 0.0
    total_entropy = 0.0
    n_tokens = 0

    for g in range(config.G):
        current_log_probs = compute_per_token_log_probs(
            model, processor, audio_np_list, generated_ids[g], device
        )

        for i in range(batch_size):
            curr_lp = current_log_probs[i]  # (gen_len,)
            old_lp = old_log_probs_list[g][i].to(device)  # (gen_len,)
            adv = advantages[i, g]

            # Per-token importance ratio
            ratio = torch.exp(curr_lp - old_lp)

            # Clipped surrogate objective (DAPO asymmetric clipping)
            clipped_ratio = ratio.clamp(
                1 - config.epsilon_low, 1 + config.epsilon_high
            )
            surrogate = torch.min(ratio * adv, clipped_ratio * adv)

            # Token-level loss (negative because we maximize)
            token_loss = -surrogate.mean()
            total_loss = total_loss + token_loss

            # Clip fraction diagnostic
            total_clip_frac += (
                (ratio < 1 - config.epsilon_low) | (ratio > 1 + config.epsilon_high)
            ).float().mean().item()

            # KL penalty
            if config.beta > 0 and ref_log_probs_list is not None:
                ref_lp = ref_log_probs_list[g][i].to(device)
                # Schulman (2020) approximator: r - log(r) - 1
                log_ratio = curr_lp - ref_lp
                kl = (torch.exp(log_ratio) - log_ratio - 1).mean()
                total_loss = total_loss + config.beta * kl
                total_kl += kl.item()

            # Entropy approximation: -mean(log_prob) per token (collapse indicator)
            total_entropy += (-curr_lp.detach()).mean().item()

            n_tokens += curr_lp.shape[0]

    # Average over batch * G
    total_loss = total_loss / (batch_size * config.G)

    metrics = {
        **reward_metrics,
        "loss": total_loss.item(),
        "advantage/mean": advantages.mean().item(),
        "advantage/std": advantages.std().item(),
        "clip_frac": total_clip_frac / (batch_size * config.G),
        "kl": total_kl / (batch_size * config.G) if config.beta > 0 else 0.0,
        "n_tokens": n_tokens,
        "entropy": total_entropy / (batch_size * config.G),
    }

    return total_loss, metrics


# -- Training Loop -------------------------------------------------------------

def _eval_wer_proxy(model, processor, eval_loader, device, max_samples=50):
    """Fast eval proxy: compute mean WER and per-group WER on a small sample.

    Returns dict with mean_wer, per_group_wer, and fairness_gap.
    """
    from whisper.normalizers import EnglishTextNormalizer
    _norm = EnglishTextNormalizer()
    model.eval()
    model_dtype = next(model.parameters()).dtype

    wers_by_group: dict[str, list[float]] = {}
    all_wers = []
    n = 0

    for batch in eval_loader:
        if n >= max_samples:
            break
        for i in range(batch["audio"].shape[0]):
            if n >= max_samples:
                break
            audio_np = batch["audio"][i].numpy()
            ref = _norm(batch["transcripts"][i])
            group = batch["demographic_groups"][i]
            if not ref:
                continue

            conversation = [
                {"role": "user", "content": [{"type": "audio", "audio": audio_np}]}
            ]
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            text += "language English<asr_text>"
            inputs = processor(text=text, audio=[audio_np], return_tensors="pt", padding=False)
            inputs_gpu = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    if v.is_floating_point():
                        v = v.to(model_dtype)
                inputs_gpu[k] = v

            with torch.no_grad():
                gen_out = model.generate(**inputs_gpu, max_new_tokens=128, do_sample=False)
            if hasattr(gen_out, "sequences"):
                output_ids = gen_out.sequences
            elif isinstance(gen_out, torch.Tensor):
                output_ids = gen_out
            else:
                output_ids = gen_out[0]

            prompt_len = inputs_gpu["input_ids"].shape[-1]
            hyp = _norm(processor.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True))

            wer_val = jiwer.wer(ref, hyp) if hyp else 1.0
            all_wers.append(wer_val)
            if group and group.strip():
                wers_by_group.setdefault(group.strip(), []).append(wer_val)
            n += 1

    model.train()

    mean_wer = np.mean(all_wers) if all_wers else 1.0
    group_means = {g: np.mean(ws) for g, ws in wers_by_group.items() if len(ws) >= 3}
    if len(group_means) >= 2:
        fairness_gap = max(group_means.values()) - min(group_means.values())
    else:
        fairness_gap = 0.0

    return {"mean_wer": mean_wer, "per_group_wer": group_means, "fairness_gap": fairness_gap}


def select_best_checkpoint(eval_history, alpha=0.5):
    """Select best checkpoint by composite metric.

    score = alpha * (1 - mean_WER) + (1 - alpha) * (1 - fairness_gap)
    Higher is better.
    """
    if not eval_history:
        return None
    best_idx = -1
    best_score = -float("inf")
    for i, entry in enumerate(eval_history):
        score = alpha * (1 - entry["mean_wer"]) + (1 - alpha) * (1 - entry["fairness_gap"])
        if score > best_score:
            best_score = score
            best_idx = i
    return eval_history[best_idx] if best_idx >= 0 else None


def train_grpo(
    config: GRPOConfig,
    train_loader,
    eval_loader,
    model,
    processor,
    output_dir: str,
    device: str = "cuda",
    wandb_run_name: str | None = None,
    eval_max_samples: int = 50,
):
    """Full GRPO training loop with early stopping and best checkpoint selection.

    Args:
        config: GRPOConfig.
        train_loader: DataLoader with DemographicStratifiedSampler.
        eval_loader: DataLoader for evaluation (sequential).
        model: PeftModel (trainable, already on device).
        processor: Qwen3ASRProcessor.
        output_dir: Directory for outputs (adapter, logs, checkpoints).
        device: CUDA device.
        wandb_run_name: Optional W&B run name.
        eval_max_samples: Max samples for fast eval proxy.

    Returns:
        Dict of final metrics, training summary, and best checkpoint path.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # Optimizer with linear warmup
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=1e-4,
    )

    # Linear warmup scheduler
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Reward computer
    reward_computer = RewardComputer(
        lambda_=config.lambda_,
        baseline_wer=config.baseline_wer,
    )

    # VRAM monitor
    vram_monitor = VRAMMonitor(budget_gb=VRAM_BUDGET_GB)

    # W&B
    if wandb_run_name:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "asr-fairness-grpo"),
            name=wandb_run_name,
            group=os.environ.get("WANDB_GROUP", None),
            config=config.to_dict(),
            tags=["grpo", f"lambda-{config.lambda_}"],
            reinit=True,
        )

    # Training state
    train_iter = iter(train_loader)
    step_metrics_log = []
    t0 = time.time()
    initial_entropy = None

    # Early stopping state
    eval_history = []
    best_eval_wer = float("inf")
    patience_counter = 0
    stopped_early = False

    print(f"\n{'='*60}")
    print(f"GRPO Training: lambda={config.lambda_}, G={config.G}, "
          f"dr_grpo={config.dr_grpo}")
    print(f"Steps: {config.max_steps}, lr={config.lr:.2e}, "
          f"beta={config.beta}, eps_low={config.epsilon_low}, eps_high={config.epsilon_high}")
    print(f"Temperature: {config.temperature}, max_new_tokens={config.max_new_tokens}")
    print(f"Early stop: patience={config.early_stop_patience}, "
          f"threshold={config.early_stop_threshold}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    optimizer.zero_grad()
    accum_loss = 0.0
    accum_metrics: dict[str, float] = {}

    for step in range(1, config.max_steps + 1):
        # Get next batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # GRPO step
        loss, metrics = grpo_step(
            model, processor, batch, config, reward_computer, device
        )

        # Track initial entropy for collapse detection
        if initial_entropy is None and "entropy" in metrics:
            initial_entropy = metrics["entropy"]

        # Gradient accumulation
        scaled_loss = loss / config.gradient_accumulation_steps
        scaled_loss.backward()
        accum_loss += loss.item()

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v

        if step % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            VRAMMonitor.cleanup()

        # Logging
        if step % config.log_every == 0:
            avg_loss = accum_loss / config.log_every
            avg_metrics = {
                k: v / config.log_every
                for k, v in accum_metrics.items()
                if isinstance(v, (int, float))
            }
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]

            entropy_val = avg_metrics.get('entropy', 0)
            entropy_warn = ""
            if initial_entropy and initial_entropy > 0 and entropy_val < initial_entropy * 0.5:
                entropy_warn = " COLLAPSE?"

            print(f"  Step {step}/{config.max_steps} | "
                  f"loss={avg_loss:.4f} | reward={avg_metrics.get('reward/mean', 0):.4f} | "
                  f"adv_std={avg_metrics.get('advantage/std', 0):.4f} | "
                  f"clip={avg_metrics.get('clip_frac', 0):.3f} | "
                  f"wer={avg_metrics.get('wer/mean', 0):.4f} | "
                  f"entropy={entropy_val:.3f}{entropy_warn} | "
                  f"lr={lr_now:.2e} | {elapsed:.0f}s")

            step_entry = {"step": step, "loss": avg_loss, "lr": lr_now, **avg_metrics}
            step_metrics_log.append(step_entry)

            if wandb_run_name:
                wandb.log(step_entry, step=step)

            accum_loss = 0.0
            accum_metrics = {}

        # VRAM check
        if step % config.log_every == 0:
            ok, peak = vram_monitor.check(step)
            if not ok:
                print(f"  VRAM WARNING at step {step}: peak={peak:.2f}GB")

        # Save checkpoint
        if step % config.save_every == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoints", f"step_{step}")
            model.save_pretrained(ckpt_dir)
            print(f"  Checkpoint saved: {ckpt_dir}")

        # Eval + early stopping check
        if step % config.eval_every == 0 and eval_loader is not None:
            eval_result = _eval_wer_proxy(model, processor, eval_loader, device, eval_max_samples)
            eval_result["step"] = step
            eval_result["checkpoint_dir"] = os.path.join(output_dir, "checkpoints", f"step_{step}")
            # Save checkpoint at eval time too (if not already saved this step)
            if step % config.save_every != 0:
                model.save_pretrained(eval_result["checkpoint_dir"])
            eval_history.append(eval_result)

            print(f"  [EVAL step {step}] mean_WER={eval_result['mean_wer']:.4f}, "
                  f"fairness_gap={eval_result['fairness_gap']:.4f}")
            if eval_result["per_group_wer"]:
                for g, w in sorted(eval_result["per_group_wer"].items()):
                    print(f"    {g}: {w:.4f}")

            if wandb_run_name:
                eval_log = {
                    "eval/mean_wer": eval_result["mean_wer"],
                    "eval/fairness_gap": eval_result["fairness_gap"],
                }
                for g, w in eval_result.get("per_group_wer", {}).items():
                    eval_log[f"eval/wer_{g}"] = w
                wandb.log(eval_log, step=step)

            # Early stopping: check if WER degraded
            if eval_result["mean_wer"] < best_eval_wer * (1 + config.early_stop_threshold):
                if eval_result["mean_wer"] < best_eval_wer:
                    best_eval_wer = eval_result["mean_wer"]
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  Early stop: patience {patience_counter}/{config.early_stop_patience}")
                if patience_counter >= config.early_stop_patience:
                    print(f"  EARLY STOPPING at step {step}: WER degraded for "
                          f"{config.early_stop_patience} consecutive evals")
                    stopped_early = True
                    break

    # Final save (always save final adapter)
    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    elapsed = time.time() - t0

    # Select best checkpoint
    best_ckpt = select_best_checkpoint(eval_history, alpha=config.checkpoint_select_alpha)
    best_ckpt_path = best_ckpt["checkpoint_dir"] if best_ckpt else adapter_dir

    # Save training log
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(step_metrics_log, f, indent=2, default=str)

    # Save eval history
    eval_log_path = os.path.join(output_dir, "eval_history.json")
    with open(eval_log_path, "w") as f:
        # Strip non-serializable per_group_wer keys
        serializable = []
        for e in eval_history:
            entry = {k: v for k, v in e.items()}
            serializable.append(entry)
        json.dump(serializable, f, indent=2, default=str)

    # Save config
    config_path = os.path.join(output_dir, "grpo_config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Summary
    summary = {
        "lambda": config.lambda_,
        "G": config.G,
        "max_steps": config.max_steps,
        "actual_steps": step,
        "stopped_early": stopped_early,
        "training_time_sec": elapsed,
        "final_loss": step_metrics_log[-1]["loss"] if step_metrics_log else None,
        "final_reward": step_metrics_log[-1].get("reward/mean") if step_metrics_log else None,
        "best_checkpoint_path": best_ckpt_path,
        "best_eval_wer": best_ckpt["mean_wer"] if best_ckpt else None,
        "best_fairness_gap": best_ckpt["fairness_gap"] if best_ckpt else None,
        "best_step": best_ckpt["step"] if best_ckpt else None,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None,
    }
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GRPO TRAINING COMPLETE{'  (EARLY STOPPED)' if stopped_early else ''}")
    print(f"{'='*60}")
    print(f"  Lambda: {config.lambda_}")
    print(f"  Steps: {step}/{config.max_steps}")
    print(f"  Time: {elapsed/60:.1f} min")
    if step_metrics_log:
        print(f"  Final loss: {step_metrics_log[-1]['loss']:.4f}")
        print(f"  Final reward: {step_metrics_log[-1].get('reward/mean', 'N/A')}")
    if best_ckpt:
        print(f"  Best checkpoint: step {best_ckpt['step']} "
              f"(WER={best_ckpt['mean_wer']:.4f}, gap={best_ckpt['fairness_gap']:.4f})")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak VRAM: {peak:.2f} GB")

    if wandb_run_name:
        wandb.finish()

    return summary


# -- Data Setup ----------------------------------------------------------------

def setup_data(args):
    """Create train/eval dataloaders."""
    print(f"\n  Creating {args.subset_size}-sample subset (equal FS/CV)...")
    subset_df = create_stratified_subset(
        args.fs_manifest, args.cv_manifest, args.subset_size, SEED
    )
    print(f"  Subset: {len(subset_df)} samples")

    train_df, eval_df = create_speaker_disjoint_split(subset_df, test_size=0.1, seed=SEED)
    print(f"  Train: {len(train_df)}, Eval: {len(eval_df)}")

    # Save subsets
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train_subset.csv")
    eval_path = os.path.join(args.output_dir, "eval_subset.csv")
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    axis = "ethnicity" if "ethnicity" in train_df.columns else "accent"

    # Train loader with demographic stratification
    train_dataset = ASRFairnessDataset(train_path, demographic_axis=axis)
    train_sampler = DemographicStratifiedSampler(
        train_dataset.demographics, batch_size=1, seed=SEED
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Eval loader (sequential)
    eval_dataset = ASRFairnessDataset(eval_path, demographic_axis=axis)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    return train_loader, eval_loader


def setup_fullscale_data(args):
    """Create train/eval dataloaders using the FULL dataset (Phase 5).

    Combines Fair-Speech (ethnicity) and Common Voice (accent) into a unified
    dataset with a 'demographic_group' column. Uses speaker-disjoint split.
    """
    print(f"\n  Loading full datasets (no subset)...")
    fs_df = pd.read_csv(args.fs_manifest)
    cv_df = pd.read_csv(args.cv_manifest)

    # Unify demographic axis: ethnicity for FS, accent for CV
    fs_df["demographic_group"] = fs_df.get("ethnicity", pd.Series([""] * len(fs_df))).fillna("")
    cv_df["demographic_group"] = cv_df.get("accent", pd.Series([""] * len(cv_df))).fillna("")

    combined = pd.concat([fs_df, cv_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"  Combined: {len(combined)} samples (FS={len(fs_df)}, CV={len(cv_df)})")

    train_df, eval_df = create_speaker_disjoint_split(combined, test_size=0.1, seed=SEED)
    print(f"  Train: {len(train_df)}, Eval: {len(eval_df)}")

    # Save splits for reproducibility
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train_fullscale.csv")
    eval_path = os.path.join(args.output_dir, "eval_fullscale.csv")
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    # Train loader with demographic stratification
    train_dataset = ASRFairnessDataset(train_path, demographic_axis="demographic_group")
    train_sampler = DemographicStratifiedSampler(
        train_dataset.demographics, batch_size=1, seed=SEED
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Eval loader (sequential)
    eval_dataset = ASRFairnessDataset(eval_path, demographic_axis="demographic_group")
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    return train_loader, eval_loader


# -- Model Setup ---------------------------------------------------------------

def setup_model(device="cuda"):
    """Load model with LoRA using locked config from Phase 2.1."""
    # Load locked config for LoRA HPs
    locked_path = "outputs/standard-lora/locked_config.json"
    if os.path.exists(locked_path):
        with open(locked_path) as f:
            locked = json.load(f)
        params = locked["params"]
        rank = int(params["rank"])
        alpha = int(params.get("alpha", rank * int(params.get("alpha_ratio", 2))))
        dropout = float(params.get("dropout", 0.05))
        target_mlp = bool(params.get("target_mlp", False))
        print(f"  LoRA from locked config: rank={rank}, alpha={alpha}, "
              f"dropout={dropout}, target_mlp={target_mlp}")
    else:
        # Defaults matching Phase 2.1 winner
        rank, alpha, dropout, target_mlp = 4, 4, 0.05, True
        print(f"  LoRA defaults: rank={rank}, alpha={alpha}, "
              f"dropout={dropout}, target_mlp={target_mlp}")

    model, processor = load_model_and_processor()
    model = apply_lora(model, rank, alpha, dropout, target_mlp)
    if torch.cuda.is_available():
        model = model.to(device)
    print_gpu_memory("After LoRA model load")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, processor


# -- Modes ---------------------------------------------------------------------

def run_prototype(args):
    """Prototype: validate GRPO training signal with single lambda."""
    print(f"\n{'='*60}")
    print("GRPO PROTOTYPE: Validating Training Signal")
    print(f"{'='*60}")

    train_loader, eval_loader = setup_data(args)
    model, processor = setup_model()

    config = GRPOConfig(
        G=args.G,
        lambda_=args.lambda_,
        lr=args.lr,
        beta=args.beta,
        epsilon=args.epsilon,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dr_grpo=args.dr_grpo,
        baseline_wer=args.baseline_wer,
    )

    lambda_dir = os.path.join(args.output_dir, f"lambda_{args.lambda_}")
    summary = train_grpo(
        config, train_loader, eval_loader, model, processor,
        lambda_dir, wandb_run_name=f"grpo-proto-lambda{args.lambda_}",
    )

    # Signal check
    log_path = os.path.join(lambda_dir, "training_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            logs = json.load(f)
        if logs:
            avg_zero_std = np.mean([l.get("reward/frac_zero_std", 1.0) for l in logs[-10:]])
            avg_adv_std = np.mean([l.get("advantage/std", 0.0) for l in logs[-10:]])
            print(f"\n  Signal check:")
            print(f"    frac_zero_std (last 10): {avg_zero_std:.3f} {'WARN: >0.5' if avg_zero_std > 0.5 else 'OK'}")
            print(f"    advantage_std (last 10): {avg_adv_std:.4f} {'WARN: <0.01' if avg_adv_std < 0.01 else 'OK'}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def run_sweep(args):
    """Lambda sweep: 3 values on prototype scale."""
    print(f"\n{'='*60}")
    print("GRPO LAMBDA SWEEP: Prototype Scale")
    print(f"{'='*60}")

    train_loader, eval_loader = setup_data(args)
    sweep_results = []

    for lambda_ in PROTOTYPE_LAMBDAS:
        print(f"\n{'='*60}")
        print(f"SWEEP: lambda={lambda_}")
        print(f"{'='*60}")

        # Fresh model for each lambda
        model, processor = setup_model()

        config = GRPOConfig(
            G=args.G,
            lambda_=lambda_,
            lr=args.lr,
            beta=args.beta,
            epsilon=args.epsilon,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_steps=args.max_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dr_grpo=args.dr_grpo,
            baseline_wer=args.baseline_wer,
        )

        lambda_dir = os.path.join(args.output_dir, f"lambda_{lambda_}")
        summary = train_grpo(
            config, train_loader, eval_loader, model, processor,
            lambda_dir, wandb_run_name=f"grpo-sweep-lambda{lambda_}",
        )
        summary["lambda"] = lambda_
        sweep_results.append(summary)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save sweep summary
    sweep_path = os.path.join(args.output_dir, "sweep_results.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\n{'='*60}")
    print("LAMBDA SWEEP COMPLETE")
    print(f"{'='*60}")
    for r in sweep_results:
        print(f"  lambda={r['lambda']}: loss={r.get('final_loss', 'N/A')}, "
              f"reward={r.get('final_reward', 'N/A')}")

    return sweep_results


def run_fullscale(args):
    """Full-scale GRPO training with a single lambda on the complete dataset (Phase 5)."""
    print(f"\n{'='*60}")
    print("GRPO FULL-SCALE: Single Lambda on Full Dataset")
    print(f"{'='*60}")

    train_loader, eval_loader = setup_fullscale_data(args)
    model, processor = setup_model()

    config = GRPOConfig(
        G=args.G,
        lambda_=args.lambda_,
        lr=args.lr,
        beta=args.beta,
        epsilon_low=args.epsilon_low,
        epsilon_high=args.epsilon_high,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dr_grpo=args.dr_grpo,
        baseline_wer=args.baseline_wer,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        warmup_steps=args.warmup_steps,
        early_stop_patience=args.early_stop_patience,
        early_stop_threshold=args.early_stop_threshold,
        checkpoint_select_alpha=args.checkpoint_select_alpha,
    )

    lambda_dir = os.path.join(args.output_dir, f"lambda_{args.lambda_:.2f}")
    summary = train_grpo(
        config, train_loader, eval_loader, model, processor,
        lambda_dir, device="cuda",
        wandb_run_name=f"grpo-full-lambda{args.lambda_:.2f}",
        eval_max_samples=args.eval_max_samples,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# -- CLI -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GRPO training for fairness-aware ASR fine-tuning")
    parser.add_argument("--mode", required=True,
                        choices=["prototype", "sweep", "fullscale"])
    parser.add_argument("--fs_manifest", default="outputs/manifests/fs_train.csv")
    parser.add_argument("--cv_manifest", default="outputs/manifests/cv_train.csv")
    parser.add_argument("--output_dir", default="outputs/grpo-prototype")
    parser.add_argument("--subset_size", type=int, default=2000)
    parser.add_argument("--wandb_project", default="asr-fairness-grpo")

    # GRPO hyperparameters
    parser.add_argument("--G", type=int, default=2, help="Candidates per input")
    parser.add_argument("--lambda_", type=float, default=0.0, help="Fairness weight")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Symmetric clip range (prototype/sweep)")
    parser.add_argument("--epsilon_low", type=float, default=0.2,
                        help="DAPO asymmetric clip lower bound (fullscale)")
    parser.add_argument("--epsilon_high", type=float, default=0.28,
                        help="DAPO asymmetric clip upper bound (fullscale)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--dr_grpo", action="store_true", default=True,
                        help="Use Dr. GRPO (no std normalization)")
    parser.add_argument("--no_dr_grpo", action="store_false", dest="dr_grpo")
    parser.add_argument("--baseline_wer", type=float, default=None,
                        help="SFT baseline WER for reward floor check")

    # Full-scale training parameters (Phase 5)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Stop if eval WER degrades for N consecutive evals")
    parser.add_argument("--early_stop_threshold", type=float, default=0.05,
                        help="Relative WER degradation threshold for early stopping")
    parser.add_argument("--checkpoint_select_alpha", type=float, default=0.5,
                        help="Weight for accuracy vs fairness in checkpoint selection")
    parser.add_argument("--eval_max_samples", type=int, default=50,
                        help="Max samples for fast eval proxy")

    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = args.wandb_project
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.mode == "prototype":
        run_prototype(args)
    elif args.mode == "sweep":
        run_sweep(args)
    elif args.mode == "fullscale":
        run_fullscale(args)


if __name__ == "__main__":
    main()
