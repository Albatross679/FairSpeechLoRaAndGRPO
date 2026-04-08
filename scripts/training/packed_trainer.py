"""
Multi-adapter round-robin training loop for PLoRA packed sweep.

Manages N LoRA adapters on one frozen base model with per-adapter
optimizers, data iterators, and evaluation. Uses a custom PyTorch
training loop (NOT HuggingFace Trainer) for full control over
adapter switching and gradient isolation.

Supports:
  - Rung-based pruning: yield eval results at rung steps for ASHA pruning decisions
  - Adapter deactivation: mark pruned adapters (free optimizer/iterator, keep weights)
  - Gradient accumulation: per-adapter accumulation steps
  - LR schedulers: linear/cosine decay with warmup (per adapter)
  - RsLoRA: pass use_rslora=True in adapter configs for rank-stable scaling

All features are backward compatible: Stage 1 configs (no rung_steps,
no grad_accum, no scheduler) behave identically to Plan 01 version.

Usage:
    from scripts.training.packed_trainer import PackedTrainer, create_packed_model

    model = load_model_and_processor()[0]
    model = create_packed_model(model, first_config)
    model = model.cuda()

    trainer = PackedTrainer(model, adapter_configs, processor, train_df, eval_df,
                            rung_steps=[25, 50, 75])
    trainer.setup_adapters()
    for step, rung_results in trainer.train(n_steps=100, eval_steps=20):
        # Make pruning decisions
        for name, loss in rung_results.items():
            if should_prune(name):
                trainer.deactivate_adapter(name)
    results = trainer.get_results()
"""

import gc
import math
import os
import sys
import tempfile

import pandas as pd
import torch
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.training.data_loader import ASRFairnessDataset
from scripts.training.data_collator import DataCollatorForQwen3ASR
from scripts.training.phase2_hp_sweep import apply_lora


# -- Cycling Iterator ---------------------------------------------------------

class CyclingIterator:
    """Wraps a DataLoader to cycle infinitely, recreating the iterator on StopIteration."""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iterator = iter(dataloader)

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)


# -- LR Scheduler Helper -----------------------------------------------------

def _create_scheduler(optimizer, config, total_steps):
    """Create an LR scheduler based on adapter config.

    Args:
        optimizer: The adapter's optimizer.
        config: Adapter config dict (may contain lr_scheduler, warmup_ratio).
        total_steps: Total training steps for this adapter.

    Returns:
        LambdaLR scheduler or None if scheduler type is 'constant'.
    """
    sched_type = config.get("lr_scheduler", "constant")
    warmup_ratio = config.get("warmup_ratio", 0.0)
    warmup_steps = int(total_steps * warmup_ratio)

    if sched_type == "constant":
        return None

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if sched_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:  # linear
            return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -- PackedTrainer ------------------------------------------------------------

class PackedTrainer:
    """Multi-adapter round-robin training loop.

    Manages N LoRA adapters on one frozen base model. Each adapter has its
    own optimizer, data iterator, and loss history. Training alternates
    between adapters at each step.

    Args:
        model: Base model with first adapter already applied via get_peft_model().
        adapter_configs: List of dicts with keys: name, rank, alpha, dropout,
            target_mlp, lr, weight_decay. Optional: grad_accum_steps,
            lr_scheduler, warmup_ratio, use_rslora.
        processor: Qwen3ASRProcessor for collation.
        train_df: DataFrame for training data.
        eval_df: DataFrame for evaluation data.
        steps_per_trial: Training steps per adapter.
        eval_steps: Evaluate every N steps.
        per_device_batch_size: Batch size per adapter (default 2).
        seed: Random seed.
        rung_steps: List of step numbers where pruning decisions are made
            (e.g., [25, 50, 75]). If None/empty, no rung-based yielding.
    """

    def __init__(self, model, adapter_configs, processor, train_df, eval_df,
                 steps_per_trial=100, eval_steps=20, per_device_batch_size=2,
                 seed=42, rung_steps=None):
        self.model = model
        self.adapter_configs = adapter_configs
        self.processor = processor
        self.train_df = train_df
        self.eval_df = eval_df
        self.steps_per_trial = steps_per_trial
        self.eval_steps = eval_steps
        self.per_device_batch_size = per_device_batch_size
        self.seed = seed

        # Rung-based pruning support
        self.rung_steps = set(rung_steps) if rung_steps else set()
        self.pruning_stats = {"pruned_at": {}, "active_history": []}

        # Gradient accumulation per adapter
        self.grad_accum_steps = {
            c["name"]: c.get("grad_accum_steps", 1)
            for c in adapter_configs
        }
        self.step_counters = {c["name"]: 0 for c in adapter_configs}

        # Detect demographic axis from train_df columns
        if "ethnicity" in train_df.columns:
            self.demographic_axis = "ethnicity"
        elif "accent" in train_df.columns:
            self.demographic_axis = "accent"
        else:
            self.demographic_axis = None

        # State (populated by setup_adapters)
        self.active_adapters = []
        self.optimizers = {}
        self.schedulers = {}
        self.data_iters = {}
        self.losses = {}
        self.eval_losses = {}
        self.eval_loader = None
        self._temp_dir = None

    def setup_adapters(self):
        """Set up adapters, optimizers, schedulers, and data iterators.

        The first adapter in adapter_configs is assumed already applied via
        get_peft_model (named "default"). Additional adapters are added via
        model.add_adapter().
        """
        self._temp_dir = tempfile.mkdtemp(prefix="packed_trainer_")
        collator = DataCollatorForQwen3ASR(self.processor)

        # Save dataframes to temp CSVs for ASRFairnessDataset
        train_path = os.path.join(self._temp_dir, "train.csv")
        eval_path = os.path.join(self._temp_dir, "eval.csv")
        self.train_df.to_csv(train_path, index=False)
        self.eval_df.to_csv(eval_path, index=False)

        # Add adapters [1:] (first is already "default")
        for i, config in enumerate(self.adapter_configs):
            name = config["name"]
            if i == 0:
                # First adapter is already applied as "default"
                self.active_adapters.append(name)
                continue

            # Build target modules list
            targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
            if config.get("target_mlp", False):
                targets.extend(["gate_proj", "up_proj", "down_proj"])

            adapter_config = LoraConfig(
                r=config["rank"],
                lora_alpha=config["alpha"],
                lora_dropout=config.get("dropout", 0.0),
                target_modules=targets,
                task_type=TaskType.CAUSAL_LM,
                use_rslora=config.get("use_rslora", False),
            )
            self.model.add_adapter(name, adapter_config)
            self.active_adapters.append(name)

        # Re-freeze audio_tower after all adapters added
        for name, param in self.model.named_parameters():
            if "audio_tower" in name:
                param.requires_grad = False

        # Create per-adapter optimizers and schedulers
        for config in self.adapter_configs:
            name = config["name"]
            self.model.set_adapter(name)

            # Collect trainable params for this adapter
            trainable = [
                p for n, p in self.model.named_parameters()
                if p.requires_grad and "lora" in n.lower()
            ]

            if not trainable:
                print(f"  WARNING: No trainable params for adapter {name}")
                continue

            self.optimizers[name] = torch.optim.AdamW(
                trainable,
                lr=config["lr"],
                weight_decay=config.get("weight_decay", 0.0),
            )

            # Create LR scheduler if configured
            scheduler = _create_scheduler(
                self.optimizers[name], config, self.steps_per_trial)
            self.schedulers[name] = scheduler

        # Create per-adapter data iterators (all use same dataset, different shuffle)
        for i, config in enumerate(self.adapter_configs):
            name = config["name"]
            dataset = ASRFairnessDataset(
                train_path,
                demographic_axis=self.demographic_axis,
            )
            loader = DataLoader(
                dataset,
                batch_size=self.per_device_batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=0,
                pin_memory=False,
            )
            self.data_iters[name] = CyclingIterator(loader)
            self.losses[name] = []
            self.eval_losses[name] = []

        # Create eval DataLoader (shared across adapters)
        eval_dataset = ASRFairnessDataset(
            eval_path,
            demographic_axis=self.demographic_axis,
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=False,
        )

        print(f"  PackedTrainer setup: {len(self.active_adapters)} adapters")
        for config in self.adapter_configs:
            sched = config.get("lr_scheduler", "constant")
            accum = config.get("grad_accum_steps", 1)
            rslora = config.get("use_rslora", False)
            print(f"    {config['name']}: rank={config['rank']}, "
                  f"mlp={config.get('target_mlp', False)}, lr={config['lr']:.2e}"
                  f"{f', sched={sched}' if sched != 'constant' else ''}"
                  f"{f', accum={accum}' if accum > 1 else ''}"
                  f"{', rslora=True' if rslora else ''}")

    def train_step(self, step):
        """Execute one round-robin training step across all active adapters.

        Supports gradient accumulation: optimizer.step() is called only
        every grad_accum_steps sub-steps for each adapter. Loss is scaled
        by 1/accum for correct gradient averaging.
        """
        for name in list(self.active_adapters):
            if name not in self.optimizers:
                continue

            self.model.set_adapter(name)
            self.model.train()

            batch = next(self.data_iters[name])
            # Move batch to GPU
            batch = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            accum = self.grad_accum_steps.get(name, 1)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(**batch)
                loss = outputs.loss / accum

            loss.backward()

            self.step_counters[name] = self.step_counters.get(name, 0) + 1
            if self.step_counters[name] % accum == 0:
                self.optimizers[name].step()
                self.optimizers[name].zero_grad()
                # Step LR scheduler after optimizer step
                if self.schedulers.get(name) is not None:
                    self.schedulers[name].step()

            self.losses[name].append(loss.item() * accum)  # Log unscaled loss

    def evaluate(self, step):
        """Evaluate all active adapters on the eval set."""
        for name in list(self.active_adapters):
            self.model.set_adapter(name)
            self.model.eval()

            total_loss = 0.0
            n_batches = 0

            with torch.no_grad():
                for batch in self.eval_loader:
                    batch = {
                        k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = self.model(**batch)
                    total_loss += outputs.loss.item()
                    n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            self.eval_losses[name].append((step, avg_loss))
            print(f"    [{name}] step {step}: eval_loss={avg_loss:.4f}")

    def deactivate_adapter(self, name):
        """Mark adapter as pruned. Frees optimizer, scheduler, and data iterator
        but keeps adapter weights on model for potential post-hoc analysis."""
        if name not in self.active_adapters:
            return
        self.active_adapters.remove(name)
        # Record pruning
        self.pruning_stats["pruned_at"][name] = {
            "step": self.eval_losses[name][-1][0] if self.eval_losses[name] else -1,
            "eval_loss": self.eval_losses[name][-1][1] if self.eval_losses[name] else None,
        }
        # Free optimizer memory
        if name in self.optimizers:
            del self.optimizers[name]
        # Free scheduler
        if name in self.schedulers:
            del self.schedulers[name]
        # Free data iterator
        if name in self.data_iters:
            del self.data_iters[name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Deactivated adapter {name}, {len(self.active_adapters)} active remaining")

    def remove_adapter(self, name):
        """Remove an adapter from active training (deletes weights too)."""
        if name in self.active_adapters:
            self.active_adapters.remove(name)

        if name in self.optimizers:
            del self.optimizers[name]

        if name in self.schedulers:
            del self.schedulers[name]

        if name in self.data_iters:
            del self.data_iters[name]

        # Try to delete adapter from PEFT model
        try:
            self.model.delete_adapter(name)
        except Exception:
            pass  # Some PEFT versions may not support delete_adapter

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Removed adapter {name}, {len(self.active_adapters)} remaining")

    def get_pruning_stats(self):
        """Return dict of pruning statistics for analysis."""
        return {
            "pruned_at": self.pruning_stats["pruned_at"],
            "active_history": self.pruning_stats["active_history"],
            "total_pruned": len(self.pruning_stats["pruned_at"]),
            "survived": list(self.active_adapters),
        }

    def train(self, n_steps, eval_steps, rung_steps=None):
        """Main training loop with generator pattern for rung results.

        Args:
            n_steps: Total training steps.
            eval_steps: Evaluate every N steps.
            rung_steps: Optional list/set of steps to yield rung results for pruning.
                Falls back to self.rung_steps if not provided.

        Yields:
            (step, rung_results) at each rung_step, where rung_results maps
            adapter name to eval_loss. Also yields at final step if no rung_steps.
        """
        effective_rungs = set(rung_steps) if rung_steps else self.rung_steps

        for step in range(1, n_steps + 1):
            if not self.active_adapters:
                print(f"  All adapters pruned at step {step}. Ending early.")
                break

            self.train_step(step)

            # Evaluate at regular intervals OR at rung steps OR at final step
            is_eval_step = (step % eval_steps == 0)
            is_rung_step = (step in effective_rungs)
            is_final = (step == n_steps)

            if is_eval_step or is_rung_step or is_final:
                self.evaluate(step)
                # Print VRAM after evaluation
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    print(f"  Step {step}: peak VRAM = {vram_mb:.0f} MB")

            # Yield at rung steps for pruning decisions
            if is_rung_step:
                rung_results = {
                    name: self.eval_losses[name][-1][1]
                    for name in self.active_adapters
                    if self.eval_losses[name]
                }
                self.pruning_stats["active_history"].append({
                    "step": step,
                    "active_count": len(self.active_adapters),
                    "active_adapters": list(self.active_adapters),
                })
                yield step, rung_results

        # Final yield if no rung_steps or last step wasn't a rung
        if not effective_rungs:
            yield n_steps, self.get_results()

    def get_results(self):
        """Return dict mapping adapter name to final eval_loss."""
        results = {}
        # Include both active and pruned adapters that have eval losses
        for config in self.adapter_configs:
            name = config["name"]
            if self.eval_losses.get(name):
                results[name] = self.eval_losses[name][-1][1]
        return results

    def get_loss_trajectories(self):
        """Return dict mapping adapter name to list of training losses."""
        return dict(self.losses)


# -- Helper: Create Packed Model ----------------------------------------------

def create_packed_model(base_model, first_config):
    """Apply LoRA to base model with first config's parameters.

    Creates the PEFT-wrapped model with the first adapter named "default".
    The caller then passes this to PackedTrainer which adds remaining adapters.

    Args:
        base_model: Raw model from load_model_and_processor().
        first_config: Dict with keys: rank, alpha, dropout, target_mlp.
            Optional: use_rslora (bool).

    Returns:
        PEFT-wrapped model with first adapter.
    """
    use_rslora = first_config.get("use_rslora", False)

    if use_rslora:
        # Use local RsLoRA-aware apply function
        return _apply_lora_with_rslora(
            base_model,
            rank=first_config["rank"],
            alpha=first_config["alpha"],
            dropout=first_config.get("dropout", 0.0),
            target_mlp=first_config.get("target_mlp", False),
            use_rslora=True,
        )
    else:
        return apply_lora(
            base_model,
            rank=first_config["rank"],
            alpha=first_config["alpha"],
            dropout=first_config.get("dropout", 0.0),
            target_mlp=first_config.get("target_mlp", False),
        )


def _apply_lora_with_rslora(model, rank, alpha, dropout, target_mlp=False, use_rslora=False):
    """Apply LoRA with RsLoRA support. Mirrors phase2_hp_sweep.apply_lora()
    but adds use_rslora parameter to LoraConfig.

    This is a local copy to avoid modifying the shared apply_lora() in
    phase2_hp_sweep.py. The only difference is the use_rslora kwarg.
    """
    from scripts.training.phase2_hp_sweep import patch_outer_forward
    from peft import get_peft_model

    patch_outer_forward(model)

    # Patch get_input_embeddings for PEFT compatibility
    cls = model.__class__
    if not hasattr(cls, "_embeddings_patched"):
        def get_input_embeddings(self):
            return self.thinker.model.embed_tokens
        def set_input_embeddings(self, value):
            self.thinker.model.embed_tokens = value
        cls.get_input_embeddings = get_input_embeddings
        cls.set_input_embeddings = set_input_embeddings
        cls._embeddings_patched = True

    # Freeze encoder
    for param in model.thinker.audio_tower.parameters():
        param.requires_grad = False

    # Gradient checkpointing
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if target_mlp:
        targets.extend(["gate_proj", "up_proj", "down_proj"])

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=targets,
        use_rslora=use_rslora,
    )

    model = get_peft_model(model, lora_config)

    # Re-freeze encoder (PEFT may have added LoRA to audio_tower q/k/v_proj)
    for name, param in model.named_parameters():
        if "audio_tower" in name:
            param.requires_grad = False

    return model
