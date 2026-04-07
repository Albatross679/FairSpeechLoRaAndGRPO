"""
Multi-adapter round-robin training loop for PLoRA packed sweep.

Manages N LoRA adapters on one frozen base model with per-adapter
optimizers, data iterators, and evaluation. Uses a custom PyTorch
training loop (NOT HuggingFace Trainer) for full control over
adapter switching and gradient isolation.

Usage:
    from scripts.training.packed_trainer import PackedTrainer, create_packed_model

    model = load_model_and_processor()[0]
    model = create_packed_model(model, first_config)
    model = model.cuda()

    trainer = PackedTrainer(model, adapter_configs, processor, train_df, eval_df)
    trainer.setup_adapters()
    for step, rung_results in trainer.train(n_steps=100, eval_steps=20):
        pass  # Handle rung results for pruning
    results = trainer.get_results()
"""

import gc
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


# -- PackedTrainer ------------------------------------------------------------

class PackedTrainer:
    """Multi-adapter round-robin training loop.

    Manages N LoRA adapters on one frozen base model. Each adapter has its
    own optimizer, data iterator, and loss history. Training alternates
    between adapters at each step.

    Args:
        model: Base model with first adapter already applied via get_peft_model().
        adapter_configs: List of dicts with keys: name, rank, alpha, dropout,
            target_mlp, lr, weight_decay.
        processor: Qwen3ASRProcessor for collation.
        train_df: DataFrame for training data.
        eval_df: DataFrame for evaluation data.
        steps_per_trial: Training steps per adapter.
        eval_steps: Evaluate every N steps.
        per_device_batch_size: Batch size per adapter (default 2).
        seed: Random seed.
    """

    def __init__(self, model, adapter_configs, processor, train_df, eval_df,
                 steps_per_trial=100, eval_steps=20, per_device_batch_size=2,
                 seed=42):
        self.model = model
        self.adapter_configs = adapter_configs
        self.processor = processor
        self.train_df = train_df
        self.eval_df = eval_df
        self.steps_per_trial = steps_per_trial
        self.eval_steps = eval_steps
        self.per_device_batch_size = per_device_batch_size
        self.seed = seed

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
        self.data_iters = {}
        self.losses = {}
        self.eval_losses = {}
        self.eval_loader = None
        self._temp_dir = None

    def setup_adapters(self):
        """Set up adapters, optimizers, and data iterators.

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
            )
            self.model.add_adapter(name, adapter_config)
            self.active_adapters.append(name)

        # Re-freeze audio_tower after all adapters added
        for name, param in self.model.named_parameters():
            if "audio_tower" in name:
                param.requires_grad = False

        # Create per-adapter optimizers
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
            trainable_count = sum(
                1 for n, p in self.model.named_parameters()
                if p.requires_grad and config["name"] in n
            ) if config["name"] != "default" else "N/A (default)"
            print(f"    {config['name']}: rank={config['rank']}, "
                  f"mlp={config.get('target_mlp', False)}, lr={config['lr']:.2e}")

    def train_step(self, step):
        """Execute one round-robin training step across all active adapters."""
        for name in self.active_adapters:
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

            self.optimizers[name].zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(**batch)
                loss = outputs.loss

            loss.backward()
            self.optimizers[name].step()
            self.losses[name].append(loss.item())

    def evaluate(self, step):
        """Evaluate all active adapters on the eval set."""
        for name in self.active_adapters:
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

        # Print VRAM after evaluation
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  Step {step}: peak VRAM = {vram_mb:.0f} MB")

    def remove_adapter(self, name):
        """Remove an adapter from active training."""
        if name in self.active_adapters:
            self.active_adapters.remove(name)

        if name in self.optimizers:
            del self.optimizers[name]

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

    def train(self, n_steps, eval_steps, rung_steps=None):
        """Main training loop with generator pattern for rung results.

        Args:
            n_steps: Total training steps.
            eval_steps: Evaluate every N steps.
            rung_steps: Optional set of steps to yield rung results for pruning.

        Yields:
            (step, rung_results) at each rung_step, where rung_results maps
            adapter name to eval_loss.
        """
        for step in range(1, n_steps + 1):
            self.train_step(step)

            if step % eval_steps == 0 or step == n_steps:
                self.evaluate(step)

            if rung_steps and step in rung_steps:
                rung_results = {
                    name: self.eval_losses[name][-1][1]
                    for name in self.active_adapters
                    if self.eval_losses[name]
                }
                yield step, rung_results

        # Final yield if no rung_steps or last step wasn't a rung
        if not rung_steps:
            yield n_steps, self.get_results()

    def get_results(self):
        """Return dict mapping adapter name to final eval_loss."""
        return {
            name: self.eval_losses[name][-1][1]
            for name in self.active_adapters
            if self.eval_losses[name]
        }

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

    Returns:
        PEFT-wrapped model with first adapter.
    """
    return apply_lora(
        base_model,
        rank=first_config["rank"],
        alpha=first_config["alpha"],
        dropout=first_config.get("dropout", 0.0),
        target_mlp=first_config.get("target_mlp", False),
    )
