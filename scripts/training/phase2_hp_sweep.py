"""
Phase 2: Small-Scale HP Exploration for LoRA fine-tuning.

Runs an Optuna TPE sweep over LoRA hyperparameters on a 500-sample
stratified subset. Each trial trains for ~100 steps and reports
validation loss. Identifies top-3 configurations for Phase 3 validation.

Usage:
    python scripts/training/phase2_hp_sweep.py \
        --fs_manifest outputs/manifests/fs_train.csv \
        --cv_manifest outputs/manifests/cv_train.csv \
        --n_trials 20 \
        --steps_per_trial 100 \
        --subset_size 500 \
        --output_dir outputs/hp-sweep
"""

import argparse
import gc
import json
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import optuna
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Subset
from transformers import TrainingArguments, Trainer

from scripts.training.data_loader import ASRFairnessDataset
from scripts.training.data_collator import DataCollatorForQwen3ASR

# -- Constants ----------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_SUBSET_SIZE = 500
DEFAULT_STEPS = 100
DEFAULT_N_TRIALS = 20
DEFAULT_EVAL_FRACTION = 0.1  # 10% of subset for validation
DEFAULT_OUTPUT_DIR = "outputs/hp-sweep"
SEED = 42


# -- Model Helpers ------------------------------------------------------------

def load_model_and_processor():
    from qwen_asr import Qwen3ASRModel
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=None,
    )
    return asr_wrapper.model, asr_wrapper.processor


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    def forward(self, input_ids=None, attention_mask=None,
                input_features=None, feature_attention_mask=None,
                labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


def apply_lora(model, rank, alpha, dropout, target_mlp=False):
    """Apply LoRA with given hyperparameters."""
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
    )

    model = get_peft_model(model, lora_config)

    # Re-freeze encoder (PEFT may have added LoRA to audio_tower q/k/v_proj)
    for name, param in model.named_parameters():
        if "audio_tower" in name:
            param.requires_grad = False

    return model


# -- Data Helpers -------------------------------------------------------------

def create_stratified_subset(fs_manifest, cv_manifest, subset_size, seed=42):
    """Create stratified subset with proportional FS/CV split."""
    fs_df = pd.read_csv(fs_manifest)
    cv_df = pd.read_csv(cv_manifest)

    total = len(fs_df) + len(cv_df)
    fs_fraction = len(fs_df) / total
    fs_n = int(subset_size * fs_fraction)
    cv_n = subset_size - fs_n

    # Stratified sample from Fair-Speech by ethnicity
    if "ethnicity" in fs_df.columns:
        fs_groups = fs_df.groupby("ethnicity")
        fs_samples = []
        for _, group_df in fs_groups:
            n = max(1, int(len(group_df) / len(fs_df) * fs_n))
            n = min(n, len(group_df))
            fs_samples.append(group_df.sample(n=n, random_state=seed))
        fs_subset = pd.concat(fs_samples).head(fs_n)
    else:
        fs_subset = fs_df.sample(n=min(fs_n, len(fs_df)), random_state=seed)

    # Stratified sample from Common Voice by accent
    cv_n = min(cv_n, len(cv_df))
    if "accent" in cv_df.columns:
        accented = cv_df[cv_df["accent"].notna() & (cv_df["accent"] != "")]
        unaccented = cv_df[~cv_df.index.isin(accented.index)]

        acc_n = int(cv_n * len(accented) / len(cv_df))
        unacc_n = cv_n - acc_n

        cv_groups = accented.groupby("accent")
        cv_samples = []
        for _, group_df in cv_groups:
            n = max(1, int(len(group_df) / len(accented) * acc_n))
            n = min(n, len(group_df))
            cv_samples.append(group_df.sample(n=n, random_state=seed))
        cv_subset_acc = pd.concat(cv_samples).head(acc_n) if cv_samples else pd.DataFrame()

        cv_subset_unacc = unaccented.sample(
            n=min(unacc_n, len(unaccented)), random_state=seed
        )
        cv_subset = pd.concat([cv_subset_acc, cv_subset_unacc])
    else:
        cv_subset = cv_df.sample(n=cv_n, random_state=seed)

    combined = pd.concat([fs_subset, cv_subset], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    return combined


# -- Objective ----------------------------------------------------------------

def create_objective(fs_manifest, cv_manifest, processor, subset_size,
                     steps_per_trial, eval_fraction, output_dir):
    """Create Optuna objective closure."""

    # Pre-create subset and split train/eval
    print("Creating stratified subset...")
    subset_df = create_stratified_subset(fs_manifest, cv_manifest, subset_size)
    n_eval = max(10, int(len(subset_df) * eval_fraction))
    n_train = len(subset_df) - n_eval

    eval_df = subset_df.tail(n_eval).reset_index(drop=True)
    train_df = subset_df.head(n_train).reset_index(drop=True)

    # Save subsets
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "sweep_train.csv")
    eval_path = os.path.join(output_dir, "sweep_eval.csv")
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    print(f"  Train: {len(train_df)} samples -> {train_path}")
    print(f"  Eval:  {len(eval_df)} samples -> {eval_path}")

    # Determine demographic axis
    axis = "ethnicity" if "ethnicity" in train_df.columns else "accent"

    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        rank = trial.suggest_categorical("rank", [4, 8, 16, 32])
        alpha_ratio = trial.suggest_categorical("alpha_ratio", [1, 2])
        alpha = rank * alpha_ratio
        dropout = trial.suggest_float("dropout", 0.0, 0.15, step=0.05)
        target_mlp = trial.suggest_categorical("target_mlp", [False, True])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        print(f"\n  Trial {trial.number}: lr={lr:.2e}, rank={rank}, "
              f"alpha={alpha}, dropout={dropout:.2f}, "
              f"mlp={target_mlp}, wd={weight_decay:.2e}")

        # Load fresh model
        model, _ = load_model_and_processor()
        model = apply_lora(model, rank, alpha, dropout, target_mlp)

        if torch.cuda.is_available():
            model = model.cuda()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable params: {trainable:,}")

        # Create dataset and collator
        train_dataset = ASRFairnessDataset(train_path, demographic_axis=axis)
        eval_dataset = ASRFairnessDataset(eval_path, demographic_axis=axis)
        collator = DataCollatorForQwen3ASR(processor)

        trial_dir = os.path.join(output_dir, f"trial_{trial.number:03d}")

        training_args = TrainingArguments(
            output_dir=trial_dir,
            max_steps=steps_per_trial,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=lr,
            weight_decay=weight_decay,
            bf16=True,
            logging_steps=max(1, steps_per_trial // 10),
            eval_strategy="steps",
            eval_steps=max(1, steps_per_trial // 5),
            save_strategy="no",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to="none",
            seed=SEED,
            load_best_model_at_end=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )

        try:
            trainer.train()

            # Get final eval loss
            eval_results = trainer.evaluate()
            eval_loss = eval_results.get("eval_loss", float("inf"))

            # Get training loss trajectory
            train_losses = [
                e["loss"] for e in trainer.state.log_history if "loss" in e
            ]

            print(f"    Train loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}"
                  if train_losses else "    No train loss logged")
            print(f"    Eval loss: {eval_loss:.4f}")

        except Exception as e:
            print(f"    FAILED: {e}")
            eval_loss = float("inf")

        finally:
            # Aggressive cleanup
            del model, trainer, train_dataset, eval_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return eval_loss

    return objective


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: HP Sweep")
    parser.add_argument("--fs_manifest", required=True)
    parser.add_argument("--cv_manifest", required=True)
    parser.add_argument("--n_trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--steps_per_trial", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--subset_size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument("--eval_fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{'='*60}")
    print("PHASE 2: Small-Scale HP Exploration")
    print(f"{'='*60}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Steps/trial: {args.steps_per_trial}")
    print(f"  Subset size: {args.subset_size}")
    print(f"  Output: {args.output_dir}")

    # Load processor once (shared across trials)
    _, processor = load_model_and_processor()

    # Create objective
    objective = create_objective(
        args.fs_manifest, args.cv_manifest, processor,
        args.subset_size, args.steps_per_trial,
        args.eval_fraction, args.output_dir,
    )

    # Run sweep
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name="lora-hp-sweep",
    )

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials)
    elapsed = time.time() - t0

    # Results
    print(f"\n{'='*60}")
    print(f"PHASE 2 RESULTS ({elapsed/3600:.1f} hours)")
    print(f"{'='*60}")

    # Top 3 trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value")

    print(f"\n  Top 3 configurations:")
    for i, (_, row) in enumerate(trials_df.head(3).iterrows()):
        print(f"\n  #{i+1} (trial {int(row['number'])}, "
              f"eval_loss={row['value']:.4f}):")
        for col in trials_df.columns:
            if col.startswith("params_"):
                param_name = col.replace("params_", "")
                print(f"    {param_name}: {row[col]}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Best params
    best = study.best_trial
    best_params = {
        "trial_number": best.number,
        "eval_loss": best.value,
        "params": best.params,
    }
    best_path = os.path.join(args.output_dir, "best_params.json")
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n  Best params saved: {best_path}")

    # Top 3 for Phase 3
    top3 = []
    for _, row in trials_df.head(3).iterrows():
        params = {col.replace("params_", ""): row[col]
                  for col in trials_df.columns if col.startswith("params_")}
        top3.append({
            "trial_number": int(row["number"]),
            "eval_loss": float(row["value"]),
            "params": params,
        })
    top3_path = os.path.join(args.output_dir, "top3_configs.json")
    with open(top3_path, "w") as f:
        json.dump(top3, f, indent=2)
    print(f"  Top 3 configs saved: {top3_path}")

    # Full results
    results_path = os.path.join(args.output_dir, "all_trials.csv")
    trials_df.to_csv(results_path, index=False)
    print(f"  All trials saved: {results_path}")

    # Gate check
    if len(trials_df) >= 3:
        top3_losses = trials_df.head(3)["value"].values
        spread = (top3_losses.max() - top3_losses.min()) / top3_losses.mean()
        print(f"\n  Top-3 loss spread: {spread*100:.1f}%")
        if spread < 0.5:
            print(f"  GATE: Rankings look stable (spread < 50%)")
            print(f"  Ready for Phase 3: Medium-Scale Validation")
        else:
            print(f"  GATE WARNING: Rankings may be noisy (spread >= 50%)")
            print(f"  Consider increasing subset_size or steps_per_trial")


if __name__ == "__main__":
    main()
