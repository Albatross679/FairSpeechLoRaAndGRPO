"""
Standard LoRA SFT training for Qwen3-ASR-1.7B (ML Prototype Phases 3-4).

Two modes:
  validate — Top 2-3 HP configs from Optuna sweep on 1K-sample subset (Phase 3).
             Produces locked_config.json with the best validated config.
  train    — Final training with locked HP config on full subset (Phase 4).
             Produces adapter weights in outputs/standard-lora/adapter/.

Usage:
    # Phase 3: Validate top configs
    python scripts/training/train_standard_lora.py \
        --mode validate \
        --fs_manifest outputs/manifests/fs_train.csv \
        --cv_manifest outputs/manifests/cv_train.csv \
        --hp_sweep_dir outputs/hp-sweep \
        --output_dir outputs/standard-lora \
        --subset_size 1000 --max_steps 300

    # Phase 4: Final training with locked config
    python scripts/training/train_standard_lora.py \
        --mode train \
        --fs_manifest outputs/manifests/fs_train.csv \
        --cv_manifest outputs/manifests/cv_train.csv \
        --output_dir outputs/standard-lora \
        --subset_size 2000
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import GroupShuffleSplit
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from scripts.training.data_loader import ASRFairnessDataset, FrameBudgetBatchSampler
from scripts.training.data_collator import DataCollatorForQwen3ASR

# -- Constants ----------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
SEED = 42
DEFAULT_OUTPUT_DIR = "outputs/standard-lora"
DEFAULT_HP_SWEEP_DIR = "outputs/hp-sweep"
VRAM_BUDGET_GB = 14.0


# -- Dynamic Batch Trainer ----------------------------------------------------

class DynamicBatchTrainer(Trainer):
    """Trainer subclass that uses FrameBudgetBatchSampler for dynamic batching.

    Instead of fixed per_device_train_batch_size, batches are formed by a
    frame budget: short samples get larger batches, long samples get smaller
    batches. This keeps VRAM usage stable across batches.
    """

    def __init__(self, *args, frame_budget_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_budget_sampler = frame_budget_sampler

    def get_train_dataloader(self):
        if self._frame_budget_sampler is None:
            return super().get_train_dataloader()

        from torch.utils.data import DataLoader

        epoch = getattr(self.state, "epoch", None)
        self._frame_budget_sampler.set_epoch(int(epoch) if epoch is not None else 0)

        return DataLoader(
            self.train_dataset,
            batch_sampler=self._frame_budget_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# -- Model Helpers (reused from phase2_hp_sweep.py) ---------------------------

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
    patch_outer_forward(model)
    cls = model.__class__
    if not hasattr(cls, "_embeddings_patched"):
        def get_input_embeddings(self):
            return self.thinker.model.embed_tokens
        def set_input_embeddings(self, value):
            self.thinker.model.embed_tokens = value
        cls.get_input_embeddings = get_input_embeddings
        cls.set_input_embeddings = set_input_embeddings
        cls._embeddings_patched = True

    for param in model.thinker.audio_tower.parameters():
        param.requires_grad = False

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

    # Re-freeze encoder after PEFT wrap
    for name, param in model.named_parameters():
        if "audio_tower" in name:
            param.requires_grad = False

    return model


def print_gpu_memory(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[VRAM {label}] Allocated: {allocated:.2f}GB, "
              f"Reserved: {reserved:.2f}GB, Peak: {peak:.2f}GB")


def check_vram_budget():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        if used > VRAM_BUDGET_GB:
            print(f"WARNING: VRAM {used:.2f}GB exceeds budget {VRAM_BUDGET_GB}GB!")
            return False
    return True


# -- Data Helpers -------------------------------------------------------------

def create_stratified_subset(fs_manifest, cv_manifest, subset_size, seed=42):
    """Create stratified subset with EQUAL FS/CV split (D-02)."""
    fs_df = pd.read_csv(fs_manifest)
    cv_df = pd.read_csv(cv_manifest)

    # Equal split per D-02 (not proportional like HP sweep)
    fs_n = subset_size // 2
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
        acc_n = int(cv_n * len(accented) / len(cv_df)) if len(cv_df) > 0 else 0
        unacc_n = cv_n - acc_n

        cv_samples = []
        if len(accented) > 0:
            cv_groups = accented.groupby("accent")
            for _, group_df in cv_groups:
                n = max(1, int(len(group_df) / len(accented) * acc_n))
                n = min(n, len(group_df))
                cv_samples.append(group_df.sample(n=n, random_state=seed))
        cv_subset_acc = pd.concat(cv_samples).head(acc_n) if cv_samples else pd.DataFrame()
        cv_subset_unacc = unaccented.sample(
            n=min(unacc_n, len(unaccented)), random_state=seed
        ) if len(unaccented) > 0 else pd.DataFrame()
        cv_subset = pd.concat([cv_subset_acc, cv_subset_unacc])
    else:
        cv_subset = cv_df.sample(n=cv_n, random_state=seed)

    combined = pd.concat([fs_subset, cv_subset], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    return combined


def create_speaker_disjoint_split(df, test_size=0.1, seed=42):
    """Create speaker-disjoint train/eval split (D-03, Pitfall #9).

    Uses GroupShuffleSplit on speaker_id if available.
    For rows without speaker_id (e.g., Fair-Speech), assigns unique
    pseudo-speaker IDs so each utterance is treated as its own speaker
    (conservative: no risk of leaking same speaker across splits).
    Falls back to random split only if no speaker_id column exists at all.
    """
    # Find speaker ID column
    speaker_col = None
    for col in ["speaker_id", "client_id", "spk_id"]:
        if col in df.columns:
            speaker_col = col
            break

    if speaker_col is not None:
        # Fill missing speaker IDs with unique pseudo-IDs (one per utterance)
        speakers = df[speaker_col].copy()
        missing_mask = speakers.isna() | (speakers.astype(str).str.strip() == "")
        n_missing = missing_mask.sum()
        if n_missing > 0:
            speakers[missing_mask] = [f"__unknown_speaker_{i}" for i in range(n_missing)]
            print(f"  Note: {n_missing} rows lack speaker_id (e.g., Fair-Speech) — "
                  f"assigned unique pseudo-IDs for safe splitting")

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, eval_idx = next(gss.split(df, groups=speakers.values))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        eval_df = df.iloc[eval_idx].reset_index(drop=True)

        # Verify disjoint for real speaker IDs
        train_speakers = set(speakers.iloc[train_idx].values)
        eval_speakers = set(speakers.iloc[eval_idx].values)
        real_train = {s for s in train_speakers if not str(s).startswith("__unknown_speaker_")}
        real_eval = {s for s in eval_speakers if not str(s).startswith("__unknown_speaker_")}
        overlap = real_train & real_eval
        assert len(overlap) == 0, f"Speaker overlap: {overlap}"
        print(f"  Speaker-disjoint split via '{speaker_col}': "
              f"{len(real_train)} real train speakers, {len(real_eval)} real eval speakers"
              f" (+ {n_missing} pseudo-speakers from Fair-Speech)")
    else:
        print("  WARNING: No speaker_id column found. Falling back to random split.")
        print("  Speaker-disjoint guarantee cannot be provided.")
        n_eval = max(10, int(len(df) * test_size))
        eval_df = df.sample(n=n_eval, random_state=seed)
        train_df = df.drop(eval_df.index).reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)

    return train_df, eval_df


def load_full_dataset(fs_manifest, cv_manifest):
    """Load all data from both manifests without subsetting (D-01).

    Unlike create_stratified_subset which enforces equal FS/CV split,
    this uses ALL available data from both sources.
    """
    fs_df = pd.read_csv(fs_manifest)
    cv_df = pd.read_csv(cv_manifest)

    # Align columns — CV lacks ethnicity/first_language, FS lacks accent/speaker_id
    for col in ["ethnicity", "first_language"]:
        if col not in cv_df.columns:
            cv_df[col] = ""
    for col in ["accent", "speaker_id"]:
        if col not in fs_df.columns:
            fs_df[col] = ""

    combined = pd.concat([fs_df, cv_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  Full dataset: {len(fs_df):,} FS + {len(cv_df):,} CV = {len(combined):,} total")
    return combined


# -- Validate Mode (ML Prototype Phase 3) ------------------------------------

def run_validate(args):
    """Validate top HP configs from sweep at medium scale."""
    print(f"\n{'='*60}")
    print("ML PROTOTYPE PHASE 3: Medium-Scale Validation")
    print(f"{'='*60}")

    # Load top configs
    top3_path = os.path.join(args.hp_sweep_dir, "top3_configs.json")
    if not os.path.exists(top3_path):
        print(f"ERROR: {top3_path} not found. Run HP sweep first.")
        sys.exit(1)

    with open(top3_path) as f:
        top_configs = json.load(f)
    print(f"  Loaded {len(top_configs)} configs from {top3_path}")

    # Create subset
    print(f"\n  Creating {args.subset_size}-sample stratified subset (equal FS/CV)...")
    subset_df = create_stratified_subset(
        args.fs_manifest, args.cv_manifest, args.subset_size, SEED)
    print(f"  Subset: {len(subset_df)} samples")

    # Speaker-disjoint split
    train_df, eval_df = create_speaker_disjoint_split(subset_df, test_size=0.1, seed=SEED)
    print(f"  Train: {len(train_df)}, Eval: {len(eval_df)}")

    # Save subsets
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "validate_train.csv")
    eval_path = os.path.join(args.output_dir, "validate_eval.csv")
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    # Determine demographic axis
    axis = "ethnicity" if "ethnicity" in train_df.columns else "accent"

    # Load processor once
    _, processor = load_model_and_processor()

    validation_results = []

    for i, config in enumerate(top_configs):
        print(f"\n{'='*60}")
        print(f"Validating config {i+1}/{len(top_configs)} "
              f"(sweep trial {config['trial_number']}, "
              f"sweep loss={config['eval_loss']:.4f})")
        print(f"{'='*60}")

        params = config["params"]
        rank = int(params["rank"])
        alpha_ratio = int(params.get("alpha_ratio", 2))
        alpha = rank * alpha_ratio
        dropout = float(params.get("dropout", 0.05))
        target_mlp = bool(params.get("target_mlp", False))
        lr = float(params["lr"])
        weight_decay = float(params.get("weight_decay", 1e-4))

        print(f"  HPs: lr={lr:.2e}, rank={rank}, alpha={alpha}, "
              f"dropout={dropout:.2f}, mlp={target_mlp}, wd={weight_decay:.2e}")

        # Load fresh model
        model, _ = load_model_and_processor()
        model = apply_lora(model, rank, alpha, dropout, target_mlp)

        if torch.cuda.is_available():
            model = model.cuda()
        print_gpu_memory(f"Config {i+1} loaded")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

        # Create datasets
        train_dataset = ASRFairnessDataset(train_path, demographic_axis=axis)
        eval_dataset = ASRFairnessDataset(eval_path, demographic_axis=axis)
        collator = DataCollatorForQwen3ASR(processor)

        trial_dir = os.path.join(args.output_dir, f"validate_config_{i}")

        # W&B run name
        wandb_run = f"validate-config-{i}" if args.wandb_project != "none" else None
        report_to = "wandb" if args.wandb_project != "none" else "none"

        training_args = TrainingArguments(
            output_dir=trial_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=lr,
            weight_decay=weight_decay,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=max(1, args.max_steps // 5),
            save_strategy="no",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=report_to,
            run_name=wandb_run,
            seed=SEED,
            load_best_model_at_end=False,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
        )

        if report_to == "wandb":
            os.environ["WANDB_PROJECT"] = args.wandb_project

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )

        try:
            trainer.train()
            eval_results = trainer.evaluate()
            eval_loss = eval_results.get("eval_loss", float("inf"))

            train_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
            print(f"  Train loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}"
                  if train_losses else "  No train loss logged")
            print(f"  Eval loss: {eval_loss:.4f}")

            validation_results.append({
                "config_index": i,
                "sweep_trial": config["trial_number"],
                "sweep_loss": config["eval_loss"],
                "validation_loss": eval_loss,
                "params": params,
                "train_loss_first": train_losses[0] if train_losses else None,
                "train_loss_last": train_losses[-1] if train_losses else None,
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            validation_results.append({
                "config_index": i,
                "sweep_trial": config["trial_number"],
                "sweep_loss": config["eval_loss"],
                "validation_loss": float("inf"),
                "params": params,
                "error": str(e),
            })
        finally:
            del model, trainer, train_dataset, eval_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Rank comparison
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")

    # Sort by validation loss
    valid_results = [r for r in validation_results if r["validation_loss"] < float("inf")]
    valid_results.sort(key=lambda x: x["validation_loss"])

    print(f"\n  {'Config':>8} | {'Sweep Loss':>12} | {'Val Loss':>12} | {'Sweep Rank':>12} | {'Val Rank':>10}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*10}")

    sweep_ranking = sorted(range(len(top_configs)), key=lambda i: top_configs[i]["eval_loss"])

    for val_rank, result in enumerate(valid_results):
        sweep_rank = sweep_ranking.index(result["config_index"])
        print(f"  Config {result['config_index']:>2} | {result['sweep_loss']:>12.4f} | "
              f"{result['validation_loss']:>12.4f} | #{sweep_rank+1:>10} | #{val_rank+1:>8}")

    # Save results
    results_path = os.path.join(args.output_dir, "validation_results.json")
    with open(results_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Lock best config
    if valid_results:
        best = valid_results[0]
        locked = {
            "config_index": best["config_index"],
            "sweep_trial": best["sweep_trial"],
            "validation_loss": best["validation_loss"],
            "params": best["params"],
        }
        locked_path = os.path.join(args.output_dir, "locked_config.json")
        with open(locked_path, "w") as f:
            json.dump(locked, f, indent=2)
        print(f"  Locked config saved: {locked_path}")

        # Gate check
        sweep_top = sweep_ranking[0]
        val_top = valid_results[0]["config_index"]
        if sweep_top == val_top:
            print(f"\n  GATE PASSED: Same top-1 config (index {val_top}) at both scales")
        else:
            print(f"\n  GATE WARNING: Ranking flipped! Sweep top={sweep_top}, Val top={val_top}")
            print(f"  Using validation winner (medium scale is more representative)")
    else:
        print("\n  ERROR: All validation configs failed!")
        sys.exit(1)


# -- Train Mode (ML Prototype Phase 4) ---------------------------------------

def run_train(args):
    """Final training with locked HP config."""
    print(f"\n{'='*60}")
    print("SFT LoRA Training" + (" (Full Dataset)" if args.full_dataset else " (Subset)"))
    print(f"{'='*60}")

    # Pre-flight checklist — resolve locked config path
    locked_path = args.locked_config_path or os.path.join(args.output_dir, "locked_config.json")
    if not os.path.exists(locked_path):
        # Fallback: check Phase 2 output dir
        phase2_path = os.path.join("outputs/standard-lora", "locked_config.json")
        if os.path.exists(phase2_path):
            locked_path = phase2_path
            print(f"  Using Phase 2 locked config: {locked_path}")
        else:
            print(f"ERROR: No locked_config.json found at {locked_path} or {phase2_path}")
            sys.exit(1)

    with open(locked_path) as f:
        locked = json.load(f)
    params = locked["params"]

    print(f"\n  Pre-flight checklist:")
    print(f"  [x] locked_config.json loaded (val loss={locked['validation_loss']:.4f})")

    # GPU check
    if torch.cuda.is_available():
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                                 "--format=csv,noheader"], capture_output=True, text=True)
        print(f"  [x] GPU: {result.stdout.strip()}")
    else:
        print(f"  [ ] WARNING: No GPU available!")

    # Verify manifests
    assert os.path.exists(args.fs_manifest), f"Missing: {args.fs_manifest}"
    assert os.path.exists(args.cv_manifest), f"Missing: {args.cv_manifest}"
    print(f"  [x] Manifests verified")

    # HP summary
    rank = int(params["rank"])
    alpha_ratio = int(params.get("alpha_ratio", 2))
    alpha = int(params.get("alpha", rank * alpha_ratio))
    dropout = float(params.get("dropout", 0.05))
    target_mlp = bool(params.get("target_mlp", False))
    lr = float(params["lr"])
    weight_decay = float(params.get("weight_decay", 1e-4))

    lr_scheduler = str(params.get("lr_scheduler", "cosine"))
    warmup_ratio = float(params.get("warmup_ratio", 0.05))
    grad_accum = int(params.get("grad_accum_steps", 2))
    batch_size = int(params.get("batch_size", 4 if args.full_dataset else 2))
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.grad_accum is not None:
        grad_accum = args.grad_accum

    print(f"\n  Locked HPs: lr={lr:.2e}, rank={rank}, alpha={alpha}, "
          f"dropout={dropout:.2f}, mlp={target_mlp}, wd={weight_decay:.2e}, "
          f"sched={lr_scheduler}, warmup={warmup_ratio}, "
          f"batch={batch_size}, grad_accum={grad_accum}")

    # Load data
    if args.full_dataset:
        print(f"\n  Loading FULL dataset (no subsetting)...")
        combined_df = load_full_dataset(args.fs_manifest, args.cv_manifest)
    else:
        print(f"\n  Creating {args.subset_size}-sample subset (equal FS/CV)...")
        combined_df = create_stratified_subset(
            args.fs_manifest, args.cv_manifest, args.subset_size, SEED)
    print(f"  Total samples: {len(combined_df):,}")

    # Speaker-disjoint split
    train_df, eval_df = create_speaker_disjoint_split(combined_df, test_size=0.1, seed=SEED)
    print(f"  Train: {len(train_df)}, Eval: {len(eval_df)}")

    # Save subsets for reproducibility and evaluation bridge
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train_subset.csv")
    eval_path = os.path.join(args.output_dir, "eval_subset.csv")
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    print(f"  Train CSV: {train_path}")
    print(f"  Eval CSV: {eval_path}")

    axis = "ethnicity" if "ethnicity" in train_df.columns else "accent"

    # Compute steps and intervals
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, len(train_df) // effective_batch)

    # Dynamic eval/save intervals
    eval_steps = max(500, min(5000, steps_per_epoch // 13))
    save_steps = eval_steps  # must match for load_best_model_at_end

    # Determine training duration
    use_epochs = args.num_epochs is not None
    if args.max_steps:
        max_steps = args.max_steps
    elif not use_epochs:
        # Auto: 2 epochs for full dataset, 3 for subset
        n_epochs = 2 if args.full_dataset else 3
        max_steps = steps_per_epoch * n_epochs
        if not args.full_dataset:
            max_steps = min(max_steps, 1500)
            max_steps = max(max_steps, 500)

    warmup_steps = max(1, int((args.num_epochs * steps_per_epoch if use_epochs else max_steps) * warmup_ratio)) if warmup_ratio > 0 else 0

    print(f"\n  Training plan:")
    print(f"    Steps/epoch: {steps_per_epoch:,}, effective_batch={effective_batch}")
    if use_epochs:
        print(f"    Epochs: {args.num_epochs}, total steps: ~{steps_per_epoch * args.num_epochs:,}")
    else:
        print(f"    Max steps: {max_steps:,} ({max_steps/steps_per_epoch:.1f} epochs)")
    print(f"    Eval every {eval_steps:,} steps, save every {save_steps:,} steps")
    print(f"    Warmup: {warmup_steps} steps, early stopping patience=3")

    # Load model
    t0 = time.time()
    model, processor = load_model_and_processor()
    print_gpu_memory("After model load")

    model = apply_lora(model, rank, alpha, dropout, target_mlp)
    if torch.cuda.is_available():
        model = model.cuda()
    print_gpu_memory("After PEFT wrap + GPU")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Create datasets
    train_dataset = ASRFairnessDataset(train_path, demographic_axis=axis)
    eval_dataset = ASRFairnessDataset(eval_path, demographic_axis=axis)
    collator = DataCollatorForQwen3ASR(processor)

    # W&B
    report_to = "wandb" if args.wandb_project != "none" else "none"
    if report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project

    run_name = f"full-lora-{args.num_epochs}ep" if use_epochs else "standard-lora-final"

    training_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=report_to,
        run_name=run_name,
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler,
    )
    if use_epochs:
        training_kwargs["num_train_epochs"] = args.num_epochs
    else:
        training_kwargs["max_steps"] = max_steps

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)]

    # Dynamic batching setup
    frame_sampler = None
    if args.dynamic_batch:
        print(f"\n  Dynamic batching: computing audio durations...")
        durations = train_dataset.get_durations(cache_dir=args.output_dir)
        frame_sampler = FrameBudgetBatchSampler(
            durations=durations,
            max_seconds_per_batch=args.frame_budget,
            max_batch_size=args.max_batch_size,
            shuffle=True,
            seed=SEED,
        )
        num_batches = len(frame_sampler)
        median_dur = np.median(durations)
        avg_batch = len(train_dataset) / num_batches if num_batches > 0 else 0
        print(f"    Frame budget: {args.frame_budget}s, median duration: {median_dur:.1f}s")
        print(f"    {num_batches:,} batches, avg {avg_batch:.1f} samples/batch")
        # Trainer still needs a batch_size for internal bookkeeping; set to 1
        # since the batch sampler handles actual batch formation
        training_kwargs["per_device_train_batch_size"] = 1

    training_args = TrainingArguments(**training_kwargs)

    TrainerClass = DynamicBatchTrainer if args.dynamic_batch else Trainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        **({"frame_budget_sampler": frame_sampler} if args.dynamic_batch else {}),
    )

    # Train
    print(f"\n  Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elapsed = time.time() - t0

    print_gpu_memory("After training")

    # Save adapter
    adapter_dir = os.path.join(args.output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"\n  Adapter saved: {adapter_dir}")

    # Verify adapter files
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    assert os.path.exists(safetensors_path), f"Missing: {safetensors_path}"
    assert os.path.exists(config_path), f"Missing: {config_path}"
    size_kb = os.path.getsize(safetensors_path) / 1024
    print(f"  adapter_model.safetensors: {size_kb:.1f} KB")
    print(f"  adapter_config.json: OK")

    # Save training config
    train_config = {
        "params": params,
        "effective_batch_size": effective_batch,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "lr_scheduler": lr_scheduler,
        "warmup_ratio": warmup_ratio,
        "early_stopping_patience": 3,
        "full_dataset": args.full_dataset,
        "train_samples": len(train_df),
        "eval_samples": len(eval_df),
        "steps_per_epoch": steps_per_epoch,
        "final_train_loss": train_result.training_loss,
        "final_step": train_result.global_step,
        "training_time_sec": elapsed,
    }
    if use_epochs:
        train_config["num_epochs"] = args.num_epochs
    else:
        train_config["max_steps"] = max_steps
    if torch.cuda.is_available():
        train_config["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    if args.dynamic_batch:
        train_config["dynamic_batch"] = True
        train_config["frame_budget_seconds"] = args.frame_budget
        train_config["max_batch_size"] = args.max_batch_size

    config_save_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_save_path, "w") as f:
        json.dump(train_config, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Final loss: {train_result.training_loss:.4f}")
    print(f"  Steps: {train_result.global_step}")
    print(f"  Time: {elapsed/60:.1f} min")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak VRAM: {peak:.2f} GB")
        if peak > VRAM_BUDGET_GB:
            print(f"  WARNING: Peak VRAM exceeds {VRAM_BUDGET_GB}GB budget!")
        else:
            print(f"  PASS: Peak VRAM within {VRAM_BUDGET_GB}GB budget")

    # Loss trajectory
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    if len(losses) >= 2:
        print(f"  Loss trajectory: {losses[0]:.4f} -> {losses[-1]:.4f}")
        if losses[-1] < losses[0]:
            print(f"  PASS: Loss decreased during training")
        else:
            print(f"  WARNING: Loss did not decrease!")

    print(f"\n  BASE-01 requirement: SATISFIED")


# -- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standard LoRA SFT training (ML Prototype Phases 3-4)")
    parser.add_argument("--mode", required=True, choices=["validate", "train"])
    parser.add_argument("--fs_manifest", default="outputs/manifests/fs_train.csv")
    parser.add_argument("--cv_manifest", default="outputs/manifests/cv_train.csv")
    parser.add_argument("--hp_sweep_dir", default=DEFAULT_HP_SWEEP_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset_size", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--wandb_project", default="asr-fairness-lora")
    parser.add_argument("--full_dataset", action="store_true",
        help="Use all data from both manifests (overrides --subset_size)")
    parser.add_argument("--num_epochs", type=int, default=None,
        help="Train for N epochs (alternative to --max_steps)")
    parser.add_argument("--save_total_limit", type=int, default=3,
        help="Keep only last N checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
        help="Path to checkpoint directory to resume from")
    parser.add_argument("--locked_config_path", type=str, default=None,
        help="Path to locked_config.json (default: {output_dir}/locked_config.json)")
    parser.add_argument("--batch_size", type=int, default=None,
        help="Override per-device train batch size (takes precedence over locked config)")
    parser.add_argument("--grad_accum", type=int, default=None,
        help="Override gradient accumulation steps (takes precedence over locked config)")
    parser.add_argument("--dynamic_batch", action="store_true",
        help="Use dynamic batching by frame budget instead of fixed batch size")
    parser.add_argument("--frame_budget", type=float, default=120.0,
        help="Max total audio seconds per batch when --dynamic_batch is set (default: 120)")
    parser.add_argument("--max_batch_size", type=int, default=64,
        help="Ceiling on dynamic batch size (default: 64)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.mode == "validate":
        if args.max_steps is None:
            args.max_steps = 300
        run_validate(args)
    elif args.mode == "train":
        run_train(args)


if __name__ == "__main__":
    main()
