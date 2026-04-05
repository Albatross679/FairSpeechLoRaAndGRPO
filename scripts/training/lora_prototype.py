"""
LoRA prototype training for Qwen3-ASR-1.7B.

Attaches LoRA to decoder self-attention, runs 50-100 training steps
on a 200-sample stratified subset, validates forward pass and VRAM.

This script is the critical risk validation for Phase 1.1: can LoRA
attach to Qwen3-ASR decoder and produce valid training within the
16GB VRAM budget on RTX A4000?

Usage:
    python scripts/training/lora_prototype.py \
        --fs_manifest /path/to/fs_train.csv \
        --cv_manifest /path/to/cv_train.csv \
        --output_dir outputs/lora-prototype \
        --num_steps 100 \
        --batch_size 2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer

from scripts.training.data_loader import ASRFairnessDataset
from scripts.training.data_collator import DataCollatorForQwen3ASR


# -- Constants ----------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_OUTPUT_DIR = "outputs/lora-prototype"
DEFAULT_NUM_STEPS = 100
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_SEED = 42

# LoRA hyperparameters (D-01, D-02: attention-only, rank 16)
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Subset sizes (D-05, D-06, D-07: 100 FS + 100 CV = 200 total)
FS_SUBSET_SIZE = 100
CV_SUBSET_SIZE = 100


# -- VRAM Profiling -----------------------------------------------------------

def print_gpu_memory(label=""):
    """Print current GPU memory usage at key training checkpoints.

    Called at 5 points: after model load, after PEFT wrap, after first
    forward, after first backward, and at end of training (D-10).
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[VRAM {label}] Allocated: {allocated:.2f}GB, "
              f"Reserved: {reserved:.2f}GB, Peak: {peak:.2f}GB")
    else:
        print(f"[VRAM {label}] No CUDA device available")


# -- Forward Patch (Pattern 2 from RESEARCH.md) -------------------------------

def patch_outer_forward(model):
    """Patch Qwen3ASRForConditionalGeneration to delegate forward to thinker.

    The outer model only has generate(), not forward(). HuggingFace Trainer
    requires forward() for training loss computation. This patch delegates
    to thinker.forward() which has the full forward + loss computation.

    Based on official SFT script pattern (D-15, RESEARCH.md Pattern 2).
    """
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no .thinker.forward. "
            "Ensure model is Qwen3ASRForConditionalGeneration."
        )

    def forward(self, input_ids=None, attention_mask=None,
                input_features=None, feature_attention_mask=None,
                labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


# -- Data Subsetting ----------------------------------------------------------

def create_stratified_subset(fs_manifest, cv_manifest, output_dir, seed=42):
    """Create a 200-sample stratified subset from Fair-Speech + Common Voice.

    Samples 100 utterances from Fair-Speech (stratified by ethnicity) and
    100 from Common Voice (stratified by accent), combining into a single
    prototype subset CSV (D-05, D-06, D-07).

    Args:
        fs_manifest: Path to Fair-Speech train manifest CSV.
        cv_manifest: Path to Common Voice train manifest CSV.
        output_dir: Directory to save the subset CSV.
        seed: Random seed for reproducibility.

    Returns:
        Path to the combined subset CSV.
    """
    print(f"\n{'='*60}")
    print("Creating stratified prototype subset")
    print(f"{'='*60}")

    # -- Fair-Speech subset (stratified by ethnicity) -------------------------
    print(f"\nLoading Fair-Speech manifest: {fs_manifest}")
    fs_df = pd.read_csv(fs_manifest)
    print(f"  Total rows: {len(fs_df):,}")

    if "ethnicity" in fs_df.columns:
        fs_groups = fs_df.groupby("ethnicity")
        group_counts = fs_groups.size()
        print(f"  Ethnicity groups: {len(group_counts)}")
        for group, count in group_counts.items():
            print(f"    {group}: {count:,}")

        # Proportional stratified sampling
        fs_samples = []
        total_available = len(fs_df)
        for group_name, group_df in fs_groups:
            proportion = len(group_df) / total_available
            n_sample = max(1, int(proportion * FS_SUBSET_SIZE))
            n_sample = min(n_sample, len(group_df))
            sampled = group_df.sample(n=n_sample, random_state=seed)
            fs_samples.append(sampled)

        fs_subset = pd.concat(fs_samples, ignore_index=True)
        # Trim or pad to exact target size
        if len(fs_subset) > FS_SUBSET_SIZE:
            fs_subset = fs_subset.sample(n=FS_SUBSET_SIZE, random_state=seed)
        elif len(fs_subset) < FS_SUBSET_SIZE:
            remaining = FS_SUBSET_SIZE - len(fs_subset)
            extra_pool = fs_df[~fs_df.index.isin(fs_subset.index)]
            if len(extra_pool) >= remaining:
                extra = extra_pool.sample(n=remaining, random_state=seed)
                fs_subset = pd.concat([fs_subset, extra], ignore_index=True)
    else:
        print("  WARNING: No 'ethnicity' column, sampling randomly")
        fs_subset = fs_df.sample(
            n=min(FS_SUBSET_SIZE, len(fs_df)), random_state=seed
        )

    print(f"  Fair-Speech subset: {len(fs_subset)} samples")

    # -- Common Voice subset (stratified by accent) ---------------------------
    print(f"\nLoading Common Voice manifest: {cv_manifest}")
    cv_df = pd.read_csv(cv_manifest)
    print(f"  Total rows: {len(cv_df):,}")

    if "accent" in cv_df.columns:
        cv_groups = cv_df.groupby("accent")
        group_counts = cv_groups.size()
        print(f"  Accent groups: {len(group_counts)}")
        for group, count in group_counts.items():
            print(f"    {group}: {count:,}")

        # Proportional stratified sampling
        cv_samples = []
        total_available = len(cv_df)
        for group_name, group_df in cv_groups:
            proportion = len(group_df) / total_available
            n_sample = max(1, int(proportion * CV_SUBSET_SIZE))
            n_sample = min(n_sample, len(group_df))
            sampled = group_df.sample(n=n_sample, random_state=seed)
            cv_samples.append(sampled)

        cv_subset = pd.concat(cv_samples, ignore_index=True)
        if len(cv_subset) > CV_SUBSET_SIZE:
            cv_subset = cv_subset.sample(n=CV_SUBSET_SIZE, random_state=seed)
        elif len(cv_subset) < CV_SUBSET_SIZE:
            remaining = CV_SUBSET_SIZE - len(cv_subset)
            extra_pool = cv_df[~cv_df.index.isin(cv_subset.index)]
            if len(extra_pool) >= remaining:
                extra = extra_pool.sample(n=remaining, random_state=seed)
                cv_subset = pd.concat([cv_subset, extra], ignore_index=True)
    else:
        print("  WARNING: No 'accent' column, sampling randomly")
        cv_subset = cv_df.sample(
            n=min(CV_SUBSET_SIZE, len(cv_df)), random_state=seed
        )

    print(f"  Common Voice subset: {len(cv_subset)} samples")

    # -- Combine and save -----------------------------------------------------
    combined = pd.concat([fs_subset, cv_subset], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    subset_path = os.path.join(output_dir, "prototype_subset.csv")
    combined.to_csv(subset_path, index=False)

    print(f"\nCombined subset: {len(combined)} samples")
    print(f"Saved to: {subset_path}")

    return subset_path


# -- Model Loading and Setup --------------------------------------------------

def load_model_and_processor():
    """Load Qwen3-ASR-1.7B via qwen-asr wrapper and extract HF model.

    Uses Pattern 1 from RESEARCH.md: load via qwen-asr then extract
    the inner Qwen3ASRForConditionalGeneration model and processor.

    Returns:
        Tuple of (model, processor).
    """
    print(f"\n{'='*60}")
    print(f"Loading model: {MODEL_ID}")
    print(f"{'='*60}")

    from qwen_asr import Qwen3ASRModel

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=None,
    )
    model = asr_wrapper.model          # Qwen3ASRForConditionalGeneration
    processor = asr_wrapper.processor   # Qwen3ASRProcessor

    print(f"  Model class: {model.__class__.__name__}")
    print(f"  Processor class: {processor.__class__.__name__}")
    print_gpu_memory("After model load")

    return model, processor


def setup_lora(model):
    """Apply LoRA adapters to decoder self-attention projections.

    Freezes the audio encoder (D-03), patches the forward method (D-15),
    enables gradient checkpointing with use_reentrant=False (D-14, INFRA-04),
    and applies LoRA to decoder attention-only modules (D-01, D-02).

    Per D-04 deferral, MLP modules are NOT included in target_modules.
    LoRA targets only: q_proj, k_proj, v_proj, o_proj in decoder layers.

    Args:
        model: Qwen3ASRForConditionalGeneration from load_model_and_processor.

    Returns:
        PEFT-wrapped model with LoRA adapters.
    """
    print(f"\n{'='*60}")
    print("Setting up LoRA adapters")
    print(f"{'='*60}")

    # Step 1: Patch forward for Trainer compatibility (Pattern 2)
    print("\n  Patching outer forward...")
    patch_outer_forward(model)
    print("  Forward patched successfully")

    # Step 2: Freeze encoder explicitly (D-03)
    print("\n  Freezing audio encoder...")
    encoder_params = 0
    for param in model.thinker.audio_tower.parameters():
        param.requires_grad = False
        encoder_params += param.numel()
    print(f"  Frozen encoder parameters: {encoder_params:,}")

    # Step 3: Enable gradient checkpointing (D-14, INFRA-04, Pattern 4)
    # use_reentrant=False is critical for frozen params (Pitfall 4)
    print("\n  Enabling gradient checkpointing (use_reentrant=False)...")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    print("  Gradient checkpointing enabled")

    # Step 4: Apply LoRA to decoder attention-only (D-01, D-02, Pattern 3)
    # NOTE: MLP modules are deliberately excluded per D-01 (attention-only)
    # and D-04 (MLP LoRA deferred to later phase)
    print(f"\n  Applying LoRA (r={LORA_RANK}, alpha={LORA_ALPHA}, "
          f"dropout={LORA_DROPOUT})...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            r"thinker\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print_gpu_memory("After PEFT wrap")

    # Verify encoder is still frozen after PEFT wrap
    encoder_grad_params = sum(
        1 for p in model.base_model.model.thinker.audio_tower.parameters()
        if p.requires_grad
    )
    if encoder_grad_params > 0:
        print(f"  WARNING: {encoder_grad_params} encoder params have "
              f"requires_grad=True after PEFT wrap!")
    else:
        print("  Verified: encoder remains frozen after PEFT wrap")

    return model


# -- Training -----------------------------------------------------------------

def train(model, processor, dataset, args):
    """Run HuggingFace Trainer for LoRA prototype training.

    Sets up TrainingArguments with bf16, gradient checkpointing (already
    enabled on model), small batch size, and the custom data collator.

    Args:
        model: PEFT-wrapped model from setup_lora.
        processor: Qwen3ASRProcessor.
        dataset: ASRFairnessDataset for the prototype subset.
        args: Parsed CLI arguments.

    Returns:
        Trainer instance (for loss history access).
    """
    print(f"\n{'='*60}")
    print("Starting LoRA prototype training")
    print(f"{'='*60}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output dir: {args.output_dir}")

    data_collator = DataCollatorForQwen3ASR(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.num_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        remove_unused_columns=False,   # Critical for custom collator
        dataloader_pin_memory=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n  Starting training loop...")
    train_result = trainer.train()

    print(f"\n{'='*60}")
    print("Training complete")
    print(f"{'='*60}")
    print(f"  Training loss: {train_result.training_loss:.4f}")
    print(f"  Global steps: {train_result.global_step}")

    # Print loss history from trainer state
    if trainer.state.log_history:
        print("\n  Loss history:")
        for entry in trainer.state.log_history:
            if "loss" in entry:
                step = entry.get("step", "?")
                loss = entry["loss"]
                print(f"    Step {step}: loss={loss:.4f}")

    print_gpu_memory("After training")

    return trainer


# -- CLI ----------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA prototype training for Qwen3-ASR-1.7B"
    )
    parser.add_argument(
        "--fs_manifest", required=True,
        help="Path to Fair-Speech train manifest CSV"
    )
    parser.add_argument(
        "--cv_manifest", required=True,
        help="Path to Common Voice train manifest CSV"
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--num_steps", type=int, default=DEFAULT_NUM_STEPS,
        help=f"Number of training steps (default: {DEFAULT_NUM_STEPS})"
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Per-device batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )

    return parser.parse_args()


# -- Main ---------------------------------------------------------------------

def main():
    """Run the LoRA prototype training pipeline.

    Steps:
        1. Parse arguments and set random seeds
        2. Create stratified 200-sample subset (100 FS + 100 CV)
        3. Load Qwen3-ASR model and processor
        4. Set up LoRA adapters on decoder attention
        5. Create dataset and train via HF Trainer
        6. Save adapter weights
        7. Print VRAM summary
    """
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print("LoRA Prototype Training for Qwen3-ASR-1.7B")
    print(f"{'='*60}")

    # Step 1: Create stratified subset
    subset_path = create_stratified_subset(
        fs_manifest=args.fs_manifest,
        cv_manifest=args.cv_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Step 2: Load model and processor
    model, processor = load_model_and_processor()

    # Step 3: Set up LoRA
    model = setup_lora(model)

    # Step 4: Create dataset from subset
    # Use "ethnicity" as axis since the combined CSV has both ethnicity and accent
    # The ASRFairnessDataset will use it for demographic tracking but the
    # collator doesn't need it for training (it's for GRPO reward in Phase 2)
    print(f"\n{'='*60}")
    print("Loading prototype dataset")
    print(f"{'='*60}")

    # Determine which demographic axis is available in the combined subset
    subset_df = pd.read_csv(subset_path)
    if "ethnicity" in subset_df.columns:
        axis = "ethnicity"
    elif "accent" in subset_df.columns:
        axis = "accent"
    else:
        axis = None

    dataset = ASRFairnessDataset(
        manifest_csv=subset_path,
        demographic_axis=axis,
    )
    print(f"  Dataset size: {len(dataset)}")

    # Step 5: Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print_gpu_memory("After model to GPU")

    # Step 6: Train
    trainer = train(model, processor, dataset, args)

    # Step 7: Save adapter
    print(f"\n{'='*60}")
    print("Saving adapter weights")
    print(f"{'='*60}")

    adapter_dir = os.path.join(args.output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"  Adapter saved to: {adapter_dir}")

    # Step 8: Final VRAM summary
    print(f"\n{'='*60}")
    print("Final VRAM Summary")
    print(f"{'='*60}")
    print_gpu_memory("End of training")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n  Peak VRAM: {peak_gb:.2f}GB")
        if peak_gb > 14.0:
            print("  WARNING: Peak VRAM exceeds 14GB budget!")
        else:
            print("  PASS: Peak VRAM within 14GB budget")

    # Print final loss summary
    if trainer.state.log_history:
        losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
        if len(losses) >= 2:
            first_loss = losses[0]
            last_loss = losses[-1]
            print(f"\n  Loss trajectory: {first_loss:.4f} -> {last_loss:.4f}")
            if last_loss < first_loss:
                print("  PASS: Loss decreased during training")
            else:
                print("  WARNING: Loss did not decrease!")

    print(f"\n{'='*60}")
    print("LoRA prototype training complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
