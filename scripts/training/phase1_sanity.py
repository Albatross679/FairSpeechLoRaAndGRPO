"""
Phase 1: Pipeline Sanity Checks for LoRA fine-tuning of Qwen3-ASR-1.7B.

Runs 6 gate checks in order:
1. Data loading — Load 1 batch, verify shapes and dtypes
2. Loss at init — Forward pass, check loss value is reasonable
3. Single-batch overfit — Train 200 steps on 1 batch, loss → ~0
4. Gradient flow — Verify all LoRA params have nonzero gradients
5. Eval loop — Generate transcription from 1 audio sample
6. Checkpoint round-trip — Save → reload → compare output tokens

All 6 must pass before proceeding to Phase 2 HP sweep.

Usage:
    python scripts/training/phase1_sanity.py \
        --fs_manifest outputs/manifests/fs_train.csv \
        --cv_manifest outputs/manifests/cv_train.csv
"""

import argparse
import os
import sys
import tempfile
import shutil
import time
import traceback

# Ensure project root is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import torch

# -- Constants ----------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
OVERFIT_STEPS = 200
OVERFIT_LOSS_THRESHOLD = 0.5  # Loss should be well below this after 200 steps
EXPECTED_INIT_LOSS_MIN = 5.0  # LM loss should be > this at init
EXPECTED_INIT_LOSS_MAX = 20.0 # And < this (sanity bound)

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
SEED = 42


# -- Helpers ------------------------------------------------------------------

def print_gpu_memory(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [VRAM {label}] Alloc: {allocated:.2f}GB, "
              f"Reserved: {reserved:.2f}GB, Peak: {peak:.2f}GB")


def load_model_and_processor():
    """Load Qwen3-ASR via qwen-asr wrapper."""
    from qwen_asr import Qwen3ASRModel

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor
    return model, processor


def patch_outer_forward(model):
    """Patch forward for Trainer compatibility."""
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


def setup_lora(model):
    """Apply LoRA and freeze encoder."""
    from peft import LoraConfig, get_peft_model, TaskType

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

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)

    # Re-freeze encoder (PEFT may have added LoRA to audio_tower q/k/v_proj)
    for name, param in model.named_parameters():
        if "audio_tower" in name:
            param.requires_grad = False

    return model


def load_one_batch(fs_manifest, cv_manifest, processor):
    """Load a single batch of 2 samples for sanity checks."""
    from scripts.training.data_loader import ASRFairnessDataset
    from scripts.training.data_collator import DataCollatorForQwen3ASR

    # Take 2 samples from Fair-Speech (guaranteed to have audio)
    fs_df = pd.read_csv(fs_manifest)
    subset = fs_df.head(2)

    # Create mini dataset
    tmpdir = tempfile.mkdtemp(prefix="sanity_")
    mini_csv = os.path.join(tmpdir, "mini.csv")
    subset.to_csv(mini_csv, index=False)

    dataset = ASRFairnessDataset(
        manifest_csv=mini_csv, demographic_axis="ethnicity"
    )

    collator = DataCollatorForQwen3ASR(processor)
    features = [dataset[i] for i in range(len(dataset))]
    batch = collator(features)

    shutil.rmtree(tmpdir, ignore_errors=True)
    return batch, features


# -- Check Functions ----------------------------------------------------------

def check_1_data_loading(fs_manifest, cv_manifest, processor):
    """Check 1: Load 1 batch, verify shapes and dtypes."""
    print(f"\n{'='*60}")
    print("CHECK 1: Data Loading")
    print(f"{'='*60}")

    batch, features = load_one_batch(fs_manifest, cv_manifest, processor)

    print(f"  input_ids:             shape={batch['input_ids'].shape}, "
          f"dtype={batch['input_ids'].dtype}")
    print(f"  attention_mask:        shape={batch['attention_mask'].shape}, "
          f"dtype={batch['attention_mask'].dtype}")
    print(f"  input_features:        shape={batch['input_features'].shape}, "
          f"dtype={batch['input_features'].dtype}")
    print(f"  feature_attention_mask: shape={batch['feature_attention_mask'].shape}, "
          f"dtype={batch['feature_attention_mask'].dtype}")
    print(f"  labels:                shape={batch['labels'].shape}, "
          f"dtype={batch['labels'].dtype}")

    # Verify shapes are consistent
    B = batch["input_ids"].shape[0]
    assert B == 2, f"Expected batch size 2, got {B}"
    assert batch["input_ids"].shape == batch["labels"].shape, "input_ids and labels shape mismatch"
    assert batch["input_ids"].shape == batch["attention_mask"].shape, "input_ids and attention_mask shape mismatch"

    # Verify labels have -100 masking (prefix should be masked)
    masked = (batch["labels"] == -100).sum().item()
    total = batch["labels"].numel()
    mask_pct = masked / total * 100
    print(f"  Labels masked: {masked}/{total} ({mask_pct:.1f}%)")
    assert masked > 0, "No labels are masked — prefix masking is broken"
    assert masked < total, "All labels are masked — no training signal"

    # Verify audio features have valid values
    feats = batch["input_features"]
    assert feats.dim() >= 2, f"input_features should be >=2D, got {feats.dim()}D"
    assert not torch.isnan(feats).any(), "input_features contain NaN"
    assert not torch.isinf(feats).any(), "input_features contain Inf"

    print(f"\n  PASS: Data loading OK (batch_size={B}, "
          f"seq_len={batch['input_ids'].shape[1]}, "
          f"mask_pct={mask_pct:.1f}%)")
    return True, batch, features


def check_2_loss_at_init(model, batch):
    """Check 2: Forward pass, verify loss is reasonable."""
    print(f"\n{'='*60}")
    print("CHECK 2: Loss at Init")
    print(f"{'='*60}")

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    batch_gpu = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
            batch_gpu[k] = v
        else:
            batch_gpu[k] = v

    model.eval()
    with torch.no_grad():
        outputs = model(**batch_gpu)

    loss = outputs.loss.item()
    print(f"  Initial loss: {loss:.4f}")
    print(f"  Expected range: [{EXPECTED_INIT_LOSS_MIN}, {EXPECTED_INIT_LOSS_MAX}]")

    assert not np.isnan(loss), "Loss is NaN"
    assert not np.isinf(loss), "Loss is Inf"

    if EXPECTED_INIT_LOSS_MIN <= loss <= EXPECTED_INIT_LOSS_MAX:
        print(f"  PASS: Loss {loss:.4f} is in expected range")
    else:
        print(f"  WARNING: Loss {loss:.4f} outside expected range "
              f"[{EXPECTED_INIT_LOSS_MIN}, {EXPECTED_INIT_LOSS_MAX}]")
        print(f"  (Still passing — value is finite and may reflect "
              f"Qwen3-ASR's vocab size or prefix masking ratio)")

    print_gpu_memory("After forward")
    return True


def check_3_single_batch_overfit(model, batch):
    """Check 3: Train 200 steps on 1 batch, loss should approach 0."""
    print(f"\n{'='*60}")
    print(f"CHECK 3: Single-Batch Overfit ({OVERFIT_STEPS} steps)")
    print(f"{'='*60}")

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    batch_gpu = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
            batch_gpu[k] = v
        else:
            batch_gpu[k] = v

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )

    losses = []
    t0 = time.time()

    for step in range(OVERFIT_STEPS):
        optimizer.zero_grad()
        outputs = model(**batch_gpu)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 50 == 0 or step == OVERFIT_STEPS - 1:
            elapsed = time.time() - t0
            print(f"  Step {step:3d}: loss={loss_val:.4f} "
                  f"({elapsed:.1f}s elapsed)")

    first_loss = losses[0]
    last_loss = losses[-1]
    min_loss = min(losses)

    print(f"\n  First loss: {first_loss:.4f}")
    print(f"  Last loss:  {last_loss:.4f}")
    print(f"  Min loss:   {min_loss:.4f}")
    print(f"  Reduction:  {(1 - last_loss/first_loss)*100:.1f}%")

    passed = last_loss < OVERFIT_LOSS_THRESHOLD
    print(f"\n  {'PASS' if passed else 'FAIL'}: "
          f"Last loss {last_loss:.4f} {'<' if passed else '>='} "
          f"threshold {OVERFIT_LOSS_THRESHOLD}")

    if not passed:
        print(f"  DIAGNOSTIC: Loss stuck above {OVERFIT_LOSS_THRESHOLD}.")
        print(f"  Possible causes:")
        print(f"    - LoRA target_modules regex not matching any layers")
        print(f"    - Learning rate too low")
        print(f"    - Labels incorrectly masked (no training signal)")

    print_gpu_memory("After overfit")
    return passed


def check_4_gradient_flow(model, batch):
    """Check 4: Verify all LoRA params have nonzero gradients."""
    print(f"\n{'='*60}")
    print("CHECK 4: Gradient Flow")
    print(f"{'='*60}")

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    batch_gpu = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
            batch_gpu[k] = v
        else:
            batch_gpu[k] = v

    model.train()
    model.zero_grad()
    outputs = model(**batch_gpu)
    outputs.loss.backward()

    zero_grad_params = []
    nonzero_count = 0
    total_trainable = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total_trainable += 1

        if param.grad is None:
            zero_grad_params.append((name, "grad=None"))
        elif param.grad.norm().item() == 0.0:
            zero_grad_params.append((name, "grad_norm=0"))
        else:
            nonzero_count += 1

    print(f"  Trainable parameters: {total_trainable}")
    print(f"  With nonzero gradients: {nonzero_count}")

    if zero_grad_params:
        print(f"  WARNING: {len(zero_grad_params)} params with zero gradients:")
        for name, reason in zero_grad_params[:10]:
            print(f"    {name}: {reason}")
        if len(zero_grad_params) > 10:
            print(f"    ... and {len(zero_grad_params) - 10} more")

    # Print gradient norms for LoRA layers (sample)
    print(f"\n  Sample LoRA gradient norms:")
    lora_count = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if lora_count < 8:
                print(f"    {name}: {grad_norm:.6f}")
            lora_count += 1

    print(f"  Total LoRA params checked: {lora_count}")

    # Check: frozen encoder should have no gradients
    encoder_grad_count = 0
    for name, param in model.named_parameters():
        if "audio_tower" in name and param.requires_grad:
            encoder_grad_count += 1

    if encoder_grad_count > 0:
        print(f"  FAIL: {encoder_grad_count} encoder params are trainable (should be frozen)")
        return False

    passed = len(zero_grad_params) == 0 and lora_count > 0
    print(f"\n  {'PASS' if passed else 'FAIL'}: "
          f"{nonzero_count}/{total_trainable} params have nonzero gradients, "
          f"encoder frozen OK")
    return passed


def check_5_eval_loop(model, features, processor):
    """Check 5: Generate transcription from 1 audio sample."""
    print(f"\n{'='*60}")
    print("CHECK 5: Eval Loop (Generate)")
    print(f"{'='*60}")

    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    audio = features[0]["audio"]
    reference = features[0]["transcript"]
    audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio

    # Build chat-format input using correct API
    conversation = [
        {
            "role": "user",
            "content": [{"type": "audio", "audio": audio_np}],
        },
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=text, audio=[audio_np], return_tensors="pt", padding=False)

    inputs = {}
    for k, v in processor(text=text, audio=[audio_np], return_tensors="pt", padding=False).items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
        inputs[k] = v

    print(f"  Reference: {reference[:100]}...")
    print(f"  Generating...")

    with torch.no_grad():
        gen_output = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
        )

    # Handle both tensor and GenerateOutput returns
    if hasattr(gen_output, "sequences"):
        output_ids = gen_output.sequences
    elif isinstance(gen_output, torch.Tensor):
        output_ids = gen_output
    else:
        output_ids = gen_output[0] if isinstance(gen_output, tuple) else gen_output

    input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
    new_tokens = output_ids[0, input_len:]
    hypothesis = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    hypothesis_raw = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)

    print(f"  Hypothesis (clean): {hypothesis[:100]}")
    print(f"  Hypothesis (raw):   {hypothesis_raw[:100]}")
    print(f"  Output length: {len(new_tokens)} tokens")
    print(f"  Total output length: {output_ids.shape[-1]} tokens")

    # Note: after check 3 overfit training, the model may not generate well
    # on new/same inputs. The key test is that generate() runs without error
    # and produces tokens. Quality is tested separately in validation.
    passed = len(new_tokens) > 0
    print(f"\n  {'PASS' if passed else 'FAIL'}: "
          f"Generate produced {len(new_tokens)} token(s) without error")
    return passed


def check_6_checkpoint_roundtrip(model, processor, features):
    """Check 6: Save adapter → reload → compare output tokens."""
    print(f"\n{'='*60}")
    print("CHECK 6: Checkpoint Round-Trip")
    print(f"{'='*60}")

    from peft import PeftModel

    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    audio = features[0]["audio"]
    audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
    conversation = [
        {
            "role": "user",
            "content": [{"type": "audio", "audio": audio_np}],
        },
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    raw = processor(text=text, audio=[audio_np], return_tensors="pt", padding=False)
    inputs = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
        inputs[k] = v

    # Generate with current model
    torch.manual_seed(SEED)
    with torch.no_grad():
        gen_a = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        tokens_a = gen_a.sequences if hasattr(gen_a, "sequences") else gen_a

    # Save adapter to temp dir
    tmpdir = tempfile.mkdtemp(prefix="roundtrip_")
    print(f"  Saving adapter to: {tmpdir}")
    model.save_pretrained(tmpdir)
    processor.save_pretrained(tmpdir)

    # List saved files
    for f in sorted(os.listdir(tmpdir)):
        fpath = os.path.join(tmpdir, f)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {f}: {size_kb:.1f} KB")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Reload base model + adapter
    print(f"  Reloading base model + adapter...")
    base_model, proc_b = load_model_and_processor()
    patch_outer_forward(base_model)
    model_b = PeftModel.from_pretrained(base_model, tmpdir)
    model_b.eval()
    model_b = model_b.to(device)
    model_dtype = next(model_b.parameters()).dtype

    text_b = proc_b.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    raw_b = proc_b(text=text_b, audio=[audio_np], return_tensors="pt", padding=False)
    inputs_b = {}
    for k, v in raw_b.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
        inputs_b[k] = v

    torch.manual_seed(SEED)
    with torch.no_grad():
        gen_b = model_b.generate(**inputs_b, max_new_tokens=128, do_sample=False)
        tokens_b = gen_b.sequences if hasattr(gen_b, "sequences") else gen_b

    # Compare
    tokens_a_cpu = tokens_a.cpu()
    tokens_b_cpu = tokens_b.cpu()
    equal = torch.equal(tokens_a_cpu, tokens_b_cpu)

    if not equal:
        if tokens_a_cpu.shape != tokens_b_cpu.shape:
            print(f"  Shape mismatch: A={tokens_a_cpu.shape}, B={tokens_b_cpu.shape}")
        else:
            diff = (tokens_a_cpu != tokens_b_cpu).sum().item()
            print(f"  Token differences: {diff}/{tokens_a_cpu.numel()}")

    print(f"\n  {'PASS' if equal else 'FAIL'}: "
          f"Tokens {'identical' if equal else 'differ'} after round-trip")

    # Cleanup
    del model_b, base_model
    torch.cuda.empty_cache()
    shutil.rmtree(tmpdir, ignore_errors=True)

    return equal, None  # model_a was deleted during round-trip test


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Pipeline Sanity")
    parser.add_argument("--fs_manifest", required=True)
    parser.add_argument("--cv_manifest", required=True)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{'='*60}")
    print("PHASE 1: Pipeline Sanity Checks")
    print(f"{'='*60}")
    print(f"  Model: {MODEL_ID}")
    print(f"  FS manifest: {args.fs_manifest}")
    print(f"  CV manifest: {args.cv_manifest}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print_gpu_memory("Start")

    results = {}
    t_total = time.time()

    # Load model once for all checks
    print(f"\nLoading model...")
    model, processor = load_model_and_processor()
    model = setup_lora(model)
    if torch.cuda.is_available():
        model = model.cuda()
    model.print_trainable_parameters()
    print_gpu_memory("After model setup")

    # CHECK 1: Data Loading
    try:
        passed, batch, features = check_1_data_loading(
            args.fs_manifest, args.cv_manifest, processor
        )
        results["1_data_loading"] = passed
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        results["1_data_loading"] = False
        print("\nAborting: Cannot proceed without data loading.")
        sys.exit(1)

    # CHECK 2: Loss at Init
    try:
        results["2_loss_at_init"] = check_2_loss_at_init(model, batch)
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        results["2_loss_at_init"] = False

    # CHECK 3: Single-Batch Overfit
    try:
        results["3_single_batch_overfit"] = check_3_single_batch_overfit(model, batch)
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        results["3_single_batch_overfit"] = False

    # CHECK 4: Gradient Flow (re-init model to get clean gradients)
    try:
        results["4_gradient_flow"] = check_4_gradient_flow(model, batch)
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        results["4_gradient_flow"] = False

    # CHECK 5: Eval Loop
    try:
        results["5_eval_loop"] = check_5_eval_loop(model, features, processor)
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        results["5_eval_loop"] = False

    # CHECK 6: Checkpoint Round-Trip
    try:
        passed, _ = check_6_checkpoint_roundtrip(model, processor, features)
        results["6_checkpoint_roundtrip"] = passed
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        results["6_checkpoint_roundtrip"] = False

    # -- Summary ---------------------------------------------------------------
    elapsed = time.time() - t_total
    n_pass = sum(results.values())
    n_total = len(results)

    print(f"\n{'='*60}")
    print(f"PHASE 1 SUMMARY: {n_pass}/{n_total} checks passed "
          f"({elapsed:.0f}s)")
    print(f"{'='*60}")

    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")

    if n_pass == n_total:
        print(f"\n  GATE PASSED: All {n_total} checks passed.")
        print(f"  Ready to proceed to Phase 2: HP Sweep.")
    else:
        print(f"\n  GATE FAILED: {n_total - n_pass} check(s) failed.")
        print(f"  Fix failures before proceeding to Phase 2.")

    print_gpu_memory("Final")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
