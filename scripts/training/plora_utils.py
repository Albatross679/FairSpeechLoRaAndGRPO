"""
PLoRA utilities: VRAM estimation, packing planner, gradient isolation test, VRAM profiler.

Provides functions to estimate adapter VRAM costs, compute how many adapters
can be packed into available GPU memory, profile empirical VRAM per adapter,
and test gradient isolation between PEFT multi-adapters.

Usage:
    python scripts/training/plora_utils.py --test-isolation
    python scripts/training/plora_utils.py --profile
    python scripts/training/plora_utils.py --estimate
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
from peft import LoraConfig, TaskType

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.training.phase2_hp_sweep import load_model_and_processor, apply_lora

# -- Constants ----------------------------------------------------------------

DEFAULT_HIDDEN_DIM = 1536
DEFAULT_N_LAYERS = 28
ACTIVATION_OVERHEAD_MB = 300  # Will be calibrated by profiling
OUTPUT_DIR = "outputs/packed-sweep"


# -- VRAM Estimation ----------------------------------------------------------

def estimate_adapter_vram_mb(rank, n_targets, hidden_dim=DEFAULT_HIDDEN_DIM,
                              n_layers=DEFAULT_N_LAYERS):
    """Theoretical VRAM per adapter in MB.

    Formula:
    - params_per_layer = 2 * hidden_dim * rank * n_targets (A + B matrices)
    - total_params = params_per_layer * n_layers
    - bytes: 2 (bf16 weight) + 8 (AdamW: 2 fp32 states) + 2 (gradient bf16) = 12
    - activation_overhead_mb = 300 (assumed, calibrated by profiling)

    Args:
        rank: LoRA rank (e.g. 4, 8, 16, 32).
        n_targets: Number of target modules (4 for attn-only, 7 for attn+mlp).
        hidden_dim: Model hidden dimension (1536 for Qwen3-ASR-1.7B).
        n_layers: Number of transformer layers (28 for Qwen3-ASR-1.7B).

    Returns:
        Estimated VRAM in MB (float).
    """
    params_per_layer = 2 * hidden_dim * rank * n_targets  # A + B matrices
    total_params = params_per_layer * n_layers

    bytes_weight = 2       # bf16
    bytes_optimizer = 8    # AdamW: 2 fp32 states (m, v)
    bytes_gradient = 2     # bf16 gradients
    bytes_per_param = bytes_weight + bytes_optimizer + bytes_gradient  # 12

    weight_mb = (total_params * bytes_weight) / (1024 ** 2)
    optimizer_mb = (total_params * bytes_optimizer) / (1024 ** 2)
    gradient_mb = (total_params * bytes_gradient) / (1024 ** 2)
    total_param_mb = (total_params * bytes_per_param) / (1024 ** 2)

    vram_mb = total_param_mb + ACTIVATION_OVERHEAD_MB

    print(f"  Estimate (rank={rank}, targets={n_targets}):")
    print(f"    Parameters: {total_params:,}")
    print(f"    Weight memory:    {weight_mb:.1f} MB")
    print(f"    Optimizer memory: {optimizer_mb:.1f} MB")
    print(f"    Gradient memory:  {gradient_mb:.1f} MB")
    print(f"    Activation est:   {ACTIVATION_OVERHEAD_MB} MB")
    print(f"    Total estimate:   {vram_mb:.1f} MB")

    return vram_mb


# -- Packing Planner ----------------------------------------------------------

def compute_pack_size(configs, available_vram_mb=5000, safety_margin_mb=0):
    """Compute how many adapters from configs fit in available VRAM.

    Greedily adds configs by calling estimate_adapter_vram_mb() until
    the budget is exceeded. Always returns at least 1.

    Args:
        configs: List of dicts with keys: rank, target_mlp (bool).
        available_vram_mb: VRAM budget for adapters (after base model).
        safety_margin_mb: Additional margin to subtract.

    Returns:
        Tuple of (pack_size, estimated_total_mb, per_adapter_mb).
    """
    budget = available_vram_mb - safety_margin_mb
    total_mb = 0.0
    per_adapter_mb = []
    pack_size = 0

    for config in configs:
        n_targets = 7 if config.get("target_mlp", False) else 4
        rank = config.get("rank", 4)
        est_mb = estimate_adapter_vram_mb(rank, n_targets)
        per_adapter_mb.append(est_mb)

        if total_mb + est_mb <= budget:
            total_mb += est_mb
            pack_size += 1
        else:
            break

    # Always return at least 1
    if pack_size == 0:
        pack_size = 1
        total_mb = per_adapter_mb[0] if per_adapter_mb else 0.0

    return pack_size, total_mb, per_adapter_mb[:pack_size]


# -- VRAM Profiler ------------------------------------------------------------

def profile_adapter_vram(model=None, rank=4, target_mlp=False, n_steps=3):
    """Empirical VRAM measurement for one adapter.

    Loads a fresh model (if not provided), applies LoRA, runs n_steps
    forward+backward, and measures peak VRAM delta.

    Args:
        model: Pre-loaded model or None (will load fresh).
        rank: LoRA rank.
        target_mlp: Whether to target MLP modules.
        n_steps: Number of training steps for profiling.

    Returns:
        VRAM delta in MB (float).
    """
    own_model = model is None
    if own_model:
        model, _ = load_model_and_processor()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model = model.cuda()
        baseline = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        print("  WARNING: No CUDA device, returning theoretical estimate")
        n_targets = 7 if target_mlp else 4
        return estimate_adapter_vram_mb(rank, n_targets)

    # Apply LoRA
    alpha = rank * 2
    model = apply_lora(model, rank, alpha, dropout=0.0, target_mlp=target_mlp)

    # Create optimizer for profiling
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    # Create dummy batch
    batch_size = 2
    seq_len = 64
    feat_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device="cuda")
    input_features = torch.randn(batch_size, feat_len, feat_len,
                                  dtype=torch.float32, device="cuda")
    feature_attention_mask = torch.ones(batch_size, feat_len,
                                        dtype=torch.long, device="cuda")
    labels = input_ids.clone()

    # Run training steps
    model.train()
    for step in range(n_steps):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        loss.backward()
        optimizer.step()

    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    delta = peak - baseline

    print(f"  Profile (rank={rank}, mlp={target_mlp}):")
    print(f"    Baseline: {baseline:.0f} MB")
    print(f"    Peak:     {peak:.0f} MB")
    print(f"    Delta:    {delta:.0f} MB")

    # Cleanup
    del optimizer, trainable_params
    del input_ids, attention_mask, input_features, feature_attention_mask, labels
    if own_model:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return delta


# -- Gradient Isolation Test --------------------------------------------------

def test_gradient_isolation(model=None):
    """Verify PEFT set_adapter() provides gradient isolation during training.

    Loads model, applies adapter_A, snapshots its weights, adds adapter_B,
    trains B for 10 steps, then verifies adapter_A weights are exactly unchanged.

    Args:
        model: Pre-loaded model or None (will load fresh).

    Returns:
        bool: True if isolation confirmed (adapter_A unchanged).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    own_model = model is None

    # Load model and apply first adapter (adapter_A = "default")
    if own_model:
        model, _ = load_model_and_processor()

    model = apply_lora(model, rank=4, alpha=4, dropout=0.0, target_mlp=False)

    if torch.cuda.is_available():
        model = model.cuda()

    # Snapshot adapter_A weights (the "default" adapter from get_peft_model)
    print("\n  Snapshotting adapter_A (default) weights...")
    snapshot = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            snapshot[name] = param.detach().clone()
    print(f"  Snapshotted {len(snapshot)} adapter_A parameters")

    # Add adapter_B
    print("  Adding adapter_B...")
    adapter_b_config = LoraConfig(
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model.add_adapter("adapter_B", adapter_b_config)

    # Re-freeze audio_tower after adding adapter
    for name, param in model.named_parameters():
        if "audio_tower" in name:
            param.requires_grad = False

    # Switch to adapter_B
    model.set_adapter("adapter_B")

    # Create optimizer for adapter_B params ONLY
    adapter_b_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "adapter_B" in n
    ]
    print(f"  Adapter_B trainable params: {len(adapter_b_params)}")
    optimizer = torch.optim.AdamW(adapter_b_params, lr=1e-3)

    # Create dummy batch
    device = next(model.parameters()).device
    batch_size = 2
    seq_len = 64
    feat_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    input_features = torch.randn(batch_size, feat_len, feat_len,
                                  dtype=torch.float32, device=device)
    feature_attention_mask = torch.ones(batch_size, feat_len,
                                        dtype=torch.long, device=device)
    labels = input_ids.clone()

    # Train adapter_B for 10 steps
    print("  Training adapter_B for 10 steps...")
    model.train()
    for step in range(10):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        if step % 3 == 0:
            print(f"    Step {step}: loss={loss.item():.4f}")

    # Switch back to adapter_A and check weights
    model.set_adapter("default")

    print("\n  Checking adapter_A weights for changes...")
    max_diff = 0.0
    n_checked = 0
    diffs = []

    for name, param in model.named_parameters():
        if name in snapshot:
            diff = (param.detach() - snapshot[name]).abs().max().item()
            max_diff = max(max_diff, diff)
            n_checked += 1
            diffs.append((name, diff))

    # Print first 5 parameter diffs
    print(f"\n  Parameter diffs (first 5 of {n_checked}):")
    for name, diff in diffs[:5]:
        short_name = name.split(".")[-3:]
        print(f"    {'.'.join(short_name)}: max_diff={diff:.2e}")

    passed = max_diff == 0.0
    result = {
        "passed": passed,
        "max_diff": float(max_diff),
        "n_params_checked": n_checked,
    }

    result_path = os.path.join(OUTPUT_DIR, "gradient_isolation_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    if passed:
        print(f"\n  GRADIENT ISOLATION: PASSED")
        print(f"    Max diff: {max_diff:.2e} (exactly 0.0)")
        print(f"    Params checked: {n_checked}")
    else:
        print(f"\n  GRADIENT ISOLATION: FAILED")
        print(f"    Max diff: {max_diff:.2e}")
        print(f"    Some adapter_A weights were modified during adapter_B training!")

    print(f"  Result saved: {result_path}")

    # Cleanup
    del optimizer, adapter_b_params
    del input_ids, attention_mask, input_features, feature_attention_mask, labels
    del snapshot
    if own_model:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return passed


# -- CLI ----------------------------------------------------------------------

def run_profile_all():
    """Profile VRAM for rank {4,8,16,32} x {attn-only, attn+mlp}."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configs = [
        (4, False), (4, True),
        (8, False), (8, True),
        (16, False), (16, True),
        (32, False), (32, True),
    ]

    profiles = []
    for rank, target_mlp in configs:
        n_targets = 7 if target_mlp else 4
        theoretical = estimate_adapter_vram_mb(rank, n_targets)

        print(f"\n  Profiling rank={rank}, mlp={target_mlp}...")
        empirical = profile_adapter_vram(rank=rank, target_mlp=target_mlp)

        profiles.append({
            "rank": rank,
            "target_mlp": target_mlp,
            "empirical_mb": round(empirical, 1),
            "theoretical_mb": round(theoretical, 1),
        })
        print(f"    Theoretical: {theoretical:.0f} MB, Empirical: {empirical:.0f} MB")

    # Compute max pack sizes
    base_model_mb = 3400
    available_mb = 13000 - base_model_mb  # 9600 MB
    max_pack_sizes = {}
    for p in profiles:
        key = f"rank_{p['rank']}_{'mlp' if p['target_mlp'] else 'attn'}"
        if p["empirical_mb"] > 0:
            max_pack_sizes[key] = int(available_mb / p["empirical_mb"])
        else:
            max_pack_sizes[key] = 1

    result = {
        "profiles": profiles,
        "base_model_mb": base_model_mb,
        "available_mb": available_mb,
        "max_pack_sizes": max_pack_sizes,
    }

    result_path = os.path.join(OUTPUT_DIR, "vram_profile.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print("VRAM Profile Summary")
    print(f"{'='*70}")
    print(f"  {'Rank':>4}  {'MLP':>5}  {'Theoretical':>12}  {'Empirical':>10}  {'Max Pack':>9}")
    print(f"  {'----':>4}  {'-----':>5}  {'----------':>12}  {'---------':>10}  {'--------':>9}")
    for p in profiles:
        key = f"rank_{p['rank']}_{'mlp' if p['target_mlp'] else 'attn'}"
        print(f"  {p['rank']:>4}  {str(p['target_mlp']):>5}  "
              f"{p['theoretical_mb']:>10.0f} MB  {p['empirical_mb']:>8.0f} MB  "
              f"{max_pack_sizes.get(key, '?'):>8}")

    print(f"\n  Result saved: {result_path}")
    return result


def run_estimate_all():
    """Print theoretical estimates for rank {4,8,16,32} x {attn-only, attn+mlp}."""
    configs = [
        (4, 4, "rank-4, attn-only"),
        (4, 7, "rank-4, attn+mlp"),
        (8, 4, "rank-8, attn-only"),
        (8, 7, "rank-8, attn+mlp"),
        (16, 4, "rank-16, attn-only"),
        (16, 7, "rank-16, attn+mlp"),
        (32, 4, "rank-32, attn-only"),
        (32, 7, "rank-32, attn+mlp"),
    ]

    print(f"\n{'='*60}")
    print("Theoretical VRAM Estimates")
    print(f"{'='*60}")
    for rank, n_targets, label in configs:
        print(f"\n  --- {label} ---")
        estimate_adapter_vram_mb(rank, n_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLoRA utilities")
    parser.add_argument("--test-isolation", action="store_true",
                        help="Run gradient isolation test")
    parser.add_argument("--profile", action="store_true",
                        help="Profile VRAM for all rank/target configs")
    parser.add_argument("--estimate", action="store_true",
                        help="Print theoretical VRAM estimates")
    args = parser.parse_args()

    if args.test_isolation:
        passed = test_gradient_isolation()
        sys.exit(0 if passed else 1)
    elif args.profile:
        run_profile_all()
    elif args.estimate:
        run_estimate_all()
    else:
        parser.print_help()
