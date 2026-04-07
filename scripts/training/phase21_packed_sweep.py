"""
Phase 2.1: Packed LoRA HP Sweep with Optuna ask/tell API.

Runs a PLoRA packed sweep using multiple LoRA adapters on a shared frozen
base model. Supports 3 stage modes:
  - Stage 1: PLoRA only (same HPs as Phase 2, validates packing correctness)
  - Stage 2: PLoRA + ASHA pruning (expanded HP space, 100 trials)
  - Stage 3: PLoRA + ASHA + RsLoRA (tests rank-stable scaling)

Uses Optuna ask/tell API for batch trial management with dynamic pack sizing
based on VRAM profiling. OOM recovery retries failed packs at pack_size=1.

Usage:
    python scripts/training/phase21_packed_sweep.py \
        --stage 1 \
        --fs_manifest outputs/manifests/fs_train.csv \
        --cv_manifest outputs/manifests/cv_train.csv \
        --n_trials 20 \
        --steps_per_trial 100 \
        --subset_size 500 \
        --output_dir outputs/packed-sweep \
        --phase2_results outputs/hp-sweep
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import optuna

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.training.phase2_hp_sweep import (
    load_model_and_processor, patch_outer_forward, apply_lora,
    create_stratified_subset, MODEL_ID, SEED,
)
from scripts.training.packed_trainer import PackedTrainer, create_packed_model
from scripts.training.plora_utils import compute_pack_size, estimate_adapter_vram_mb

# -- Constants ----------------------------------------------------------------

DEFAULT_SUBSET_SIZE = 500
DEFAULT_STEPS = 100
DEFAULT_EVAL_STEPS = 20
DEFAULT_EVAL_FRACTION = 0.1
DEFAULT_OUTPUT_DIR = "outputs/packed-sweep"
VRAM_BUDGET_MB = 13000   # 13 GB cap per D-07
BASE_MODEL_VRAM_MB = 3400  # ~3.4 GB for Qwen3-ASR-1.7B bf16
AVAILABLE_FOR_ADAPTERS_MB = VRAM_BUDGET_MB - BASE_MODEL_VRAM_MB  # ~9600 MB
PHASE2_BASELINE_SECONDS = 5520  # Phase 2 took ~1h 32min


# -- HP Suggestion ------------------------------------------------------------

def suggest_hps(trial, stage):
    """Suggest hyperparameters for a trial.

    Stage 1: Same 6 HP dimensions as Phase 2.
    Stage >= 2: Adds lr_scheduler, warmup_ratio, grad_accum_steps.

    Args:
        trial: Optuna trial object.
        stage: Stage number (1, 2, or 3).

    Returns:
        Dict with all hyperparameters including computed alpha.
    """
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    rank = trial.suggest_categorical("rank", [4, 8, 16, 32])
    alpha_ratio = trial.suggest_categorical("alpha_ratio", [1, 2])
    dropout = trial.suggest_float("dropout", 0.0, 0.15, step=0.05)
    target_mlp = trial.suggest_categorical("target_mlp", [False, True])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    alpha = rank * alpha_ratio

    params = {
        "lr": lr,
        "rank": rank,
        "alpha_ratio": alpha_ratio,
        "alpha": alpha,
        "dropout": dropout,
        "target_mlp": target_mlp,
        "weight_decay": weight_decay,
    }

    # Stage 2+: expanded HP space (placeholder for Plan 02)
    if stage >= 2:
        params["lr_scheduler"] = trial.suggest_categorical(
            "lr_scheduler", ["linear", "cosine"])
        params["warmup_ratio"] = trial.suggest_categorical(
            "warmup_ratio", [0.0, 0.03, 0.1])
        params["grad_accum_steps"] = trial.suggest_categorical(
            "grad_accum_steps", [1, 2, 4])

    return params


# -- Results Saving -----------------------------------------------------------

def save_results(study, all_results, elapsed, args):
    """Save sweep results to disk.

    Args:
        study: Optuna study object.
        all_results: List of dicts with trial results.
        elapsed: Wall-clock time in seconds.
        args: CLI arguments.
    """
    stage = args.stage
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Sort by eval_loss
    all_results.sort(key=lambda x: x["eval_loss"])

    # Top-3 configs
    top3 = all_results[:3]
    top3_path = os.path.join(output_dir, f"stage{stage}_top3.json")
    with open(top3_path, "w") as f:
        json.dump(top3, f, indent=2)

    # All results with metadata
    results_data = {
        "stage": stage,
        "n_trials": len(all_results),
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "all_trials": all_results,
    }
    results_path = os.path.join(output_dir, f"stage{stage}_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # All trials CSV
    if all_results:
        rows = []
        for r in all_results:
            row = {"trial_number": r["trial_number"], "eval_loss": r["eval_loss"]}
            row.update(r["params"])
            rows.append(row)
        csv_df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"stage{stage}_all_trials.csv")
        csv_df.to_csv(csv_path, index=False)

    # Print results
    print(f"\n{'='*60}")
    print(f"STAGE {stage} RESULTS ({elapsed/60:.1f} min, {elapsed/3600:.2f} hours)")
    print(f"{'='*60}")

    speedup = PHASE2_BASELINE_SECONDS / elapsed if elapsed > 0 else 0
    print(f"  Wall-clock: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Phase 2 baseline: {PHASE2_BASELINE_SECONDS}s ({PHASE2_BASELINE_SECONDS/60:.0f} min)")
    print(f"  Speedup: {speedup:.2f}x")

    print(f"\n  Top 3 configurations:")
    for i, r in enumerate(top3):
        print(f"\n  #{i+1} (trial {r['trial_number']}, eval_loss={r['eval_loss']:.4f}):")
        for k, v in r["params"].items():
            print(f"    {k}: {v}")

    print(f"\n  Results saved: {results_path}")
    print(f"  Top 3 saved: {top3_path}")


# -- Gate Validation ----------------------------------------------------------

def validate_against_phase2(study, args):
    """Gate validation: compare Stage 1 top-3 vs Phase 2 top-3.

    Match by (rank, target_mlp) tuple. Gate passes if >= 2 shared.

    Args:
        study: Optuna study object.
        args: CLI arguments.
    """
    output_dir = args.output_dir
    phase2_dir = args.phase2_results

    # Load Phase 2 top-3
    phase2_top3_path = os.path.join(phase2_dir, "top3_configs.json")
    if not os.path.exists(phase2_top3_path):
        print(f"\n  WARNING: Phase 2 top-3 not found at {phase2_top3_path}")
        print("  Skipping gate validation")
        return

    with open(phase2_top3_path) as f:
        phase2_top3 = json.load(f)

    # Get Stage 1 top-3 from saved results
    stage1_top3_path = os.path.join(output_dir, f"stage{args.stage}_top3.json")
    with open(stage1_top3_path) as f:
        stage1_top3 = json.load(f)

    # Extract (rank, target_mlp) tuples
    phase2_configs = set()
    for entry in phase2_top3:
        p = entry["params"]
        phase2_configs.add((int(p["rank"]), bool(p["target_mlp"])))

    stage1_configs = set()
    for entry in stage1_top3:
        p = entry["params"]
        stage1_configs.add((int(p["rank"]), bool(p["target_mlp"])))

    shared = phase2_configs & stage1_configs
    shared_count = len(shared)
    gate_passed = shared_count >= 2

    # Compute speedup
    results_path = os.path.join(output_dir, f"stage{args.stage}_results.json")
    with open(results_path) as f:
        results_data = json.load(f)
    elapsed = results_data["elapsed_seconds"]
    speedup = PHASE2_BASELINE_SECONDS / elapsed if elapsed > 0 else 0

    validation = {
        "gate": "PASSED" if gate_passed else "FAILED",
        "shared_count": shared_count,
        "shared_configs": [list(c) for c in shared],
        "phase2_top3": phase2_top3,
        "stage1_top3": stage1_top3,
        "speedup_factor": round(speedup, 2),
    }

    val_path = os.path.join(output_dir, f"stage{args.stage}_validation.json")
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GATE VALIDATION (Stage {args.stage} vs Phase 2)")
    print(f"{'='*60}")
    print(f"  Phase 2 top-3 configs: {phase2_configs}")
    print(f"  Stage 1 top-3 configs: {stage1_configs}")
    print(f"  Shared: {shared} ({shared_count} matches)")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"\n  {'GATE PASSED' if gate_passed else 'GATE FAILED'}: "
          f"{shared_count} >= 2 shared (rank, target_mlp) pairs")
    print(f"  Validation saved: {val_path}")


# -- Main Sweep Loop ----------------------------------------------------------

def run_packed_sweep(args):
    """Main packed sweep loop using Optuna ask/tell API.

    Args:
        args: CLI arguments.

    Returns:
        Optuna study object.
    """
    stage = args.stage
    n_trials = args.n_trials
    steps = args.steps_per_trial
    eval_steps = DEFAULT_EVAL_STEPS

    # 1. Create subset
    print(f"\nCreating stratified subset (size={args.subset_size})...")
    subset_df = create_stratified_subset(
        args.fs_manifest, args.cv_manifest, args.subset_size, seed=SEED)

    # Try to reuse Phase 2 eval split for fair comparison
    phase2_eval_path = os.path.join(args.phase2_results, "sweep_eval.csv")
    if os.path.exists(phase2_eval_path):
        print(f"  Reusing Phase 2 eval split: {phase2_eval_path}")
        eval_df = pd.read_csv(phase2_eval_path)
        # Exclude eval rows from train set by utterance_id
        if "utterance_id" in eval_df.columns and "utterance_id" in subset_df.columns:
            eval_ids = set(eval_df["utterance_id"].values)
            train_df = subset_df[~subset_df["utterance_id"].isin(eval_ids)].reset_index(drop=True)
        else:
            # Fall back to tail split
            n_eval = max(10, int(len(subset_df) * DEFAULT_EVAL_FRACTION))
            eval_df = subset_df.tail(n_eval).reset_index(drop=True)
            train_df = subset_df.head(len(subset_df) - n_eval).reset_index(drop=True)
    else:
        n_eval = max(10, int(len(subset_df) * DEFAULT_EVAL_FRACTION))
        eval_df = subset_df.tail(n_eval).reset_index(drop=True)
        train_df = subset_df.head(len(subset_df) - n_eval).reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_dir, "sweep_train.csv"), index=False)
    eval_df.to_csv(os.path.join(args.output_dir, "sweep_eval.csv"), index=False)
    print(f"  Train: {len(train_df)}, Eval: {len(eval_df)}")

    # 2. Create Optuna study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)  # Stage 1
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner,
        study_name=f"packed-sweep-stage{stage}",
    )

    # 3. Load processor once
    _, processor = load_model_and_processor()

    # 4. Ask/tell loop with packing
    all_results = []
    completed = 0
    t0 = time.time()
    oom_configs = set()  # Track (max_rank, has_mlp) combos that caused OOM

    while completed < n_trials:
        remaining = n_trials - completed

        # Ask trials for this batch
        pending_trials = []
        pending_configs = []
        for _ in range(remaining):
            trial = study.ask()
            params = suggest_hps(trial, stage)
            pending_trials.append(trial)
            pending_configs.append({
                "name": f"trial_{trial.number:03d}",
                **params,
            })

        # Compute pack size based on configs
        pack_size, est_total, per_adapter = compute_pack_size(
            pending_configs, available_vram_mb=AVAILABLE_FOR_ADAPTERS_MB)
        pack_size = min(pack_size, len(pending_configs))

        # Check OOM history: if this config combo caused OOM before, force pack_size=1
        max_rank = max(c["rank"] for c in pending_configs[:pack_size])
        has_mlp = any(c.get("target_mlp", False) for c in pending_configs[:pack_size])
        if (max_rank, has_mlp) in oom_configs:
            pack_size = 1
            print(f"  OOM history: forcing pack_size=1 for rank={max_rank}, mlp={has_mlp}")

        # Take first pack_size configs for this batch
        batch_trials = pending_trials[:pack_size]
        batch_configs = pending_configs[:pack_size]

        # Tell unused trials as pruned so we can re-ask them
        for unused_trial in pending_trials[pack_size:]:
            study.tell(unused_trial, float("inf"),
                       state=optuna.trial.TrialState.PRUNED)

        print(f"\n{'='*60}")
        print(f"Pack batch: {len(batch_configs)} adapters "
              f"(est {est_total:.0f} MB, budget {AVAILABLE_FOR_ADAPTERS_MB} MB)")
        for c in batch_configs:
            print(f"  {c['name']}: rank={c['rank']}, mlp={c.get('target_mlp', False)}, "
                  f"lr={c['lr']:.2e}")
        print(f"{'='*60}")

        try:
            # Load fresh model + apply first adapter
            model, _ = load_model_and_processor()
            model = create_packed_model(model, batch_configs[0])
            if torch.cuda.is_available():
                model = model.cuda()

            # First adapter uses "default" name from get_peft_model
            batch_configs[0]["name"] = "default"

            # Create PackedTrainer
            trainer = PackedTrainer(
                model=model,
                adapter_configs=batch_configs,
                processor=processor,
                train_df=train_df,
                eval_df=eval_df,
                steps_per_trial=steps,
                eval_steps=eval_steps,
                per_device_batch_size=2,
                seed=SEED,
            )
            trainer.setup_adapters()

            # Train
            for step, rung_results in trainer.train(
                    n_steps=steps, eval_steps=eval_steps):
                pass  # Stage 1: no pruning mid-training

            # Collect results
            results = trainer.get_results()
            for trial_obj, config in zip(batch_trials, batch_configs):
                name = config["name"]
                if name in results:
                    eval_loss = results[name]
                    study.tell(trial_obj, eval_loss)
                    all_results.append({
                        "trial_number": trial_obj.number,
                        "eval_loss": eval_loss,
                        "params": {k: config[k] for k in
                                   ["rank", "alpha_ratio", "alpha", "dropout",
                                    "target_mlp", "lr", "weight_decay"]
                                   if k in config},
                    })
                    print(f"  {name}: eval_loss={eval_loss:.4f}")
                else:
                    study.tell(trial_obj, float("inf"),
                               state=optuna.trial.TrialState.FAIL)
                    all_results.append({
                        "trial_number": trial_obj.number,
                        "eval_loss": float("inf"),
                        "params": {k: config[k] for k in
                                   ["rank", "alpha_ratio", "alpha", "dropout",
                                    "target_mlp", "lr", "weight_decay"]
                                   if k in config},
                    })

            completed += len(batch_trials)

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM with pack_size={len(batch_configs)}! "
                  f"Cleaning up and retrying at pack_size=1...")
            oom_configs.add((max_rank, has_mlp))

            # Tell all trials in this batch as failed
            for trial_obj in batch_trials:
                try:
                    study.tell(trial_obj, float("inf"),
                               state=optuna.trial.TrialState.FAIL)
                except Exception:
                    pass

            # Don't increment completed -- these trials need to be re-asked
            # (they were told as FAIL, so Optuna will ask new ones)

        except Exception as e:
            print(f"\n  ERROR in pack batch: {e}")
            for trial_obj in batch_trials:
                try:
                    study.tell(trial_obj, float("inf"),
                               state=optuna.trial.TrialState.FAIL)
                except Exception:
                    pass
            completed += len(batch_trials)

        finally:
            # Cleanup
            try:
                del model, trainer
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.time() - t0

    # 5. Save results
    save_results(study, all_results, elapsed, args)

    # 6. Gate validation (Stage 1 only)
    if stage == 1 and args.phase2_results:
        validate_against_phase2(study, args)

    return study


# -- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.1: Packed LoRA HP Sweep")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3],
                        help="Stage to run (1=PLoRA, 2=+ASHA, 3=+RsLoRA)")
    parser.add_argument("--fs_manifest", default="outputs/manifests/fs_train.csv",
                        help="Fair-Speech train manifest")
    parser.add_argument("--cv_manifest", default="outputs/manifests/cv_train.csv",
                        help="Common Voice train manifest")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of trials (default: 20)")
    parser.add_argument("--steps_per_trial", type=int, default=DEFAULT_STEPS,
                        help="Steps per trial (default: 100)")
    parser.add_argument("--subset_size", type=int, default=DEFAULT_SUBSET_SIZE,
                        help="Subset size (default: 500)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--phase2_results", default="outputs/hp-sweep",
                        help="Phase 2 results dir for gate validation")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{'='*60}")
    print(f"PHASE 2.1: Packed LoRA HP Sweep (Stage {args.stage})")
    print(f"{'='*60}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Steps/trial: {args.steps_per_trial}")
    print(f"  Subset size: {args.subset_size}")
    print(f"  VRAM budget: {VRAM_BUDGET_MB} MB")
    print(f"  Available for adapters: {AVAILABLE_FOR_ADAPTERS_MB} MB")
    print(f"  Output: {args.output_dir}")

    run_packed_sweep(args)


if __name__ == "__main__":
    main()
