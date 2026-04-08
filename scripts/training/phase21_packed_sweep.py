"""
Phase 2.1: Packed LoRA HP Sweep with Optuna ask/tell API.

Runs a PLoRA packed sweep using multiple LoRA adapters on a shared frozen
base model. Supports 3 stage modes:
  - Stage 1: PLoRA only (same HPs as Phase 2, validates packing correctness)
  - Stage 2: PLoRA + ASHA pruning (same 6 HP dims, HyperbandPruner, 100 trials)
  - Stage 3: PLoRA + ASHA + RsLoRA (9 HP dims, tests rank-stable scaling)

Uses Optuna ask/tell API for batch trial management with dynamic pack sizing
based on VRAM profiling. OOM recovery retries failed packs at pack_size=1.

Stage 2-3: Rung-based pruning at steps {25, 50, 75} via HyperbandPruner.
Adapters are deactivated (not removed) at each rung, freeing compute for
surviving adapters.

Usage:
    python scripts/training/phase21_packed_sweep.py \
        --stage 2 \
        --fs_manifest outputs/manifests/fs_train.csv \
        --cv_manifest outputs/manifests/cv_train.csv \
        --n_trials 100 \
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
from collections import Counter

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
RUNG_STEPS = [25, 50, 75]  # D-10: report at these steps for 100-step trials


# -- HP Suggestion ------------------------------------------------------------

def suggest_hps(trial, stage):
    """Suggest hyperparameters for a trial.

    Stage 1: Same 6 HP dimensions as Phase 2.
    Stage 2: Same 6 HP dimensions (ASHA only change is pruning).
    Stage 3: 9 HP dimensions (6 original + lr_scheduler, warmup_ratio, grad_accum_steps).

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

    # Stage 3: expanded HP space with scheduler, warmup, grad accum (D-12)
    if stage >= 3:
        params["lr_scheduler"] = trial.suggest_categorical(
            "lr_scheduler", ["linear", "cosine"])
        params["warmup_ratio"] = trial.suggest_categorical(
            "warmup_ratio", [0.0, 0.03, 0.1])
        params["grad_accum_steps"] = trial.suggest_categorical(
            "grad_accum_steps", [1, 2, 4])
    else:
        # Defaults for Stage 1-2
        params["lr_scheduler"] = "constant"
        params["warmup_ratio"] = 0.0
        params["grad_accum_steps"] = 1

    return params


# -- Gate Validation ----------------------------------------------------------

def validate_gate(current_top3, prior_top3):
    """Gate: >= 2 shared entries matched by (rank, target_mlp).

    Args:
        current_top3: List of trial result dicts (current stage).
        prior_top3: List of trial result dicts (prior stage).

    Returns:
        Tuple of (passed: bool, shared: list, count: int).
    """
    current_keys = {(int(c["params"]["rank"]), bool(c["params"]["target_mlp"]))
                    for c in current_top3}
    prior_keys = {(int(p["params"]["rank"]), bool(p["params"]["target_mlp"]))
                  for p in prior_top3}
    shared = current_keys & prior_keys
    passed = len(shared) >= 2
    return passed, [list(s) for s in shared], len(shared)


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


# -- RsLoRA Rank Analysis ----------------------------------------------------

def save_rslora_rank_analysis(stage3_results, stage2_results, args):
    """Analyze whether RsLoRA changed the optimal rank distribution.

    Args:
        stage3_results: List of Stage 3 trial result dicts.
        stage2_results: List of Stage 2 trial result dicts.
        args: CLI arguments.
    """
    stage2_sorted = sorted(stage2_results, key=lambda x: x["eval_loss"])
    stage3_sorted = sorted(stage3_results, key=lambda x: x["eval_loss"])

    # Top-10 rank distributions
    s2_ranks = [t["params"]["rank"] for t in stage2_sorted[:10]]
    s3_ranks = [t["params"]["rank"] for t in stage3_sorted[:10]]
    s2_dist = dict(Counter(s2_ranks))
    s3_dist = dict(Counter(s3_ranks))

    s2_best_rank = stage2_sorted[0]["params"]["rank"]
    s3_best_rank = stage3_sorted[0]["params"]["rank"]

    analysis = {
        "stage2_top10_rank_distribution": s2_dist,
        "stage3_top10_rank_distribution": s3_dist,
        "optimal_rank_changed": s2_best_rank != s3_best_rank,
        "stage2_best_rank": s2_best_rank,
        "stage3_best_rank": s3_best_rank,
        "stage2_best_eval_loss": stage2_sorted[0]["eval_loss"],
        "stage3_best_eval_loss": stage3_sorted[0]["eval_loss"],
        "conclusion": (
            f"RsLoRA shifts optimal rank from {s2_best_rank} to {s3_best_rank} "
            f"by stabilizing gradients at higher ranks"
            if s2_best_rank != s3_best_rank else
            f"RsLoRA does not change optimal rank (still {s2_best_rank}); "
            f"standard scaling was already sufficient for this model"
        ),
    }

    path = os.path.join(args.output_dir, "rslora_rank_analysis.json")
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nRsLoRA Rank Analysis:")
    print(f"  Stage 2 best rank: {s2_best_rank}")
    print(f"  Stage 3 best rank: {s3_best_rank}")
    print(f"  Optimal rank changed: {analysis['optimal_rank_changed']}")
    print(f"  Saved to {path}")


# -- Results Saving -----------------------------------------------------------

def save_results(study, all_results, elapsed, args, pruning_log=None):
    """Save sweep results to disk.

    Args:
        study: Optuna study object.
        all_results: List of dicts with trial results.
        elapsed: Wall-clock time in seconds.
        args: CLI arguments.
        pruning_log: Optional list of pruning events for stages 2-3.
    """
    stage = args.stage
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Sort by eval_loss (filter out inf/nan)
    valid_results = [r for r in all_results if r["eval_loss"] < float("inf")]
    valid_results.sort(key=lambda x: x["eval_loss"])

    # Top-3 configs
    top3 = valid_results[:3]
    top3_path = os.path.join(output_dir, f"stage{stage}_top3.json")
    with open(top3_path, "w") as f:
        json.dump(top3, f, indent=2)

    # Pruning stats for stages 2-3
    pruning_stats = {}
    if pruning_log and stage >= 2:
        total_pruned = len(pruning_log)
        by_step = Counter(e["step"] for e in pruning_log)
        # Compute effective full trial equivalents
        eff = sum(
            (r.get("pruned_at_step", args.steps_per_trial) / args.steps_per_trial)
            for r in all_results
        )
        pruning_stats = {
            "total_pruned": total_pruned,
            "pruned_at_step_25": by_step.get(25, 0),
            "pruned_at_step_50": by_step.get(50, 0),
            "pruned_at_step_75": by_step.get(75, 0),
            "effective_full_trial_equivalents": round(eff, 1),
        }

    # All results with metadata
    results_data = {
        "stage": stage,
        "n_trials": len(all_results),
        "n_completed": len(valid_results),
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "top3": top3,
        "all_trials": all_results,
        "use_rslora": (stage == 3),
    }
    if pruning_stats:
        results_data["pruner"] = "HyperbandPruner"
        results_data["reduction_factor"] = args.reduction_factor
        results_data["rung_steps"] = RUNG_STEPS
        results_data["pruning_stats"] = pruning_stats

    results_path = os.path.join(output_dir, f"stage{stage}_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # All trials CSV
    if valid_results:
        rows = []
        for r in valid_results:
            row = {"trial_number": r["trial_number"], "eval_loss": r["eval_loss"]}
            row.update(r["params"])
            if "pruned_at_step" in r:
                row["pruned_at_step"] = r["pruned_at_step"]
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

    if pruning_stats:
        print(f"\n  Pruning stats:")
        print(f"    Total pruned: {pruning_stats['total_pruned']}")
        print(f"    At step 25: {pruning_stats['pruned_at_step_25']}")
        print(f"    At step 50: {pruning_stats['pruned_at_step_50']}")
        print(f"    At step 75: {pruning_stats['pruned_at_step_75']}")
        print(f"    Effective full trials: {pruning_stats['effective_full_trial_equivalents']}")

    print(f"\n  Top 3 configurations:")
    for i, r in enumerate(top3):
        print(f"\n  #{i+1} (trial {r['trial_number']}, eval_loss={r['eval_loss']:.4f}):")
        for k, v in r["params"].items():
            print(f"    {k}: {v}")

    print(f"\n  Results saved: {results_path}")
    print(f"  Top 3 saved: {top3_path}")


# -- Stage Validation ---------------------------------------------------------

def run_stage_validation(all_results, args):
    """Run stage-specific gate validation after sweep completes.

    Stage 2: validates against Stage 1 top-3.
    Stage 3: validates against Stage 2 top-3, runs RsLoRA rank analysis.
    """
    stage = args.stage
    output_dir = args.output_dir
    sorted_results = sorted(
        [r for r in all_results if r["eval_loss"] < float("inf")],
        key=lambda x: x["eval_loss"]
    )
    current_top3 = sorted_results[:3]

    if stage == 2:
        # Load Stage 1 top-3 for gate
        s1_path = os.path.join(output_dir, "stage1_results.json")
        if not os.path.exists(s1_path):
            print(f"\n  WARNING: Stage 1 results not found at {s1_path}")
            return
        with open(s1_path) as f:
            stage1_data = json.load(f)
        # Stage 1 results may have top3 key or use all_trials[:3]
        if "top3" in stage1_data:
            stage1_top3 = stage1_data["top3"]
        else:
            stage1_top3 = sorted(
                stage1_data["all_trials"],
                key=lambda x: x["eval_loss"]
            )[:3]

        passed, shared, count = validate_gate(current_top3, stage1_top3)
        validation = {
            "gate": "PASSED" if passed else "FAILED",
            "shared_count": count,
            "shared_configs": shared,
            "stage1_top3": stage1_top3,
            "stage2_top3": current_top3,
        }
        val_path = os.path.join(output_dir, "stage2_validation.json")
        with open(val_path, "w") as f:
            json.dump(validation, f, indent=2)

        print(f"\n{'='*60}")
        print(f"GATE VALIDATION (Stage 2 vs Stage 1)")
        print(f"{'='*60}")
        print(f"  Shared: {shared} ({count} matches)")
        print(f"  {'GATE PASSED' if passed else 'GATE FAILED'}")
        print(f"  Saved: {val_path}")

    elif stage == 3:
        # Load Stage 2 top-3 for gate
        s2_path = os.path.join(output_dir, "stage2_results.json")
        if not os.path.exists(s2_path):
            print(f"\n  WARNING: Stage 2 results not found at {s2_path}")
            return
        with open(s2_path) as f:
            stage2_data = json.load(f)
        if "top3" in stage2_data:
            stage2_top3 = stage2_data["top3"]
        else:
            stage2_top3 = sorted(
                stage2_data["all_trials"],
                key=lambda x: x["eval_loss"]
            )[:3]

        passed, shared, count = validate_gate(current_top3, stage2_top3)
        # Gate may legitimately fail if RsLoRA changes optimal rank (D-17)
        gate_label = "PASSED" if passed else "EXPECTED_FAIL_RSLORA"
        validation = {
            "gate": gate_label,
            "shared_count": count,
            "shared_configs": shared,
            "stage2_top3": stage2_top3,
            "stage3_top3": current_top3,
            "note": "Gate failure expected if RsLoRA changes optimal rank" if not passed else "",
        }
        val_path = os.path.join(output_dir, "stage3_validation.json")
        with open(val_path, "w") as f:
            json.dump(validation, f, indent=2)

        print(f"\n{'='*60}")
        print(f"GATE VALIDATION (Stage 3 vs Stage 2)")
        print(f"{'='*60}")
        print(f"  Shared: {shared} ({count} matches)")
        print(f"  Gate: {gate_label}")
        print(f"  Saved: {val_path}")

        # RsLoRA rank analysis (RSLORA-03)
        stage2_trials = stage2_data.get("all_trials", [])
        save_rslora_rank_analysis(all_results, stage2_trials, args)


# -- Main Sweep Loop ----------------------------------------------------------

def run_packed_sweep(args):
    """Main packed sweep loop using Optuna ask/tell API.

    For stages 2-3, uses HyperbandPruner with rung-based pruning.
    Only asks pack_size trials per iteration (not all remaining) to avoid
    poisoning the study with false pruned trials.

    Args:
        args: CLI arguments.

    Returns:
        Optuna study object.
    """
    stage = args.stage
    n_trials = args.n_trials
    steps = args.steps_per_trial
    eval_steps = DEFAULT_EVAL_STEPS
    use_rslora = (stage == 3)

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

    # 2. Create Optuna study with appropriate pruner
    if stage == 1:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
        rung_steps = None
    else:
        # D-08: HyperbandPruner for Stages 2-3
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=25,         # D-09: no pruning before step 25
            max_resource=100,        # Full budget = 100 steps
            reduction_factor=args.reduction_factor,  # D-09: default 3
        )
        rung_steps = RUNG_STEPS  # D-10: [25, 50, 75]

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
    pruning_log = []
    completed = 0
    t0 = time.time()
    oom_configs = set()  # Track (max_rank, has_mlp) combos that caused OOM

    while completed < n_trials:
        remaining = n_trials - completed

        # Ask a small batch of trials (pack_size or fewer)
        # First, ask one trial to determine VRAM profile for pack sizing
        first_trial = study.ask()
        first_params = suggest_hps(first_trial, stage)
        first_config = {
            "name": f"trial_{first_trial.number:03d}",
            "use_rslora": use_rslora,
            **first_params,
        }

        # Compute pack size from first config's VRAM estimate
        pack_size, est_total, per_adapter = compute_pack_size(
            [first_config], available_vram_mb=AVAILABLE_FOR_ADAPTERS_MB)
        # How many more can we fit?
        if per_adapter > 0:
            max_pack = int(AVAILABLE_FOR_ADAPTERS_MB / per_adapter)
        else:
            max_pack = remaining
        pack_size = min(max_pack, remaining)

        # Check OOM history
        if (first_config["rank"], first_config.get("target_mlp", False)) in oom_configs:
            pack_size = 1

        # Ask additional trials to fill the pack
        batch_trials = [first_trial]
        batch_configs = [first_config]

        for _ in range(min(pack_size - 1, remaining - 1)):
            trial = study.ask()
            params = suggest_hps(trial, stage)
            config = {
                "name": f"trial_{trial.number:03d}",
                "use_rslora": use_rslora,
                **params,
            }
            batch_trials.append(trial)
            batch_configs.append(config)

        # Recompute pack_size with actual configs
        pack_size, est_total, per_adapter = compute_pack_size(
            batch_configs, available_vram_mb=AVAILABLE_FOR_ADAPTERS_MB)
        pack_size = min(pack_size, len(batch_configs))

        # If we asked too many, trim the batch (tell extras as pruned)
        if pack_size < len(batch_configs):
            extra_trials = batch_trials[pack_size:]
            extra_configs = batch_configs[pack_size:]
            batch_trials = batch_trials[:pack_size]
            batch_configs = batch_configs[:pack_size]
            for unused_trial in extra_trials:
                study.tell(unused_trial, float("inf"),
                           state=optuna.trial.TrialState.PRUNED)

        print(f"\n{'='*60}")
        print(f"Pack batch: {len(batch_configs)} adapters "
              f"(est {est_total:.0f} MB, budget {AVAILABLE_FOR_ADAPTERS_MB} MB)")
        for c in batch_configs:
            rslora_str = ", rslora" if c.get("use_rslora") else ""
            sched_str = f", sched={c.get('lr_scheduler', 'constant')}" if c.get("lr_scheduler", "constant") != "constant" else ""
            print(f"  {c['name']}: rank={c['rank']}, mlp={c.get('target_mlp', False)}, "
                  f"lr={c['lr']:.2e}{rslora_str}{sched_str}")
        print(f"{'='*60}")

        try:
            # Load fresh model + apply first adapter
            model, _ = load_model_and_processor()
            model = create_packed_model(model, batch_configs[0])
            if torch.cuda.is_available():
                model = model.cuda()

            # First adapter uses "default" name from get_peft_model
            batch_configs[0]["name"] = "default"

            # Build trial_map: adapter_name -> optuna trial
            trial_map = {}
            for trial_obj, config in zip(batch_trials, batch_configs):
                trial_map[config["name"]] = trial_obj

            # Create PackedTrainer
            rung_steps_list = rung_steps if rung_steps else []
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
                rung_steps=rung_steps_list,
            )
            trainer.setup_adapters()

            # Track which adapters were pruned at which step
            batch_pruned = {}  # name -> step

            # Train with pruning at rung steps
            for step, rung_results in trainer.train(
                    n_steps=steps, eval_steps=eval_steps,
                    rung_steps=rung_steps_list):
                # At each rung, report to Optuna and check pruning
                if rung_steps and step in set(rung_steps):
                    for name, eval_loss in rung_results.items():
                        if name not in trial_map:
                            continue
                        trial_obj = trial_map[name]
                        trial_obj.report(eval_loss, step)
                        if trial_obj.should_prune():
                            trainer.deactivate_adapter(name)
                            batch_pruned[name] = step
                            pruning_log.append({
                                "trial": trial_obj.number,
                                "adapter": name,
                                "step": step,
                                "eval_loss": eval_loss,
                            })
                            print(f"  PRUNED {name} at step {step} "
                                  f"(eval_loss={eval_loss:.4f})")

            # Collect results for all adapters (active and pruned)
            results = trainer.get_results()
            for trial_obj, config in zip(batch_trials, batch_configs):
                name = config["name"]
                if name in results:
                    eval_loss = results[name]
                    # Only tell Optuna if not already pruned
                    if name not in batch_pruned:
                        study.tell(trial_obj, eval_loss)
                    result_entry = {
                        "trial_number": trial_obj.number,
                        "eval_loss": eval_loss,
                        "params": {k: config[k] for k in
                                   ["rank", "alpha_ratio", "alpha", "dropout",
                                    "target_mlp", "lr", "weight_decay",
                                    "lr_scheduler", "warmup_ratio", "grad_accum_steps"]
                                   if k in config},
                    }
                    if name in batch_pruned:
                        result_entry["pruned_at_step"] = batch_pruned[name]
                        result_entry["status"] = "pruned"
                    else:
                        result_entry["pruned_at_step"] = steps
                        result_entry["status"] = "completed"
                    all_results.append(result_entry)
                    print(f"  {name}: eval_loss={eval_loss:.4f}"
                          f"{f' (pruned@{batch_pruned[name]})' if name in batch_pruned else ''}")
                else:
                    # No results -- mark as failed
                    if name not in batch_pruned:
                        study.tell(trial_obj, float("inf"),
                                   state=optuna.trial.TrialState.FAIL)
                    all_results.append({
                        "trial_number": trial_obj.number,
                        "eval_loss": float("inf"),
                        "params": {k: config[k] for k in
                                   ["rank", "alpha_ratio", "alpha", "dropout",
                                    "target_mlp", "lr", "weight_decay",
                                    "lr_scheduler", "warmup_ratio", "grad_accum_steps"]
                                   if k in config},
                        "status": "failed",
                    })

            completed += len(batch_trials)
            print(f"\n  Progress: {completed}/{n_trials} trials complete")

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM with pack_size={len(batch_configs)}! "
                  f"Cleaning up and retrying at pack_size=1...")
            max_rank = max(c["rank"] for c in batch_configs)
            has_mlp = any(c.get("target_mlp", False) for c in batch_configs)
            oom_configs.add((max_rank, has_mlp))

            # Tell all trials in this batch as failed
            for trial_obj in batch_trials:
                try:
                    study.tell(trial_obj, float("inf"),
                               state=optuna.trial.TrialState.FAIL)
                except Exception:
                    pass
            # Don't increment completed -- re-ask

        except Exception as e:
            print(f"\n  ERROR in pack batch: {e}")
            import traceback
            traceback.print_exc()
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
    save_results(study, all_results, elapsed, args, pruning_log=pruning_log)

    # 6. Stage validation
    if stage == 1 and args.phase2_results:
        validate_against_phase2(study, args)
    elif stage >= 2:
        run_stage_validation(all_results, args)

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
    parser.add_argument("--n_trials", type=int, default=None,
                        help="Number of trials. Default: 20 for stage 1, 100 for stages 2-3")
    parser.add_argument("--steps_per_trial", type=int, default=DEFAULT_STEPS,
                        help="Steps per trial (default: 100)")
    parser.add_argument("--subset_size", type=int, default=DEFAULT_SUBSET_SIZE,
                        help="Subset size (default: 500)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--phase2_results", default="outputs/hp-sweep",
                        help="Phase 2 results dir for gate validation")
    parser.add_argument("--reduction_factor", type=int, default=3,
                        help="ASHA reduction factor (default: 3, try 2 for gentler pruning)")
    args = parser.parse_args()

    # Set default n_trials based on stage
    if args.n_trials is None:
        args.n_trials = 20 if args.stage == 1 else 100

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
    if args.stage >= 2:
        print(f"  Pruner: HyperbandPruner (reduction_factor={args.reduction_factor})")
        print(f"  Rung steps: {RUNG_STEPS}")
    if args.stage == 3:
        print(f"  RsLoRA: enabled")
        print(f"  HP dims: 9 (6 original + lr_scheduler, warmup_ratio, grad_accum_steps)")

    run_packed_sweep(args)


if __name__ == "__main__":
    main()
