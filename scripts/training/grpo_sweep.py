"""
Full-scale GRPO lambda sweep orchestrator (Phase 5).

Iterates over a configurable set of lambda values, running full-scale GRPO
training for each one sequentially on the complete dataset (~45K utterances).
Each lambda produces a trained LoRA adapter checkpoint. After all runs complete,
evaluates all checkpoints to compute Pareto frontier metrics.

Uses train_grpo.py's fullscale infrastructure: full dataset, DAPO asymmetric
clipping, early stopping, best checkpoint selection.

Lambda sweep strategy (from Phase 5 research):
  - Non-linear spacing: {0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0}
  - Sequential execution (single T4 GPU)
  - Fresh LoRA adapter per lambda (no warm-starting between lambdas)
  - Budget: ~1000-2000 steps per lambda, 3-5 days wall clock total

Usage:
    # Full sweep (all 8 lambda values)
    python scripts/training/grpo_sweep.py \
        --fs_manifest outputs/manifests/fs_train.csv \
        --cv_manifest outputs/manifests/cv_train.csv \
        --output_dir outputs/grpo \
        --max_steps 1500 --G 2

    # Coarse sweep first (identify transition zone)
    python scripts/training/grpo_sweep.py \
        --lambdas 0.0 0.2 0.5 0.8 1.0 \
        --output_dir outputs/grpo \
        --max_steps 1000

    # Fine-grained follow-up in transition zone
    python scripts/training/grpo_sweep.py \
        --lambdas 0.1 0.15 0.25 0.3 \
        --output_dir outputs/grpo \
        --max_steps 1500 --resume
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

from scripts.training.train_grpo import (
    GRPOConfig,
    setup_fullscale_data,
    setup_model,
    train_grpo,
    SEED,
)

# -- Default lambda values (non-linear spacing per research) -------------------

DEFAULT_LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
COARSE_LAMBDAS = [0.0, 0.2, 0.5, 0.8, 1.0]


def load_existing_results(output_dir):
    """Load sweep results already on disk (for --resume)."""
    results_path = os.path.join(output_dir, "sweep_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return []


def lambda_already_done(output_dir, lambda_val):
    """Check if a lambda value has a completed training summary."""
    lambda_dir = os.path.join(output_dir, f"lambda_{lambda_val:.2f}")
    summary_path = os.path.join(lambda_dir, "training_summary.json")
    return os.path.exists(summary_path)


def run_lambda_sweep(args):
    """Run full-scale GRPO training for each lambda value sequentially."""
    lambdas = args.lambdas
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load existing results if resuming
    all_results = load_existing_results(output_dir) if args.resume else []
    completed_lambdas = {r["lambda"] for r in all_results}

    # Check GPU before starting
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}, {gpu_mem:.1f}GB total")
    else:
        print("WARNING: No CUDA device found. Training will be extremely slow.")

    # Set up data once (shared across all lambda runs)
    print(f"\n{'='*60}")
    print(f"GRPO LAMBDA SWEEP: Full-Scale ({len(lambdas)} lambda values)")
    print(f"Lambdas: {lambdas}")
    print(f"{'='*60}")

    train_loader, eval_loader = setup_fullscale_data(args)

    # W&B group for sweep comparison
    os.environ["WANDB_GROUP"] = f"grpo-sweep-{time.strftime('%Y%m%d-%H%M')}"

    sweep_t0 = time.time()

    for i, lambda_val in enumerate(lambdas):
        print(f"\n{'='*60}")
        print(f"LAMBDA {i+1}/{len(lambdas)}: {lambda_val}")
        print(f"{'='*60}")

        # Skip if already completed (resume mode)
        if args.resume and lambda_val in completed_lambdas:
            print(f"  SKIPPED: lambda={lambda_val} already completed")
            continue
        if args.resume and lambda_already_done(output_dir, lambda_val):
            print(f"  SKIPPED: lambda={lambda_val} has results on disk")
            # Load and add to results
            summary_path = os.path.join(
                output_dir, f"lambda_{lambda_val:.2f}", "training_summary.json"
            )
            with open(summary_path) as f:
                summary = json.load(f)
            all_results.append(summary)
            completed_lambdas.add(lambda_val)
            continue

        # Fresh model for each lambda
        model, processor = setup_model()

        config = GRPOConfig(
            G=args.G,
            lambda_=lambda_val,
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

        lambda_dir = os.path.join(output_dir, f"lambda_{lambda_val:.2f}")
        summary = train_grpo(
            config, train_loader, eval_loader, model, processor,
            lambda_dir, device="cuda",
            wandb_run_name=f"grpo-lambda-{lambda_val:.2f}",
            eval_max_samples=args.eval_max_samples,
        )

        all_results.append(summary)
        completed_lambdas.add(lambda_val)

        # Free GPU memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save intermediate sweep results (crash recovery)
        sweep_path = os.path.join(output_dir, "sweep_results.json")
        with open(sweep_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Intermediate sweep results saved: {sweep_path}")

    sweep_elapsed = time.time() - sweep_t0

    # Final sweep summary
    sweep_summary = {
        "lambdas": lambdas,
        "n_lambdas": len(lambdas),
        "total_time_sec": sweep_elapsed,
        "total_time_hours": sweep_elapsed / 3600,
        "results": all_results,
    }
    sweep_path = os.path.join(output_dir, "sweep_results.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"LAMBDA SWEEP COMPLETE ({sweep_elapsed/3600:.1f} hours)")
    print(f"{'='*60}")
    print(f"  {'Lambda':>8} {'Steps':>6} {'Loss':>8} {'Best WER':>10} "
          f"{'Gap':>8} {'Early?':>6} {'Time':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")
    for r in all_results:
        lam = r.get("lambda", 0)
        steps = r.get("actual_steps", r.get("max_steps", "?"))
        loss = r.get("final_loss", None)
        wer = r.get("best_eval_wer", None)
        gap = r.get("best_fairness_gap", None)
        early = "Y" if r.get("stopped_early", False) else ""
        t = r.get("training_time_sec", 0)
        loss_s = f"{loss:>8.4f}" if loss is not None else f"{'N/A':>8}"
        wer_s = f"{wer:>10.4f}" if wer is not None else f"{'N/A':>10}"
        gap_s = f"{gap:>8.4f}" if gap is not None else f"{'N/A':>8}"
        print(f"  {lam:>8.2f} {steps:>6} {loss_s} {wer_s} "
              f"{gap_s} {early:>6} {t/60:>7.1f}m")

    print(f"\n  Results saved: {sweep_path}")
    print(f"  Next: python scripts/training/grpo_evaluate.py "
          f"--sweep_dir {output_dir}")

    return sweep_summary


def main():
    parser = argparse.ArgumentParser(
        description="Full-scale GRPO lambda sweep (Phase 5)")

    # Data
    parser.add_argument("--fs_manifest", default="outputs/manifests/fs_train.csv")
    parser.add_argument("--cv_manifest", default="outputs/manifests/cv_train.csv")
    parser.add_argument("--output_dir", default="outputs/grpo")

    # Lambda sweep
    parser.add_argument("--lambdas", type=float, nargs="+", default=None,
                        help="Lambda values to sweep (default: non-linear 8 values)")
    parser.add_argument("--coarse", action="store_true",
                        help="Use coarse lambda set (5 values) for initial exploration")
    parser.add_argument("--resume", action="store_true",
                        help="Skip lambda values that already have results")
    parser.add_argument("--wandb_project", default="asr-fairness-grpo")

    # GRPO hyperparameters (validated in Phase 4)
    parser.add_argument("--G", type=int, default=2,
                        help="Candidates per input (2=safe for T4, 4=if VRAM allows)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epsilon_low", type=float, default=0.2,
                        help="DAPO asymmetric clip lower bound")
    parser.add_argument("--epsilon_high", type=float, default=0.28,
                        help="DAPO asymmetric clip upper bound")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=1500,
                        help="Steps per lambda (1000-2000 recommended)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--dr_grpo", action="store_true", default=True)
    parser.add_argument("--no_dr_grpo", action="store_false", dest="dr_grpo")
    parser.add_argument("--baseline_wer", type=float, default=None,
                        help="SFT baseline WER (from Phase 3 adapter)")

    # Training schedule
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=50)

    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--early_stop_threshold", type=float, default=0.05)

    # Checkpoint selection
    parser.add_argument("--checkpoint_select_alpha", type=float, default=0.5)
    parser.add_argument("--eval_max_samples", type=int, default=50)

    args = parser.parse_args()

    # Resolve lambda set
    if args.lambdas is None:
        args.lambdas = COARSE_LAMBDAS if args.coarse else DEFAULT_LAMBDAS

    os.environ["WANDB_PROJECT"] = args.wandb_project
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    run_lambda_sweep(args)


if __name__ == "__main__":
    main()
