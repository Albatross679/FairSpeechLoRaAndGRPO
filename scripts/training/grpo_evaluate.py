"""
Post-training evaluation and Pareto frontier analysis for GRPO lambda sweep (Phase 5).

After grpo_sweep.py produces trained adapters for each lambda value, this script:
1. Evaluates each adapter's best checkpoint on Fair-Speech and Common Voice
2. Computes per-group WER and fairness metrics for each adapter
3. Identifies the Pareto frontier (non-dominated points in WER-fairness space)
4. Runs bootstrap pairwise significance tests between GRPO and baselines
5. Generates Pareto frontier scatter plot

Uses the existing evaluation bridge (evaluate_adapter.py) to ensure consistent
metrics with the rest of the pipeline.

Usage:
    # Evaluate all lambda checkpoints and generate Pareto frontier
    python scripts/training/grpo_evaluate.py \
        --sweep_dir outputs/grpo \
        --fs_manifest outputs/manifests/fs_train.csv \
        --cv_manifest outputs/manifests/cv_dev.csv \
        --output_dir outputs/grpo/pareto

    # Include SFT baseline for comparison
    python scripts/training/grpo_evaluate.py \
        --sweep_dir outputs/grpo \
        --baseline_adapter outputs/standard-lora/adapter \
        --baseline_name standard-lora \
        --output_dir outputs/grpo/pareto

    # Quick mode (skip bootstrap for speed)
    python scripts/training/grpo_evaluate.py \
        --sweep_dir outputs/grpo \
        --skip_bootstrap \
        --output_dir outputs/grpo/pareto
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jiwer
import numpy as np
import pandas as pd
import torch

from scripts.training.evaluate_adapter import (
    load_model_with_adapter,
    evaluate_on_dataset,
    print_gpu_memory,
)


# -- Pareto Frontier -----------------------------------------------------------

def is_dominated(point, others):
    """Check if a point is dominated by any point in others.

    A point (wer, gap) is dominated if there exists another point with
    both lower WER AND lower fairness gap.
    """
    for other in others:
        if other["mean_wer"] <= point["mean_wer"] and \
           other["fairness_gap"] <= point["fairness_gap"] and \
           (other["mean_wer"] < point["mean_wer"] or
            other["fairness_gap"] < point["fairness_gap"]):
            return True
    return False


def compute_pareto_frontier(results):
    """Identify non-dominated points in the WER-fairness space.

    Returns list of results on the Pareto frontier, sorted by WER.
    """
    frontier = []
    for i, point in enumerate(results):
        others = [r for j, r in enumerate(results) if j != i]
        if not is_dominated(point, others):
            frontier.append(point)
    frontier.sort(key=lambda x: x["mean_wer"])
    return frontier


# -- Bootstrap Pairwise Significance Testing -----------------------------------

def bootstrap_pairwise_test(
    predictions_a, predictions_b, n_bootstrap=1000, seed=42
):
    """Bootstrap paired test for WER difference between two models.

    Tests H0: WER_A == WER_B.
    Returns p-value (two-sided) and mean WER difference.
    """
    rng = np.random.RandomState(seed)

    # Align predictions by utterance_id
    merged = predictions_a.merge(
        predictions_b, on="utterance_id", suffixes=("_a", "_b")
    )
    if len(merged) < 20:
        return {"p_value": None, "mean_diff": None, "n_paired": len(merged),
                "error": "Too few paired samples"}

    # Per-utterance WER difference
    diffs = []
    for _, row in merged.iterrows():
        ref = str(row.get("reference_a", "")).strip()
        hyp_a = str(row.get("hypothesis_a", "")).strip()
        hyp_b = str(row.get("hypothesis_b", "")).strip()
        if not ref:
            continue
        wer_a = jiwer.wer(ref, hyp_a) if hyp_a else 1.0
        wer_b = jiwer.wer(ref, hyp_b) if hyp_b else 1.0
        diffs.append(wer_a - wer_b)

    diffs = np.array(diffs)
    if len(diffs) < 20:
        return {"p_value": None, "mean_diff": None, "n_paired": len(diffs),
                "error": "Too few valid pairs"}

    observed_diff = diffs.mean()

    # Bootstrap: resample diffs, compute mean
    boot_diffs = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_diffs.append(sample.mean())
    boot_diffs = np.array(boot_diffs)

    # Two-sided p-value: fraction of bootstrap diffs on wrong side of zero
    if observed_diff >= 0:
        p_value = (boot_diffs <= 0).mean() * 2
    else:
        p_value = (boot_diffs >= 0).mean() * 2
    p_value = min(p_value, 1.0)

    return {
        "p_value": float(p_value),
        "mean_diff": float(observed_diff),
        "ci_lower": float(np.percentile(boot_diffs, 2.5)),
        "ci_upper": float(np.percentile(boot_diffs, 97.5)),
        "n_paired": len(diffs),
    }


# -- Pareto Plot ----------------------------------------------------------------

def generate_pareto_plot(results, frontier, output_path, title=None):
    """Generate Pareto frontier scatter plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    # All points
    wers = [r["mean_wer"] for r in results]
    gaps = [r["fairness_gap"] for r in results]
    labels = [r.get("label", f"λ={r.get('lambda', '?')}") for r in results]
    colors = []
    for r in results:
        if r.get("is_baseline", False):
            colors.append("red")
        elif r in frontier:
            colors.append("blue")
        else:
            colors.append("gray")

    ax.scatter(wers, gaps, c=colors, s=80, zorder=3, edgecolors="black", linewidths=0.5)

    # Label points
    for w, g, label in zip(wers, gaps, labels):
        ax.annotate(label, (w, g), textcoords="offset points", xytext=(5, 5),
                    fontsize=8, alpha=0.8)

    # Frontier line
    if len(frontier) >= 2:
        f_wers = [r["mean_wer"] for r in frontier]
        f_gaps = [r["fairness_gap"] for r in frontier]
        ax.plot(f_wers, f_gaps, "b--", alpha=0.5, linewidth=1.5, label="Pareto frontier")

    ax.set_xlabel("Mean WER", fontsize=12)
    ax.set_ylabel("Fairness Gap (max group WER - min group WER)", fontsize=12)
    ax.set_title(title or "Accuracy-Fairness Pareto Frontier", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Color legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
               markersize=8, label="GRPO (Pareto)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, label="GRPO (dominated)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=8, label="Baseline"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Pareto plot saved: {output_path}")


# -- Main Evaluation Pipeline --------------------------------------------------

def evaluate_sweep(args):
    """Evaluate all lambda checkpoints and compute Pareto frontier."""
    sweep_dir = args.sweep_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load sweep results to find checkpoint paths
    sweep_results_path = os.path.join(sweep_dir, "sweep_results.json")
    if not os.path.exists(sweep_results_path):
        print(f"ERROR: No sweep results found at {sweep_results_path}")
        print("Run grpo_sweep.py first.")
        sys.exit(1)

    with open(sweep_results_path) as f:
        sweep_data = json.load(f)

    # Handle both old format (list) and new format (dict with "results" key)
    if isinstance(sweep_data, list):
        sweep_results = sweep_data
    else:
        sweep_results = sweep_data.get("results", sweep_data)

    print(f"\n{'='*60}")
    print(f"GRPO EVALUATION: {len(sweep_results)} lambda checkpoints")
    print(f"{'='*60}")

    all_eval_results = []

    # Evaluate each lambda's best checkpoint
    for i, run_result in enumerate(sweep_results):
        lambda_val = run_result.get("lambda", "?")
        ckpt_path = run_result.get("best_checkpoint_path")
        if not ckpt_path or not os.path.exists(ckpt_path):
            # Fallback to adapter directory
            ckpt_path = os.path.join(sweep_dir, f"lambda_{lambda_val:.2f}", "adapter")
            if not os.path.exists(ckpt_path):
                print(f"  SKIP lambda={lambda_val}: no checkpoint found")
                continue

        print(f"\n{'='*60}")
        print(f"Evaluating lambda={lambda_val} ({i+1}/{len(sweep_results)})")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        model, processor = load_model_with_adapter(ckpt_path, args.device)
        model_name = f"grpo-lambda-{lambda_val:.2f}"

        eval_result = {"lambda": lambda_val, "checkpoint_path": ckpt_path,
                       "label": f"λ={lambda_val}", "is_baseline": False}

        # Fair-Speech evaluation
        if os.path.exists(args.fs_manifest):
            fs_result = evaluate_on_dataset(
                model, processor, args.fs_manifest, "fairspeech", "ethnicity",
                model_name, output_dir, args.device,
                n_bootstrap=args.n_bootstrap, skip_bootstrap=args.skip_bootstrap,
            )
            eval_result["fairspeech"] = fs_result
            eval_result["mean_wer"] = fs_result["overall_wer"]
            fm = fs_result.get("fairness_metrics", {})
            eval_result["fairness_gap"] = fm.get("wer_range", 0.0)
            eval_result["max_min_ratio"] = fm.get("max_min_ratio", 1.0)

        # Common Voice evaluation
        if os.path.exists(args.cv_manifest):
            cv_result = evaluate_on_dataset(
                model, processor, args.cv_manifest, "commonvoice", "accent",
                model_name, output_dir, args.device,
                n_bootstrap=args.n_bootstrap, skip_bootstrap=args.skip_bootstrap,
            )
            eval_result["commonvoice"] = cv_result

        all_eval_results.append(eval_result)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Evaluate baseline adapter if provided
    if args.baseline_adapter and os.path.exists(args.baseline_adapter):
        print(f"\n{'='*60}")
        print(f"Evaluating baseline: {args.baseline_name}")
        print(f"{'='*60}")

        model, processor = load_model_with_adapter(args.baseline_adapter, args.device)

        baseline_result = {
            "lambda": None, "checkpoint_path": args.baseline_adapter,
            "label": args.baseline_name, "is_baseline": True,
        }

        if os.path.exists(args.fs_manifest):
            fs_result = evaluate_on_dataset(
                model, processor, args.fs_manifest, "fairspeech", "ethnicity",
                args.baseline_name, output_dir, args.device,
                n_bootstrap=args.n_bootstrap, skip_bootstrap=args.skip_bootstrap,
            )
            baseline_result["fairspeech"] = fs_result
            baseline_result["mean_wer"] = fs_result["overall_wer"]
            fm = fs_result.get("fairness_metrics", {})
            baseline_result["fairness_gap"] = fm.get("wer_range", 0.0)

        if os.path.exists(args.cv_manifest):
            cv_result = evaluate_on_dataset(
                model, processor, args.cv_manifest, "commonvoice", "accent",
                args.baseline_name, output_dir, args.device,
                n_bootstrap=args.n_bootstrap, skip_bootstrap=args.skip_bootstrap,
            )
            baseline_result["commonvoice"] = cv_result

        all_eval_results.append(baseline_result)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute Pareto frontier (only on points with valid metrics)
    valid_results = [r for r in all_eval_results
                     if r.get("mean_wer") is not None and r.get("fairness_gap") is not None]
    frontier = compute_pareto_frontier(valid_results)

    print(f"\n{'='*60}")
    print(f"PARETO FRONTIER: {len(frontier)} non-dominated points "
          f"out of {len(valid_results)} evaluated")
    print(f"{'='*60}")
    for r in frontier:
        print(f"  {r['label']}: WER={r['mean_wer']:.4f}, gap={r['fairness_gap']:.4f}")

    # Pairwise significance tests (GRPO vs baseline)
    significance_results = {}
    if args.baseline_adapter and not args.skip_bootstrap:
        baseline_preds = _load_predictions(output_dir, args.baseline_name, "fairspeech")
        if baseline_preds is not None:
            for r in all_eval_results:
                if r.get("is_baseline"):
                    continue
                model_name = f"grpo-lambda-{r['lambda']:.2f}"
                model_preds = _load_predictions(output_dir, model_name, "fairspeech")
                if model_preds is not None:
                    sig = bootstrap_pairwise_test(baseline_preds, model_preds)
                    significance_results[model_name] = sig
                    star = "***" if sig["p_value"] and sig["p_value"] < 0.001 else \
                           "**" if sig["p_value"] and sig["p_value"] < 0.01 else \
                           "*" if sig["p_value"] and sig["p_value"] < 0.05 else ""
                    print(f"  {model_name} vs {args.baseline_name}: "
                          f"Δ={sig['mean_diff']:.4f}, p={sig['p_value']:.4f} {star}")

    # Generate Pareto plot
    plot_path = os.path.join(output_dir, "pareto_frontier.png")
    if valid_results:
        generate_pareto_plot(valid_results, frontier, plot_path)

    # Save full results
    results_path = os.path.join(output_dir, "pareto_results.json")
    serializable_results = []
    for r in all_eval_results:
        entry = {k: v for k, v in r.items()
                 if k not in ("fairspeech", "commonvoice")}
        if "fairspeech" in r:
            entry["fairspeech_wer"] = r["fairspeech"].get("overall_wer")
            entry["fairspeech_fairness"] = r["fairspeech"].get("fairness_metrics")
        if "commonvoice" in r:
            entry["commonvoice_wer"] = r["commonvoice"].get("overall_wer")
            entry["commonvoice_fairness"] = r["commonvoice"].get("fairness_metrics")
        serializable_results.append(entry)

    output = {
        "evaluation_date": time.strftime("%Y-%m-%d"),
        "n_evaluated": len(all_eval_results),
        "pareto_frontier": [r["label"] for r in frontier],
        "significance_tests": significance_results,
        "results": serializable_results,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Full results saved: {results_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Label':>20} {'WER':>8} {'Gap':>8} {'Pareto?':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for r in sorted(valid_results, key=lambda x: x.get("mean_wer", 999)):
        on_frontier = "✓" if r in frontier else ""
        print(f"  {r['label']:>20} {r['mean_wer']:>8.4f} "
              f"{r['fairness_gap']:>8.4f} {on_frontier:>8}")

    return output


def _load_predictions(output_dir, model_name, dataset_name):
    """Load prediction CSV for a model+dataset pair."""
    csv_path = os.path.join(output_dir, f"predictions_{model_name}_{dataset_name}.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO sweep checkpoints and compute Pareto frontier")

    parser.add_argument("--sweep_dir", required=True,
                        help="Directory containing grpo_sweep.py outputs")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: sweep_dir/pareto)")
    parser.add_argument("--fs_manifest", default="outputs/manifests/fs_train.csv")
    parser.add_argument("--cv_manifest", default="outputs/manifests/cv_dev.csv")
    parser.add_argument("--device", default="cuda")

    # Baseline comparison
    parser.add_argument("--baseline_adapter", default=None,
                        help="Path to SFT baseline adapter for comparison")
    parser.add_argument("--baseline_name", default="standard-lora",
                        help="Name for the baseline model")

    # Options
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--skip_bootstrap", action="store_true",
                        help="Skip bootstrap CIs and significance tests (faster)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.sweep_dir, "pareto")

    evaluate_sweep(args)


if __name__ == "__main__":
    main()
