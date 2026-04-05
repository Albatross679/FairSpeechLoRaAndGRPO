#!/usr/bin/env python3
"""
Generate all publication-quality figures for COLM2026 paper.

This is a unified plotting script that generates Figures 1–12 from
the pre-computed analysis JSONs. Each figure is saved as PNG (300 DPI).

Usage:
    python scripts/generate_all_plots.py [--cv_analysis results/commonvoice/analysis/full_analysis.json]
                                         [--fs_analysis results/fairspeech/analysis/full_analysis_fs.json]
                                         [--output_dir results/figures/]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Model ordering: generation-first, then compression (low → medium → high)
MODEL_ORDER = [
    "wav2vec2-large",         # Gen 1
    "whisper-small",          # Gen 2
    "whisper-medium",         # Gen 2
    "whisper-large-v3",       # Gen 2
    "qwen3-asr-0.6b",        # Gen 3, low compression
    "qwen3-asr-1.7b",        # Gen 3, low compression
    "canary-qwen-2.5b",      # Gen 3, medium compression
    "granite-speech-3.3-2b",  # Gen 3, high compression
    "granite-speech-3.3-8b",  # Gen 3, high compression
]

MODEL_SHORT = {
    "wav2vec2-large": "W2V2-L",
    "whisper-small": "Wh-S",
    "whisper-medium": "Wh-M",
    "whisper-large-v3": "Wh-L",
    "qwen3-asr-0.6b": "Q3-0.6B",
    "qwen3-asr-1.7b": "Q3-1.7B",
    "granite-speech-3.3-2b": "Gr-2B",
    "granite-speech-3.3-8b": "Gr-8B",
    "canary-qwen-2.5b": "Can-2.5B",
}

MODEL_PARAMS = {
    "wav2vec2-large": 0.317,
    "whisper-small": 0.244,
    "whisper-medium": 0.764,
    "whisper-large-v3": 1.5,
    "qwen3-asr-0.6b": 0.6,
    "qwen3-asr-1.7b": 1.7,
    "granite-speech-3.3-2b": 2.0,
    "granite-speech-3.3-8b": 8.0,
    "canary-qwen-2.5b": 2.5,
}

MODEL_GEN = {
    "wav2vec2-large": 1,
    "whisper-small": 2, "whisper-medium": 2, "whisper-large-v3": 2,
    "qwen3-asr-0.6b": 3, "qwen3-asr-1.7b": 3,
    "granite-speech-3.3-2b": 3, "granite-speech-3.3-8b": 3, "canary-qwen-2.5b": 3,
}

# Distinct colors
MODEL_COLORS = {
    "wav2vec2-large":        "#2E86AB",  # Steel blue
    "whisper-small":         "#F18F01",  # Amber
    "whisper-medium":        "#3B7A57",  # Forest green
    "whisper-large-v3":      "#A23B72",  # Plum
    "qwen3-asr-0.6b":       "#FFFF00",  # Yellow
    "qwen3-asr-1.7b":       "#44AF69",  # Green
    "granite-speech-3.3-2b": "#6B4C9A",  # Purple
    "granite-speech-3.3-8b": "#E84855",  # Coral red
    "canary-qwen-2.5b":     "#D4A574",  # Tan
}

GEN_COLORS = {1: "#2E86AB", 2: "#C73E1D", 3: "#44AF69"}
GEN_LABELS = {1: "Gen 1 (CTC)", 2: "Gen 2 (Enc-Dec)", 3: "Gen 3 (LLM-ASR)"}

# Logical x-axis orderings
ETHNICITY_ORDER = ["White", "Black/AA", "Hispanic", "Asian", "Native American", "Pacific Islander", "Middle Eastern"]
ACCENT_ORDER = ["us", "england", "canada", "australia", "indian", "african"]
GENDER_ORDER = ["female", "male"]
SES_ORDER = ["Low", "Medium", "Affluent"]
L1_GROUP_ORDER = ["English", "Spanish", "Mandarin", "Hindi", "Other"]

# CV age uses its own labels
CV_AGE_ORDER = ["teens", "twenties", "thirties", "forties", "fifties"]
FS_AGE_ORDER = ["18-22", "23-30", "31-45", "46-65"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_analysis", type=str,
                        default="/users/PAS2030/srishti/asr_fairness/results/commonvoice/analysis/full_analysis.json")
    parser.add_argument("--cv_error_decomp", type=str,
                        default="/users/PAS2030/srishti/asr_fairness/results/commonvoice/analysis/error_decomposition.json")
    parser.add_argument("--fs_analysis", type=str,
                        default="/users/PAS2030/srishti/asr_fairness/results/fairspeech/analysis/full_analysis_fs.json")
    parser.add_argument("--output_dir", type=str,
                        default="/users/PAS2030/srishti/asr_fairness/results/figures")
    return parser.parse_args()


def save_fig(fig, output_dir, name):
    """Save figure as PNG and PDF (PDF for LaTeX, PNG for preview)."""
    # Ensure PDF text is crisp (embed as TrueType, not Type 3)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved: {name}")


def get_ordered_models(data):
    """Return models in generation-first, size-second order."""
    return [m for m in MODEL_ORDER if m in data]


def get_group_wers(data, model, axis):
    """Extract group WERs for a model and axis."""
    if model not in data:
        return {}
    axis_data = data[model].get(axis, {})
    gw = axis_data.get("group_wers", {})
    return {k: v.get("wer", 0) for k, v in gw.items()}


def get_mmr(model_data, axis):
    """Get MMR from model data, handling both key names."""
    axis_data = model_data.get(axis, {})
    if "max_min_ratio" in axis_data:
        return axis_data["max_min_ratio"]
    return axis_data.get("mmr", np.nan)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Ethnicity WER Heatmap (Fair-Speech)
# ═══════════════════════════════════════════════════════════════════════════
def fig1_ethnicity_heatmap(fs_data, output_dir):
    """9 models x 7 ethnicities annotated heatmap, columns sorted by avg WER."""
    models = get_ordered_models(fs_data)

    # Compute average WER per ethnicity across all models to sort columns
    avg_wer = {}
    for g in ETHNICITY_ORDER:
        vals = []
        for m in models:
            gw = get_group_wers(fs_data, m, "ethnicity")
            v = gw.get(g, np.nan)
            if not np.isnan(v):
                vals.append(v)
        avg_wer[g] = np.mean(vals) if vals else 0
    groups = sorted(ETHNICITY_ORDER, key=lambda g: avg_wer[g])

    matrix = []
    for m in models:
        gw = get_group_wers(fs_data, m, "ethnicity")
        row = [gw.get(g, np.nan) * 100 for g in groups]
        matrix.append(row)

    df = pd.DataFrame(matrix, index=[MODEL_SHORT.get(m, m) for m in models], columns=groups)

    fig, ax = plt.subplots(figsize=(12, max(3, len(models))))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "WER (%)"}, annot_kws={"fontsize": 11})
    ax.set_title("WER (%) by Ethnicity (Fair-Speech)", fontsize=18)
    ax.set_xlabel("Ethnicity", fontsize=16)
    ax.set_ylabel("Model", fontsize=16)
    ax.tick_params(axis='x', rotation=30, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    save_fig(fig, output_dir, "fig1_ethnicity_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Accent WER Heatmap (Common Voice)
# ═══════════════════════════════════════════════════════════════════════════
def fig2_accent_heatmap(cv_data, output_dir):
    """9 models x 6 accents annotated heatmap."""
    models = get_ordered_models(cv_data)
    groups = ACCENT_ORDER

    matrix = []
    for m in models:
        gw = get_group_wers(cv_data, m, "accent")
        row = [gw.get(g, np.nan) * 100 for g in groups]
        matrix.append(row)

    df = pd.DataFrame(matrix, index=[MODEL_SHORT.get(m, m) for m in models], columns=groups)

    fig, ax = plt.subplots(figsize=(10, max(3, len(models))))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "WER (%)"}, annot_kws={"fontsize": 11})
    ax.set_title("WER (%) by Accent (Common Voice)", fontsize=18)
    ax.set_xlabel("Accent", fontsize=16)
    ax.set_ylabel("Model", fontsize=16)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    save_fig(fig, output_dir, "fig2_accent_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Whisper 3-point Scaling Trajectory
# ═══════════════════════════════════════════════════════════════════════════
def fig3_whisper_scaling(cv_data, fs_data, cv_error_decomp, output_dir):
    """Multi-panel: accent MMR, ethnicity MMR, Indian insertion rate vs params."""
    whisper_models = ["whisper-small", "whisper-medium", "whisper-large-v3"]
    params = [MODEL_PARAMS[m] for m in whisper_models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Accent MMR (CV)
    ax = axes[0]
    mmrs = [get_mmr(cv_data.get(m, {}), "accent") for m in whisper_models]
    ax.plot(params, mmrs, 'o-', color="#C73E1D", markersize=10, linewidth=2.5)
    for x, y, m in zip(params, mmrs, whisper_models):
        ax.annotate(MODEL_SHORT[m], (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel("Accent MMR")
    ax.set_title("(a) Accent Fairness")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Ethnicity MMR (FS)
    ax = axes[1]
    mmrs = [get_mmr(fs_data.get(m, {}), "ethnicity") for m in whisper_models]
    ax.plot(params, mmrs, 's-', color="#A23B72", markersize=10, linewidth=2.5)
    for x, y, m in zip(params, mmrs, whisper_models):
        ax.annotate(MODEL_SHORT[m], (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel("Ethnicity MMR")
    ax.set_title("(b) Ethnicity Fairness")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Indian accent insertion rate (from error_decomposition.json)
    ax = axes[2]
    ins_rates = []
    for m in whisper_models:
        ed = cv_error_decomp.get(m, {}).get("accent", {}).get("indian", {})
        rate = ed.get("insertion_rate", np.nan)
        ins_rates.append(rate * 100 if not np.isnan(rate) else np.nan)
    valid = [(p, r, m) for p, r, m in zip(params, ins_rates, whisper_models) if not np.isnan(r)]
    if valid:
        vp, vr, vm = zip(*valid)
        ax.plot(vp, vr, 'D-', color="#F18F01", markersize=10, linewidth=2.5)
        for x, y, m in zip(vp, vr, vm):
            ax.annotate(MODEL_SHORT[m], (x, y), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel("Insertion Rate (%)")
    ax.set_title("(c) Indian Accent Insertions")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Whisper Scaling Trajectory: Fairness Degrades with Scale", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, output_dir, "fig3_whisper_scaling")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Compression x Fairness (Gen 3)
# ═══════════════════════════════════════════════════════════════════════════
def fig5_compression_fairness(cv_data, fs_data, output_dir):
    """Scatter: MMR vs compression level for Gen 3 models."""
    gen3 = ["qwen3-asr-0.6b", "qwen3-asr-1.7b", "granite-speech-3.3-2b",
            "granite-speech-3.3-8b", "canary-qwen-2.5b"]
    compression = {
        "qwen3-asr-0.6b": 1, "qwen3-asr-1.7b": 1,
        "canary-qwen-2.5b": 2,
        "granite-speech-3.3-2b": 3, "granite-speech-3.3-8b": 3,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (data, axis, title) in zip(axes, [
        (cv_data, "accent", "(a) Common Voice"),
        (fs_data, "ethnicity", "(b) Fair-Speech"),
    ]):
        for m in gen3:
            if m in data:
                mmr = get_mmr(data[m], axis)
                if not np.isnan(mmr):
                    ax.scatter(compression[m], mmr, s=150, c=MODEL_COLORS[m],
                              edgecolors="black", linewidth=1, zorder=5)
                    ax.annotate(MODEL_SHORT[m], (compression[m], mmr),
                               textcoords="offset points", xytext=(8, 5), fontsize=9)
        ylabel = "Accent MMR" if axis == "accent" else "Ethnicity MMR"
        ax.set_xlabel("Compression Level (1=Low, 3=High)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Low\n(Qwen3)", "Medium\n(Canary)", "High\n(Granite)"])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Audio Compression Level Predicts Fairness (Gen 3 Models)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, output_dir, "fig5_compression_fairness")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: L1 WER Bar Chart (Fair-Speech)
# ═══════════════════════════════════════════════════════════════════════════
def fig6_l1_analysis(fs_data, output_dir):
    """Grouped bar chart: WER by L1 group x models."""
    models = get_ordered_models(fs_data)
    groups = L1_GROUP_ORDER

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(models))

    for i, m in enumerate(models):
        gw = get_group_wers(fs_data, m, "l1_group")
        wers = [gw.get(g, 0) * 100 for g in groups]
        ax.bar(x + i * width, wers, width, label=MODEL_SHORT.get(m, m),
               color=MODEL_COLORS.get(m, "#999"), edgecolor="white", linewidth=0.5)

    ax.set_xlabel("First Language (L1)", fontsize=16)
    ax.set_ylabel("WER (%)", fontsize=16)
    ax.set_title("WER by First Language (Fair-Speech)", fontsize=18)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(groups, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(ncol=3, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, output_dir, "fig6_l1_analysis")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Fairness-Accuracy Pareto (Fair-Speech)
# ═══════════════════════════════════════════════════════════════════════════
def fig7_pareto(fs_data, output_dir):
    """Scatter: Overall WER vs Ethnicity MMR, colored by generation, with Pareto frontier."""
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect all points for Pareto frontier
    points = []
    for m in MODEL_ORDER:
        if m not in fs_data:
            continue
        wer = fs_data[m].get("overall_wer", np.nan)
        mmr = get_mmr(fs_data[m], "ethnicity")
        if np.isnan(wer) or np.isnan(mmr):
            continue
        gen = MODEL_GEN[m]
        points.append((wer * 100, mmr, m, gen))
        ax.scatter(wer * 100, mmr, s=180, c=GEN_COLORS[gen],
                  edgecolors="black", linewidth=1, zorder=5)
        ax.annotate(MODEL_SHORT[m], (wer * 100, mmr),
                   textcoords="offset points", xytext=(8, 5), fontsize=12)

    # Compute and draw Pareto frontier (lower WER AND lower MMR = better)
    # A point is non-dominated if no other point is better on BOTH axes
    if points:
        pareto = []
        for w, r, m, g in points:
            dominated = any(w2 <= w and r2 <= r and (w2 < w or r2 < r)
                           for w2, r2, _, _ in points)
            if not dominated:
                pareto.append((w, r))
        pareto.sort(key=lambda p: p[0])
        if len(pareto) >= 2:
            pw, pr = zip(*pareto)
            ax.plot(pw, pr, '--', color='#555555', linewidth=1.5, alpha=0.7, zorder=3,
                    label='Pareto frontier')

    handles = [Patch(facecolor=GEN_COLORS[g], edgecolor="black", label=GEN_LABELS[g])
               for g in [1, 2, 3]]
    ax.legend(handles=handles, fontsize=12)
    ax.set_xlabel("Overall WER (%)", fontsize=16)
    ax.set_ylabel("Ethnicity MMR (higher = less fair)", fontsize=16)
    ax.set_title("Accuracy vs. Fairness (Fair-Speech)", fontsize=18)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, output_dir, "fig7_pareto")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Scaling x Fairness (3 lines)
# ═══════════════════════════════════════════════════════════════════════════
def fig8_scaling_curves(cv_data, fs_data, output_dir):
    """X=params(log), Y=MMR. Three lines: Whisper, Qwen3, Granite."""
    scaling_families = {
        "Whisper": (["whisper-small", "whisper-medium", "whisper-large-v3"], "#C73E1D", "o"),
        "Qwen3": (["qwen3-asr-0.6b", "qwen3-asr-1.7b"], "#44AF69", "s"),
        "Granite": (["granite-speech-3.3-2b", "granite-speech-3.3-8b"], "#6B4C9A", "D"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (data, axis, title) in zip(axes, [
        (cv_data, "accent", "Common Voice: Accent MMR"),
        (fs_data, "ethnicity", "Fair-Speech: Ethnicity MMR"),
    ]):
        for family, (models, color, marker) in scaling_families.items():
            params, mmrs = [], []
            for m in models:
                if m in data:
                    mmr = get_mmr(data[m], axis)
                    if not np.isnan(mmr):
                        params.append(MODEL_PARAMS[m])
                        mmrs.append(mmr)

            if params:
                ax.plot(params, mmrs, f'{marker}-', color=color, markersize=10,
                       linewidth=2.5, label=family)
                for x, y, m in zip(params, mmrs, [m for m in models if m in data]):
                    ax.annotate(MODEL_SHORT[m], (x, y), textcoords="offset points",
                               xytext=(0, 10), ha='center', fontsize=8)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))
        ax.set_xlabel("Parameters (log scale)")
        ax.set_ylabel("Max/Min Ratio (MMR)")
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Scaling x Fairness: Three Architecture Families", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, output_dir, "fig8_scaling_curves")


# ═══════════════════════════════════════════════════════════════════════════
# Generic grouped bar chart (for Age, Gender, SES)
# ═══════════════════════════════════════════════════════════════════════════
def plot_grouped_bars(data, axis, groups, output_dir, title, fname):
    """Generic grouped bar chart for a demographic axis."""
    models = get_ordered_models(data)
    if not models:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(models))

    for i, m in enumerate(models):
        gw = get_group_wers(data, m, axis)
        wers = [gw.get(g, 0) * 100 for g in groups]
        ax.bar(x + i * width, wers, width, label=MODEL_SHORT.get(m, m),
               color=MODEL_COLORS.get(m, "#999"), edgecolor="white", linewidth=0.5)

    ax.set_xlabel(axis.replace("_", " ").title(), fontsize=16)
    ax.set_ylabel("WER (%)", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(groups, rotation=45 if len(groups) > 5 else 0,
                       ha="right" if len(groups) > 5 else "center", fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(ncol=3, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, output_dir, fname)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load analysis data
    cv_data, fs_data, cv_error_decomp = {}, {}, {}

    if os.path.exists(args.cv_analysis):
        with open(args.cv_analysis) as f:
            cv_data = json.load(f)
        print(f"Loaded CV analysis: {len(cv_data)} models")
    else:
        print(f"CV analysis not found: {args.cv_analysis}")

    if os.path.exists(args.fs_analysis):
        with open(args.fs_analysis) as f:
            fs_data = json.load(f)
        print(f"Loaded FS analysis: {len(fs_data)} models")
    else:
        print(f"FS analysis not found: {args.fs_analysis}")

    if os.path.exists(args.cv_error_decomp):
        with open(args.cv_error_decomp) as f:
            cv_error_decomp = json.load(f)
        print(f"Loaded CV error decomposition: {len(cv_error_decomp)} models")
    else:
        print(f"CV error decomposition not found: {args.cv_error_decomp}")

    # ── Generate figures ────────────────────────────────────────────
    if fs_data:
        print("\nGenerating Fair-Speech figures...")
        fig1_ethnicity_heatmap(fs_data, args.output_dir)
        fig6_l1_analysis(fs_data, args.output_dir)
        fig7_pareto(fs_data, args.output_dir)

        # Appendix figures
        plot_grouped_bars(fs_data, "age", FS_AGE_ORDER, args.output_dir,
                         "WER by Age (Fair-Speech)", "fig10_age_fs")
        plot_grouped_bars(fs_data, "gender", GENDER_ORDER, args.output_dir,
                         "WER by Gender (Fair-Speech)", "fig11_gender_fs")
        plot_grouped_bars(fs_data, "socioeconomic", SES_ORDER, args.output_dir,
                         "WER by Socioeconomic Status (Fair-Speech)", "fig12_ses_fs")

    if cv_data:
        print("\nGenerating Common Voice figures...")
        fig2_accent_heatmap(cv_data, args.output_dir)

        # CV age uses its own labels (teens, twenties, etc.)
        plot_grouped_bars(cv_data, "age", CV_AGE_ORDER, args.output_dir,
                         "WER by Age (Common Voice)", "fig10_age_cv")
        plot_grouped_bars(cv_data, "gender", GENDER_ORDER, args.output_dir,
                         "WER by Gender (Common Voice)", "fig11_gender_cv")

    if cv_data and fs_data:
        print("\nGenerating cross-dataset figures...")
        fig3_whisper_scaling(cv_data, fs_data, cv_error_decomp, args.output_dir)
        fig5_compression_fairness(cv_data, fs_data, args.output_dir)
        fig8_scaling_curves(cv_data, fs_data, args.output_dir)

    print(f"\nAll figures in: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
