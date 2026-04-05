#!/usr/bin/env python3
"""
Generate publication-quality perturbation figures for COLM2026 paper.

Reads pre-computed CSVs from results/tables/ and results/figures/ (output of
compute_perturbation_metrics.py) and generates PDF + PNG plots.

Figures generated:
  P1. Fairness amplification heatmap (ethnicity × FS, accent × CV)
  P2. WERD by generation under SNR (ethnicity × FS)
  P3. Hallucination type distribution under masking (all models)
  P4. Pareto under degradation (accuracy vs fairness at each perturbation)
  P5. Insertion rate by demographic group under perturbation (ethnicity × FS)
  P6. WER degradation curves by perturbation type (all models)

Usage:
    python scripts/generate_perturbation_plots.py
    python scripts/generate_perturbation_plots.py --output_dir results/figures/
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Constants (match generate_all_plots.py) ──────────────────────────────────
PROJECT_DIR = "/users/PAS2030/srishti/asr_fairness"
TABLES_DIR = os.path.join(PROJECT_DIR, "results/tables")
FIGURES_DIR = os.path.join(PROJECT_DIR, "results/figures")

MODEL_ORDER = [
    "wav2vec2-large",
    "whisper-small", "whisper-medium", "whisper-large-v3",
    "qwen3-asr-0.6b", "qwen3-asr-1.7b",
    "canary-qwen-2.5b",
    "granite-speech-3.3-2b", "granite-speech-3.3-8b",
]

MODEL_SHORT = {
    "wav2vec2-large": "W2V2-L",
    "whisper-small": "Wh-S",
    "whisper-medium": "Wh-M",
    "whisper-large-v3": "Wh-L",
    "qwen3-asr-0.6b": "Q3-0.6B",
    "qwen3-asr-1.7b": "Q3-1.7B",
    "canary-qwen-2.5b": "Can-2.5B",
    "granite-speech-3.3-2b": "Gr-2B",
    "granite-speech-3.3-8b": "Gr-8B",
}

MODEL_GEN = {
    "wav2vec2-large": 1,
    "whisper-small": 2, "whisper-medium": 2, "whisper-large-v3": 2,
    "qwen3-asr-0.6b": 3, "qwen3-asr-1.7b": 3,
    "canary-qwen-2.5b": 3,
    "granite-speech-3.3-2b": 3, "granite-speech-3.3-8b": 3,
}

MODEL_COLORS = {
    "wav2vec2-large":        "#2E86AB",
    "whisper-small":         "#F18F01",
    "whisper-medium":        "#3B7A57",
    "whisper-large-v3":      "#A23B72",
    "qwen3-asr-0.6b":       "#C5B800",
    "qwen3-asr-1.7b":       "#44AF69",
    "canary-qwen-2.5b":     "#D4A574",
    "granite-speech-3.3-2b": "#6B4C9A",
    "granite-speech-3.3-8b": "#E84855",
}

GEN_COLORS = {1: "#2E86AB", 2: "#C73E1D", 3: "#44AF69"}
GEN_LABELS = {1: "Gen 1 (CTC)", 2: "Gen 2 (Enc-Dec)", 3: "Gen 3 (LLM-ASR)"}

PERTURBATION_ORDER = [
    "snr_20db", "snr_10db", "snr_0db",
    "reverb_0.3s", "reverb_0.6s", "reverb_1.0s",
    "silence_25pct", "silence_50pct", "silence_75pct",
    "mask_10pct", "mask_20pct", "mask_30pct",
]

PERTURBATION_SHORT = {
    "snr_20db": "SNR\n20dB", "snr_10db": "SNR\n10dB", "snr_0db": "SNR\n0dB",
    "reverb_0.3s": "Rev\n0.3s", "reverb_0.6s": "Rev\n0.6s", "reverb_1.0s": "Rev\n1.0s",
    "silence_25pct": "Sil\n25%", "silence_50pct": "Sil\n50%", "silence_75pct": "Sil\n75%",
    "mask_10pct": "Mask\n10%", "mask_20pct": "Mask\n20%", "mask_30pct": "Mask\n30%",
}

PERTURBATION_TYPE_ORDER = ["snr", "reverb", "silence", "mask"]
PERTURBATION_TYPE_LABELS = {
    "snr": "Additive Noise", "reverb": "Reverberation",
    "silence": "Silence Injection", "mask": "Chunk Masking",
}

ETHNICITY_ORDER = ["White", "Black/AA", "Hispanic", "Asian", "Native American", "Pacific Islander", "Middle Eastern"]


def save_fig(fig, output_dir, name):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# P1: FAIRNESS AMPLIFICATION HEATMAP
# ═════════════════════════════════════════════════════════════════════════════

def plot_fairness_amplification_heatmap(output_dir, axis, dataset):
    """Heatmap of MMR_perturbed / MMR_clean (9 models × 12 conditions)."""
    ds_label = "fs" if dataset == "fs" else "cv"
    path = os.path.join(FIGURES_DIR, f"fig_fairness_amplification_heatmap_{axis}_{ds_label}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    df = pd.read_csv(path)

    # Pivot to matrix
    df["model_short"] = df["model"].map(MODEL_SHORT)
    df["pert_short"] = df["perturbation"].map(PERTURBATION_SHORT)

    # Ensure ordering
    model_order = [MODEL_SHORT[m] for m in MODEL_ORDER if m in df["model"].values]
    pert_order = [PERTURBATION_SHORT[p] for p in PERTURBATION_ORDER if p in df["perturbation"].values]

    pivot = df.pivot(index="model_short", columns="pert_short", values="amplification")
    pivot = pivot.reindex(index=model_order, columns=pert_order)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Diverging colormap centered at 1.0 (amplification = 1 means no change)
    vmin = min(0.5, pivot.min().min())
    vmax = max(1.5, pivot.max().max())

    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn_r", center=1.0,
        vmin=vmin, vmax=vmax,
        annot=True, fmt=".2f", annot_kws={"fontsize": 9},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Fairness Amplification\n(MMR_pert / MMR_clean)", "shrink": 0.8},
    )

    # Add vertical lines to separate perturbation types
    for i in [3, 6, 9]:
        ax.axvline(x=i, color="black", linewidth=1.5)

    # Add horizontal line to separate generations
    gen_boundaries = []
    prev_gen = None
    for i, m in enumerate(model_order):
        model_full = [k for k, v in MODEL_SHORT.items() if v == m][0]
        gen = MODEL_GEN[model_full]
        if prev_gen is not None and gen != prev_gen:
            gen_boundaries.append(i)
        prev_gen = gen
    for b in gen_boundaries:
        ax.axhline(y=b, color="black", linewidth=1.5)

    axis_label = axis.replace("_", " ").title()
    ds_name = "Fair-Speech" if dataset == "fs" else "Common Voice"
    ax.set_title(f"Fairness Gap Amplification Under Perturbation: {axis_label} ({ds_name})", fontsize=14, pad=12)
    ax.set_ylabel("")
    ax.set_xlabel("")

    # Perturbation type labels at top
    type_positions = [(1.5, "Noise"), (4.5, "Reverb"), (7.5, "Silence"), (10.5, "Masking")]
    for pos, label in type_positions:
        if pos < len(pert_order):
            ax.text(pos, -0.8, label, ha="center", va="bottom", fontsize=11, fontweight="bold",
                    transform=ax.get_xaxis_transform())

    fig.tight_layout()
    save_fig(fig, output_dir, f"perturbation_amplification_heatmap_{axis}_{ds_label}")


# ═════════════════════════════════════════════════════════════════════════════
# P2: WERD BY GENERATION UNDER SNR
# ═════════════════════════════════════════════════════════════════════════════

def plot_werd_by_generation(output_dir, axis, dataset):
    """WERD at each SNR level, grouped by generation."""
    ds_label = "fs" if dataset == "fs" else "cv"
    path = os.path.join(FIGURES_DIR, f"fig_werd_by_generation_and_snr_{axis}_{ds_label}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    df = pd.read_csv(path)

    # Average WERD across groups for each model × perturbation
    avg = df.groupby(["model", "generation", "perturbation"])["werd"].mean().reset_index()

    snr_order = ["snr_20db", "snr_10db", "snr_0db"]
    snr_labels = ["20 dB", "10 dB", "0 dB"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODEL_ORDER:
        mdata = avg[avg["model"] == model]
        if mdata.empty:
            continue
        gen = MODEL_GEN[model]
        werds = []
        for s in snr_order:
            row = mdata[mdata["perturbation"] == s]
            werds.append(row["werd"].values[0] * 100 if len(row) > 0 else np.nan)

        ax.plot(snr_labels, werds,
                marker="o", markersize=6,
                color=MODEL_COLORS[model],
                linewidth=2, alpha=0.8,
                label=MODEL_SHORT[model])

    ax.set_xlabel("SNR Level", fontsize=14)
    ax.set_ylabel("Mean WERD (%)", fontsize=14)
    axis_label = axis.replace("_", " ").title()
    ds_name = "Fair-Speech" if dataset == "fs" else "Common Voice"
    ax.set_title(f"WER Degradation Under Noise: {axis_label} ({ds_name})", fontsize=14)
    ax.legend(fontsize=9, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    save_fig(fig, output_dir, f"perturbation_werd_snr_{axis}_{ds_label}")


# ═════════════════════════════════════════════════════════════════════════════
# P3: HALLUCINATION TYPE DISTRIBUTION UNDER MASKING
# ═════════════════════════════════════════════════════════════════════════════

def plot_hallucination_types(output_dir, perturbation_type, dataset):
    """Stacked bar chart: repetition vs syntactic vs content insertions."""
    ds_label = "fs" if dataset == "fs" else "cv"
    path = os.path.join(TABLES_DIR, f"table_hallucination_classification_{perturbation_type}_{ds_label}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    df = pd.read_csv(path)

    # Use the harshest level for each perturbation type
    if perturbation_type == "masking":
        harshest = "mask_30pct"
        title_level = "30% Masking"
    else:
        harshest = "silence_75pct"
        title_level = "75% Silence"

    sub = df[df["perturbation"] == harshest].copy()
    if sub.empty:
        print(f"  SKIP: no data for {harshest}")
        return

    # Order by MODEL_ORDER
    sub["model_short"] = sub["model"].map(MODEL_SHORT)
    model_order = [MODEL_SHORT[m] for m in MODEL_ORDER if MODEL_SHORT[m] in sub["model_short"].values]
    sub = sub.set_index("model_short").reindex(model_order).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(model_order))
    width = 0.6

    repetition = sub["pct_repetition"].values
    syntactic = sub["pct_syntactic"].values
    content = sub["pct_content"].values

    ax.bar(x, repetition, width, label="Repetition", color="#E84855", alpha=0.85)
    ax.bar(x, syntactic, width, bottom=repetition, label="Syntactic (function words)", color="#F18F01", alpha=0.85)
    ax.bar(x, content, width, bottom=repetition + syntactic, label="Content (semantic)", color="#2E86AB", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=11, rotation=0)
    ax.set_ylabel("Insertion Type Distribution (%)", fontsize=14)
    ds_name = "Fair-Speech" if dataset == "fs" else "Common Voice"
    ax.set_title(f"Hallucination Types Under {title_level} ({ds_name})", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis="y")

    # Add total insertion count annotations
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(i, 101, f"n={int(row['total_insertions']):,}", ha="center", va="bottom", fontsize=8)

    # Add generation background shading
    ax.axvspan(-0.5, 0.5, alpha=0.06, color=GEN_COLORS[1])
    ax.axvspan(0.5, 3.5, alpha=0.06, color=GEN_COLORS[2])
    ax.axvspan(3.5, 8.5, alpha=0.06, color=GEN_COLORS[3])

    fig.tight_layout()
    save_fig(fig, output_dir, f"perturbation_hallucination_types_{perturbation_type}_{ds_label}")


# ═════════════════════════════════════════════════════════════════════════════
# P4: PARETO UNDER DEGRADATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_pareto_under_degradation(output_dir, axis, dataset):
    """Accuracy vs fairness scatter at clean, mild, and harsh perturbation."""
    ds_label = "fs" if dataset == "fs" else "cv"
    path = os.path.join(TABLES_DIR, f"table_wer_by_perturbation_and_{axis}_{ds_label}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    df = pd.read_csv(path)

    # Pick representative conditions: clean, snr_10db, mask_20pct
    conditions = [
        ("clean", "Clean", "o", 10),
        ("snr_10db", "SNR 10dB", "s", 8),
        ("mask_20pct", "Mask 20%", "^", 8),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    for cond, cond_label, marker, ms in conditions:
        sub = df[df["perturbation"] == cond]
        for _, row in sub.iterrows():
            model = row["model"]
            if model not in MODEL_ORDER:
                continue
            wer = row["overall_wer"] * 100
            mmr = row["mmr"]
            gen = MODEL_GEN[model]

            # Alpha varies by condition
            alpha = 1.0 if cond == "clean" else 0.6

            ax.scatter(wer, mmr, marker=marker, s=ms * 12, alpha=alpha,
                       color=MODEL_COLORS[model], edgecolors="black", linewidth=0.5,
                       zorder=5)

            # Label clean points
            if cond == "clean":
                ax.annotate(MODEL_SHORT[model], (wer, mmr),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=8, alpha=0.8)

    # Draw arrows from clean to harshest for each model
    for model in MODEL_ORDER:
        clean_row = df[(df["perturbation"] == "clean") & (df["model"] == model)]
        harsh_row = df[(df["perturbation"] == "mask_20pct") & (df["model"] == model)]
        if clean_row.empty or harsh_row.empty:
            continue
        x0, y0 = clean_row["overall_wer"].values[0] * 100, clean_row["mmr"].values[0]
        x1, y1 = harsh_row["overall_wer"].values[0] * 100, harsh_row["mmr"].values[0]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=MODEL_COLORS[model],
                                    alpha=0.4, lw=1.2))

    # Legend for conditions
    for cond, cond_label, marker, ms in conditions:
        ax.scatter([], [], marker=marker, s=ms * 12, color="gray",
                   edgecolors="black", linewidth=0.5, label=cond_label)
    ax.legend(fontsize=11, loc="upper right", title="Condition", title_fontsize=12)

    ax.set_xlabel("WER (%)", fontsize=14)
    ax.set_ylabel(f"MMR ({axis.replace('_', ' ').title()})", fontsize=14)
    ds_name = "Fair-Speech" if dataset == "fs" else "Common Voice"
    ax.set_title(f"Accuracy vs. Fairness Under Degradation ({ds_name})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    save_fig(fig, output_dir, f"perturbation_pareto_{axis}_{ds_label}")


# ═════════════════════════════════════════════════════════════════════════════
# P5: INSERTION RATE BY DEMOGRAPHIC GROUP UNDER PERTURBATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_insertion_rate_by_group(output_dir, axis, dataset):
    """Grouped bar chart: insertion rate per demographic group, clean vs perturbed."""
    ds_label = "fs" if dataset == "fs" else "cv"
    path = os.path.join(TABLES_DIR, f"table_insertion_rate_by_perturbation_and_{axis}_{ds_label}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    df = pd.read_csv(path)

    # Focus on whisper-large-v3 (most interesting for hallucination) at clean vs mask_20pct
    ins_cols = [c for c in df.columns if c.startswith("ins_rate_")]
    if not ins_cols:
        print(f"  SKIP: no ins_rate columns in {path}")
        return

    groups = [c.replace("ins_rate_", "") for c in ins_cols]

    models_to_show = ["whisper-large-v3", "qwen3-asr-1.7b", "granite-speech-3.3-8b"]
    conditions = ["clean", "mask_20pct"]
    cond_labels = {"clean": "Clean", "mask_20pct": "Mask 20%"}

    fig, axes = plt.subplots(1, len(models_to_show), figsize=(5 * len(models_to_show), 5), sharey=True)

    for idx, model in enumerate(models_to_show):
        ax = axes[idx]
        x = np.arange(len(groups))
        width = 0.35

        for j, cond in enumerate(conditions):
            row = df[(df["model"] == model) & (df["perturbation"] == cond)]
            if row.empty:
                continue
            vals = [row[f"ins_rate_{g}"].values[0] * 100 for g in groups]
            offset = -width / 2 + j * width
            bars = ax.bar(x + offset, vals, width, label=cond_labels[cond],
                          color=MODEL_COLORS[model] if j == 0 else "#888888",
                          alpha=0.85 if j == 0 else 0.5)

        ax.set_xticks(x)
        # Shorten group labels
        short_groups = [g[:8] + "." if len(g) > 9 else g for g in groups]
        ax.set_xticklabels(short_groups, fontsize=9, rotation=45, ha="right")
        ax.set_title(MODEL_SHORT[model], fontsize=13)
        ax.grid(True, alpha=0.2, axis="y")
        if idx == 0:
            ax.set_ylabel("Insertion Rate (%)", fontsize=13)
        ax.legend(fontsize=10)

    axis_label = axis.replace("_", " ").title()
    ds_name = "Fair-Speech" if dataset == "fs" else "Common Voice"
    fig.suptitle(f"Insertion Rate by {axis_label}, Clean vs. Masked ({ds_name})", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir, f"perturbation_insertion_rate_{axis}_{ds_label}")


# ═════════════════════════════════════════════════════════════════════════════
# P6: WER DEGRADATION CURVES BY PERTURBATION TYPE
# ═════════════════════════════════════════════════════════════════════════════

def plot_wer_degradation_curves(output_dir, axis, dataset):
    """4-panel plot: WER at each perturbation level for all 9 models."""
    ds_label = "fs" if dataset == "fs" else "cv"
    path = os.path.join(TABLES_DIR, f"table_wer_by_perturbation_and_{axis}_{ds_label}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    df = pd.read_csv(path)

    pert_types = {
        "Additive Noise": ["clean", "snr_20db", "snr_10db", "snr_0db"],
        "Reverberation": ["clean", "reverb_0.3s", "reverb_0.6s", "reverb_1.0s"],
        "Silence Injection": ["clean", "silence_25pct", "silence_50pct", "silence_75pct"],
        "Chunk Masking": ["clean", "mask_10pct", "mask_20pct", "mask_30pct"],
    }

    x_labels = {
        "Additive Noise": ["Clean", "20dB", "10dB", "0dB"],
        "Reverberation": ["Clean", "0.3s", "0.6s", "1.0s"],
        "Silence Injection": ["Clean", "25%", "50%", "75%"],
        "Chunk Masking": ["Clean", "10%", "20%", "30%"],
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)

    for idx, (title, perts) in enumerate(pert_types.items()):
        ax = axes[idx]
        for model in MODEL_ORDER:
            wers = []
            for p in perts:
                row = df[(df["model"] == model) & (df["perturbation"] == p)]
                if not row.empty:
                    wers.append(row["overall_wer"].values[0] * 100)
                else:
                    wers.append(np.nan)

            ax.plot(x_labels[title], wers,
                    marker="o", markersize=5,
                    color=MODEL_COLORS[model],
                    linewidth=1.8, alpha=0.8,
                    label=MODEL_SHORT[model] if idx == 0 else None)

        ax.set_title(title, fontsize=13)
        ax.set_ylabel("WER (%)" if idx == 0 else "", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=9, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))

    ds_name = "Fair-Speech" if dataset == "fs" else "Common Voice"
    fig.suptitle(f"WER Degradation Under Perturbation ({ds_name})", fontsize=15, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir, f"perturbation_wer_curves_{ds_label}")


# ═════════════════════════════════════════════════════════════════════════════
# P7: FAIRNESS AMPLIFICATION BY PERTURBATION TYPE (compact summary)
# ═════════════════════════════════════════════════════════════════════════════

def plot_amplification_by_type(output_dir, axis, dataset):
    """Grouped bar chart: mean amplification per perturbation type, colored by generation."""
    ds_label = "fs" if dataset == "fs" else "cv"
    path = os.path.join(FIGURES_DIR, f"fig_fairness_amplification_heatmap_{axis}_{ds_label}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    df = pd.read_csv(path)

    # Add perturbation type
    def get_pert_type(p):
        if p.startswith("snr"): return "snr"
        if p.startswith("reverb"): return "reverb"
        if p.startswith("silence"): return "silence"
        if p.startswith("mask"): return "mask"
        return "other"

    df["pert_type"] = df["perturbation"].apply(get_pert_type)

    # Average amplification across levels within each perturbation type
    avg = df.groupby(["model", "pert_type"])["amplification"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(MODEL_ORDER))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors = ["#2E86AB", "#F18F01", "#A23B72", "#44AF69"]

    for i, (pt, color) in enumerate(zip(PERTURBATION_TYPE_ORDER, colors)):
        vals = []
        for model in MODEL_ORDER:
            row = avg[(avg["model"] == model) & (avg["pert_type"] == pt)]
            vals.append(row["amplification"].values[0] if not row.empty else np.nan)
        ax.bar(x + offsets[i] * width, vals, width, label=PERTURBATION_TYPE_LABELS[pt],
               color=color, alpha=0.8)

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], fontsize=11)
    ax.set_ylabel("Mean Fairness Amplification\n(MMR_pert / MMR_clean)", fontsize=12)
    ax.legend(fontsize=10, ncol=4, loc="upper center")
    ax.grid(True, alpha=0.2, axis="y")

    # Generation background
    ax.axvspan(-0.5, 0.5, alpha=0.06, color=GEN_COLORS[1])
    ax.axvspan(0.5, 3.5, alpha=0.06, color=GEN_COLORS[2])
    ax.axvspan(3.5, 8.5, alpha=0.06, color=GEN_COLORS[3])

    axis_label = axis.replace("_", " ").title()
    ds_name = "Fair-Speech" if dataset == "fs" else "Common Voice"
    ax.set_title(f"Mean Fairness Amplification by Perturbation Type: {axis_label} ({ds_name})", fontsize=14)

    fig.tight_layout()
    save_fig(fig, output_dir, f"perturbation_amplification_by_type_{axis}_{ds_label}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate perturbation figures")
    parser.add_argument("--output_dir", type=str, default=FIGURES_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating perturbation figures")
    print("=" * 60)

    # P1: Fairness amplification heatmaps (main axes)
    print("\nP1: Fairness amplification heatmaps")
    plot_fairness_amplification_heatmap(output_dir, "ethnicity", "fs")
    plot_fairness_amplification_heatmap(output_dir, "accent", "cv")
    plot_fairness_amplification_heatmap(output_dir, "gender", "fs")
    plot_fairness_amplification_heatmap(output_dir, "age", "fs")

    # P2: WERD by generation under SNR
    print("\nP2: WERD by generation under SNR")
    plot_werd_by_generation(output_dir, "ethnicity", "fs")
    plot_werd_by_generation(output_dir, "accent", "cv")

    # P3: Hallucination type distribution
    print("\nP3: Hallucination type distribution")
    plot_hallucination_types(output_dir, "masking", "fs")
    plot_hallucination_types(output_dir, "masking", "cv")
    plot_hallucination_types(output_dir, "silence", "fs")
    plot_hallucination_types(output_dir, "silence", "cv")

    # P4: Pareto under degradation
    print("\nP4: Pareto under degradation")
    plot_pareto_under_degradation(output_dir, "ethnicity", "fs")
    plot_pareto_under_degradation(output_dir, "accent", "cv")

    # P5: Insertion rate by demographic group
    print("\nP5: Insertion rate by demographic group")
    plot_insertion_rate_by_group(output_dir, "ethnicity", "fs")
    plot_insertion_rate_by_group(output_dir, "accent", "cv")

    # P6: WER degradation curves (4-panel)
    print("\nP6: WER degradation curves")
    plot_wer_degradation_curves(output_dir, "ethnicity", "fs")
    plot_wer_degradation_curves(output_dir, "accent", "cv")

    # P7: Amplification by perturbation type (compact)
    print("\nP7: Amplification by perturbation type")
    plot_amplification_by_type(output_dir, "ethnicity", "fs")
    plot_amplification_by_type(output_dir, "accent", "cv")

    print(f"\n{'=' * 60}")
    print(f"Done! {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
