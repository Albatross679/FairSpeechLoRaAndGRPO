#!/usr/bin/env python3
"""
Regenerate 10 paper figures with larger fonts for readability.

Produces ONLY:
  1. fig1_ethnicity_heatmap.pdf
  2. fig2_accent_heatmap.pdf
  3. hallucination_categories_CV.pdf
  4. fig8_scaling_curves.pdf
  5. perturbation_wer_curves_fs.pdf
  6. perturbation_amplification_heatmap_ethnicity_fs.pdf
  7. perturbation_amplification_heatmap_accent_cv.pdf
  8. perturbation_hallucination_types_masking_fs.pdf
  9. fig7_pareto.pdf
  10. perturbation_pareto_ethnicity_fs.pdf

Data, colors, labels are identical to originals. Only font sizes and figsize change.

Usage:
    venv/bin/python scripts/regenerate_figures_large_fonts.py
"""

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

# ── Global font settings ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = "/users/PAS2030/srishti/asr_fairness"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "overleaf/figures")
TABLES_DIR = os.path.join(PROJECT_DIR, "results/tables")
FIGURES_DIR = os.path.join(PROJECT_DIR, "results/figures")

CV_ANALYSIS = os.path.join(PROJECT_DIR, "results/commonvoice/analysis/full_analysis.json")
FS_ANALYSIS = os.path.join(PROJECT_DIR, "results/fairspeech/analysis/full_analysis_fs.json")
HALLUCINATION_JSON = os.path.join(PROJECT_DIR, "results/hallucination_analysis/hallucination_analysis.json")

# ── Constants (identical to original scripts) ────────────────────────────────
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

# For hallucination_categories_CV, original used slightly different yellow
MODEL_COLORS_HALLU = dict(MODEL_COLORS)
MODEL_COLORS_HALLU["qwen3-asr-0.6b"] = "#FFFF00"

GEN_COLORS = {1: "#2E86AB", 2: "#C73E1D", 3: "#44AF69"}
GEN_LABELS = {1: "Gen 1 (CTC)", 2: "Gen 2 (Enc-Dec)", 3: "Gen 3 (LLM-ASR)"}

ETHNICITY_ORDER = ["White", "Black/AA", "Hispanic", "Asian", "Native American", "Pacific Islander", "Middle Eastern"]
ACCENT_ORDER = ["us", "england", "canada", "australia", "indian", "african"]

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def save_fig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.pdf")


def get_ordered_models(data):
    return [m for m in MODEL_ORDER if m in data]


def get_group_wers(data, model, axis):
    if model not in data:
        return {}
    axis_data = data[model].get(axis, {})
    gw = axis_data.get("group_wers", {})
    return {k: v.get("wer", 0) for k, v in gw.items()}


def get_mmr(model_data, axis):
    axis_data = model_data.get(axis, {})
    if "max_min_ratio" in axis_data:
        return axis_data["max_min_ratio"]
    return axis_data.get("mmr", np.nan)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Ethnicity WER Heatmap (Fair-Speech) -- side-by-side at 0.49\textwidth
# ═════════════════════════════════════════════════════════════════════════════

def gen_fig1_ethnicity_heatmap(fs_data):
    models = get_ordered_models(fs_data)

    # Sort columns by average WER
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

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "WER (%)"}, annot_kws={"fontsize": 13})
    ax.set_title("WER (%) by Ethnicity (Fair-Speech)", fontsize=18)
    ax.set_xlabel("Ethnicity", fontsize=16)
    ax.set_ylabel("Model", fontsize=16)
    ax.tick_params(axis='x', rotation=30, labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    plt.tight_layout()
    save_fig(fig, "fig1_ethnicity_heatmap")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Accent WER Heatmap (Common Voice) -- side-by-side at 0.49\textwidth
# ═════════════════════════════════════════════════════════════════════════════

def gen_fig2_accent_heatmap(cv_data):
    models = get_ordered_models(cv_data)
    groups = ACCENT_ORDER

    matrix = []
    for m in models:
        gw = get_group_wers(cv_data, m, "accent")
        row = [gw.get(g, np.nan) * 100 for g in groups]
        matrix.append(row)

    df = pd.DataFrame(matrix, index=[MODEL_SHORT.get(m, m) for m in models], columns=groups)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "WER (%)"}, annot_kws={"fontsize": 13})
    ax.set_title("WER (%) by Accent (Common Voice)", fontsize=18)
    ax.set_xlabel("Accent", fontsize=16)
    ax.set_ylabel("Model", fontsize=16)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    plt.tight_layout()
    save_fig(fig, "fig2_accent_heatmap")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: hallucination_categories_CV -- stacked bar at 0.8\textwidth
# ═════════════════════════════════════════════════════════════════════════════

def gen_hallucination_categories_cv(hallu_data):
    """Stacked bar chart of hallucination categories per model (CV dataset)."""
    cv_data = hallu_data.get("cv", {})
    if not cv_data:
        print("  SKIP: no CV data in hallucination JSON")
        return

    models = [m for m in MODEL_ORDER if m in cv_data]
    categories = ["syntactic_completion", "content_hallucination", "repetition"]
    cat_colors = {"syntactic_completion": "#2E86AB", "content_hallucination": "#C73E1D", "repetition": "#F18F01"}
    cat_labels = {"syntactic_completion": "Syntactic", "content_hallucination": "Content", "repetition": "Repetition"}

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    bar_width = 0.6

    bottoms = np.zeros(len(models))
    for cat in categories:
        vals = []
        for model in models:
            total_ins = sum(r.get("total_insertions", 0) for r in cv_data[model].values())
            cat_count = sum(r.get("category_counts", {}).get(cat, 0) for r in cv_data[model].values())
            vals.append(cat_count / max(1, total_ins) * 100)
        ax.bar(x, vals, bar_width, bottom=bottoms, label=cat_labels[cat],
               color=cat_colors[cat], edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_ylabel("% of Insertions", fontsize=16)
    ax.set_title("Hallucination Category Distribution (Common Voice)", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "hallucination_categories_CV")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: fig8_scaling_curves -- line plot at 0.8\textwidth
# ═════════════════════════════════════════════════════════════════════════════

def gen_fig8_scaling_curves(cv_data, fs_data):
    scaling_families = {
        "Whisper": (["whisper-small", "whisper-medium", "whisper-large-v3"], "#C73E1D", "o"),
        "Qwen3": (["qwen3-asr-0.6b", "qwen3-asr-1.7b"], "#44AF69", "s"),
        "Granite": (["granite-speech-3.3-2b", "granite-speech-3.3-8b"], "#6B4C9A", "D"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

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
                ax.plot(params, mmrs, f'{marker}-', color=color, markersize=12,
                       linewidth=2.5, label=family)
                for x, y, m in zip(params, mmrs, [m for m in models if m in data]):
                    ax.annotate(MODEL_SHORT[m], (x, y), textcoords="offset points",
                               xytext=(0, 12), ha='center', fontsize=11)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))
        ax.set_xlabel("Parameters (log scale)", fontsize=15)
        ax.set_ylabel("Max/Min Ratio (MMR)", fontsize=15)
        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=13)

    fig.suptitle("Scaling x Fairness: Three Architecture Families", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig8_scaling_curves")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: perturbation_wer_curves_fs -- 4-panel line plot at 0.8\textwidth
# ═════════════════════════════════════════════════════════════════════════════

def gen_perturbation_wer_curves_fs():
    axis = "ethnicity"
    ds_label = "fs"
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

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=False)

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
                    marker="o", markersize=6,
                    color=MODEL_COLORS[model],
                    linewidth=2, alpha=0.8,
                    label=MODEL_SHORT[model] if idx == 0 else None)

        ax.set_title(title, fontsize=15)
        ax.set_ylabel("WER (%)" if idx == 0 else "", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=9, fontsize=11,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("WER Degradation Under Perturbation (Fair-Speech)", fontsize=17, y=1.02)
    fig.tight_layout()
    save_fig(fig, "perturbation_wer_curves_fs")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES 6-7: perturbation_amplification_heatmap -- full width at \textwidth
# ═════════════════════════════════════════════════════════════════════════════

def gen_perturbation_amplification_heatmap(axis, dataset):
    ds_label = "fs" if dataset == "fs" else "cv"
    path = os.path.join(FIGURES_DIR, f"fig_fairness_amplification_heatmap_{axis}_{ds_label}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    df = pd.read_csv(path)

    df["model_short"] = df["model"].map(MODEL_SHORT)
    df["pert_short"] = df["perturbation"].map(PERTURBATION_SHORT)

    model_order = [MODEL_SHORT[m] for m in MODEL_ORDER if m in df["model"].values]
    pert_order = [PERTURBATION_SHORT[p] for p in PERTURBATION_ORDER if p in df["perturbation"].values]

    pivot = df.pivot(index="model_short", columns="pert_short", values="amplification")
    pivot = pivot.reindex(index=model_order, columns=pert_order)

    fig, ax = plt.subplots(figsize=(14, 6))

    vmin = min(0.5, pivot.min().min())
    vmax = max(1.5, pivot.max().max())

    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn_r", center=1.0,
        vmin=vmin, vmax=vmax,
        annot=True, fmt=".2f", annot_kws={"fontsize": 11},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "$\\alpha$ (MMR_pert / MMR_clean)", "shrink": 0.8},
    )

    # Vertical lines to separate perturbation types
    for i in [3, 6, 9]:
        ax.axvline(x=i, color="black", linewidth=1.5)

    # Horizontal lines to separate generations
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
    ax.set_title(f"Fairness Gap Amplification Under Perturbation: {axis_label} ({ds_name})", fontsize=16, pad=12)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=13)

    # Perturbation type labels at top
    type_positions = [(1.5, "Noise"), (4.5, "Reverb"), (7.5, "Silence"), (10.5, "Masking")]
    for pos, label in type_positions:
        if pos < len(pert_order):
            ax.text(pos, -0.8, label, ha="center", va="bottom", fontsize=13, fontweight="bold",
                    transform=ax.get_xaxis_transform())

    fig.tight_layout()
    save_fig(fig, f"perturbation_amplification_heatmap_{axis}_{ds_label}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8: perturbation_hallucination_types_masking_fs -- stacked bar at 0.8\textwidth
# ═════════════════════════════════════════════════════════════════════════════

def gen_perturbation_hallucination_types_masking_fs():
    ds_label = "fs"
    path = os.path.join(TABLES_DIR, f"table_hallucination_classification_masking_{ds_label}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    df = pd.read_csv(path)

    harshest = "mask_30pct"
    title_level = "30% Masking"

    sub = df[df["perturbation"] == harshest].copy()
    if sub.empty:
        print(f"  SKIP: no data for {harshest}")
        return

    sub["model_short"] = sub["model"].map(MODEL_SHORT)
    model_order = [MODEL_SHORT[m] for m in MODEL_ORDER if MODEL_SHORT[m] in sub["model_short"].values]
    sub = sub.set_index("model_short").reindex(model_order).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(model_order))
    width = 0.6

    repetition = sub["pct_repetition"].values
    syntactic = sub["pct_syntactic"].values
    content = sub["pct_content"].values

    ax.bar(x, repetition, width, label="Repetition", color="#E84855", alpha=0.85)
    ax.bar(x, syntactic, width, bottom=repetition, label="Syntactic (function words)", color="#F18F01", alpha=0.85)
    ax.bar(x, content, width, bottom=repetition + syntactic, label="Content (semantic)", color="#2E86AB", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=13, rotation=0)
    ax.set_ylabel("Insertion Type Distribution (%)", fontsize=15)
    ax.set_title(f"Hallucination Types Under {title_level} (Fair-Speech)", fontsize=16)
    ax.legend(fontsize=13, loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis="y")
    ax.tick_params(axis='y', labelsize=13)

    # Total insertion count annotations
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(i, 101, f"n={int(row['total_insertions']):,}", ha="center", va="bottom", fontsize=10)

    # Generation background shading
    ax.axvspan(-0.5, 0.5, alpha=0.06, color=GEN_COLORS[1])
    ax.axvspan(0.5, 3.5, alpha=0.06, color=GEN_COLORS[2])
    ax.axvspan(3.5, 8.5, alpha=0.06, color=GEN_COLORS[3])

    fig.tight_layout()
    save_fig(fig, "perturbation_hallucination_types_masking_fs")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 9: fig7_pareto -- side-by-side at 0.49\textwidth
# ═════════════════════════════════════════════════════════════════════════════

def gen_fig7_pareto(fs_data):
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 7))

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
        ax.scatter(wer * 100, mmr, s=200, c=GEN_COLORS[gen],
                  edgecolors="black", linewidth=1, zorder=5)
        ax.annotate(MODEL_SHORT[m], (wer * 100, mmr),
                   textcoords="offset points", xytext=(8, 5), fontsize=13)

    # Pareto frontier
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
    ax.legend(handles=handles, fontsize=13)
    ax.set_xlabel("Overall WER (%)", fontsize=16)
    ax.set_ylabel("Ethnicity MMR (higher = less fair)", fontsize=16)
    ax.set_title("Accuracy vs. Fairness (Fair-Speech)", fontsize=18)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "fig7_pareto")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 10: perturbation_pareto_ethnicity_fs -- side-by-side at 0.49\textwidth
# ═════════════════════════════════════════════════════════════════════════════

def gen_perturbation_pareto_ethnicity_fs():
    axis = "ethnicity"
    ds_label = "fs"
    path = os.path.join(TABLES_DIR, f"table_wer_by_perturbation_and_{axis}_{ds_label}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    df = pd.read_csv(path)

    conditions = [
        ("clean", "Clean", "o", 10),
        ("snr_10db", "SNR 10dB", "s", 8),
        ("mask_20pct", "Mask 20%", "^", 8),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    for cond, cond_label, marker, ms in conditions:
        sub = df[df["perturbation"] == cond]
        for _, row in sub.iterrows():
            model = row["model"]
            if model not in MODEL_ORDER:
                continue
            wer = row["overall_wer"] * 100
            mmr = row["mmr"]

            alpha = 1.0 if cond == "clean" else 0.6

            ax.scatter(wer, mmr, marker=marker, s=ms * 14, alpha=alpha,
                       color=MODEL_COLORS[model], edgecolors="black", linewidth=0.5,
                       zorder=5)

            if cond == "clean":
                ax.annotate(MODEL_SHORT[model], (wer, mmr),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=11, alpha=0.8)

    # Arrows from clean to harshest
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
        ax.scatter([], [], marker=marker, s=ms * 14, color="gray",
                   edgecolors="black", linewidth=0.5, label=cond_label)
    ax.legend(fontsize=13, loc="upper right", title="Condition", title_fontsize=14)

    ax.set_xlabel("WER (%)", fontsize=16)
    ax.set_ylabel("MMR (Ethnicity)", fontsize=16)
    ax.set_title("Accuracy vs. Fairness Under Degradation (Fair-Speech)", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=13)

    fig.tight_layout()
    save_fig(fig, "perturbation_pareto_ethnicity_fs")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Regenerating 10 figures with large fonts")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Load data
    cv_data, fs_data, hallu_data = {}, {}, {}

    if os.path.exists(CV_ANALYSIS):
        with open(CV_ANALYSIS) as f:
            cv_data = json.load(f)
        print(f"Loaded CV analysis: {len(cv_data)} models")

    if os.path.exists(FS_ANALYSIS):
        with open(FS_ANALYSIS) as f:
            fs_data = json.load(f)
        print(f"Loaded FS analysis: {len(fs_data)} models")

    if os.path.exists(HALLUCINATION_JSON):
        with open(HALLUCINATION_JSON) as f:
            hallu_data = json.load(f)
        print(f"Loaded hallucination data")

    # Generate all 10 figures
    print("\n1. fig1_ethnicity_heatmap")
    gen_fig1_ethnicity_heatmap(fs_data)

    print("\n2. fig2_accent_heatmap")
    gen_fig2_accent_heatmap(cv_data)

    print("\n3. hallucination_categories_CV")
    gen_hallucination_categories_cv(hallu_data)

    print("\n4. fig8_scaling_curves")
    gen_fig8_scaling_curves(cv_data, fs_data)

    print("\n5. perturbation_wer_curves_fs")
    gen_perturbation_wer_curves_fs()

    print("\n6. perturbation_amplification_heatmap_ethnicity_fs")
    gen_perturbation_amplification_heatmap("ethnicity", "fs")

    print("\n7. perturbation_amplification_heatmap_accent_cv")
    gen_perturbation_amplification_heatmap("accent", "cv")

    print("\n8. perturbation_hallucination_types_masking_fs")
    gen_perturbation_hallucination_types_masking_fs()

    print("\n9. fig7_pareto")
    gen_fig7_pareto(fs_data)

    print("\n10. perturbation_pareto_ethnicity_fs")
    gen_perturbation_pareto_ethnicity_fs()

    print(f"\n{'=' * 60}")
    print(f"Done! All 10 figures saved to {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
