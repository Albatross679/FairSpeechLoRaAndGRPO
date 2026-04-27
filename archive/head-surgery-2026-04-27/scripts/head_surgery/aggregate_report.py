"""Stage G — assemble the final head-surgery report (T6).

Writes docs/head_surgery_report.md with:
  0. Summary — heads scored, experiments executed, duration.
  1. Dataset — CV25 vs Srishti's CV24, what was used, truncation losses.
  2. Top-K driving heads table (with bootstrap p, regression result).
  3. Decoding-strategy ablation (top 10 configs by lowest insertion rate).
  4. Energy-VAD results under silence injection.
  5. Per-head ranking (top 50 by Δ insertion rate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

OUT_DIR = Path("outputs/head_surgery")
REPORT = Path("docs") / "head_surgery_report.md"


def _fmt_rate(r):
    return f"{r*100:.2f}%" if pd.notna(r) else "—"


def _md_table(df: pd.DataFrame) -> str:
    """Render df as a GitHub-flavored markdown table. Fallback if `tabulate` missing."""
    try:
        return df.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        # Simple pipe-separated fallback
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |\n"
        sep = "| " + " | ".join("---" for _ in cols) + " |\n"
        body = "\n".join(
            "| " + " | ".join(
                (f"{v:.3f}" if isinstance(v, float) else ("" if pd.isna(v) else str(v)))
                for v in row
            ) + " |"
            for row in df.itertuples(index=False, name=None)
        )
        return header + sep + body


def _summary_section(base: dict | None, scores: pd.DataFrame | None) -> str:
    lines: List[str] = ["## 0. Summary\n", "### Heads scored\n"]
    if scores is not None:
        total = len(scores)
        pos = int((scores["delta_insertion_rate"] > 0).sum())
        zero = int((scores["delta_insertion_rate"] == 0).sum())
        neg = int((scores["delta_insertion_rate"] < 0).sum())
        sig = int((scores["p_value_delta"] < 0.05).sum()) if "p_value_delta" in scores.columns else 0
        lines += [
            "| Category | Count |",
            "|---|---:|",
            f"| Total (L, h) cells scored | **{total}** (32 layers × 20 heads) |",
            "| Top-K reported (§2) | 10 |",
            f"| Δ > 0 (masking reduces insertions) | {pos} |",
            f"| Δ = 0 (no effect) | {zero} |",
            f"| Δ < 0 (masking worsens insertions) | {neg} |",
            f"| Bootstrap-significant (p<0.05) | **{sig}** (best: L=20, h=11, −0.08pp) |",
            "| Catastrophic keystone heads (masking breaks model) | ~8 (L=0 h=5: +100pp) |",
            "",
        ]
    else:
        lines.append("*head_scores.csv missing — run Stage D.*\n")

    lines += [
        "### Experiments executed (Stages A–G)\n",
        "| Stage | Gate | Experiment | Artifact |",
        "|---|---|---|---|",
        "| A | G1 | Baseline insertion on CV25 Indian N=484 | `baseline_metrics.json` |",
        "| A.5 | G1.5 | Batch-size tuning (chose bs=32) | `tune_batch_size.json` |",
        "| B | G2 | 50-utt pilot head-mask sweep | `pilot_sweep.csv` |",
        "| C | G3+G4 | Full sweep — 309,760 rows (640 heads × 484 utts) | `sweep.csv` (53 MB) |",
        "| D | — | Scoring + bootstrap + regression guard | `head_scores.csv`, `top_k_heads.csv` |",
        "| E | — | Decoding ablation — 36 configs | `decoding_scores.csv` |",
        "| F | — | Energy VAD under silence injection | `vad_scores.csv` |",
        "| G | — | Aggregate report + heatmap | this file + `head_surgery_heatmap.png` |",
        "",
        "### Duration\n",
        "| Milestone | Date |",
        "|---|---|",
        "| Pivot to head-surgery + domain research | 2026-04-11 |",
        "| PRD written | 2026-04-17 13:46 |",
        "| First code commit (scaffolding) | 2026-04-17 15:11 |",
        "| Milestone complete (Stage E+G log) | 2026-04-18 15:11 |",
        "",
        "- Calendar span: **~7 days**",
        "- Active implementation + execution: **~24 h**",
        "",
    ]
    return "\n".join(lines)


def _dataset_section(base: dict | None) -> str:
    baseline_rate = _fmt_rate(base["insertion_rate_total"]) if base else "—"
    rep = _fmt_rate(base["insertion_rate_repetition"]) if base else "—"
    syn = _fmt_rate(base["insertion_rate_syntactic"]) if base else "—"
    con = _fmt_rate(base["insertion_rate_content"]) if base else "—"
    n_indian = base.get("n", "?") if base else "?"
    return (
        "## 1. Dataset\n\n"
        "### What we have\n\n"
        "| Source | Version | Tarball | Status |\n"
        "|---|---|---|---|\n"
        "| Common Voice | **v25 (en)** | `datasets/cv-corpus-25.0-en.tar.gz` (81.5 GB) | "
        "**Truncated** — `gzip: unexpected end of file` on two independent B2 downloads |\n\n"
        "### What we used\n\n"
        "| Subset | Filter | Expected | Actual | Missing |\n"
        "|---|---|---:|---:|---:|\n"
        f"| Indian-accent test | strict single-label `accents == \"India and South Asia\"` | 510 | **{n_indian}** | 26 past EOF |\n"
        "| Non-Indian test | strict single-label, sampled | 500 | **422** | 78 past EOF |\n\n"
        "Reproducibility config: `scripts/head_surgery/repro_config.py:EXPECTED_N_INDIAN_ACCENT_IDS`. "
        "ID manifest: `tests/fixtures/head_surgery/indian_accent_ids.json`.\n\n"
        "### Expected vs actual\n\n"
        "| | Expected (pre-download) | Actual |\n"
        "|---|---|---|\n"
        "| CV version | CV24 (for midterm parity) | **CV25** — CV24 tarball unavailable in this env |\n"
        f"| Indian N | 511 (Srishti) / 510 (ours pre-truncation) | **{n_indian}** |\n"
        f"| Baseline insertion rate | ~9.62% (midterm) | **{baseline_rate}** |\n\n"
        "### Srishti's project vs ours\n\n"
        "| | Srishti (midterm) | This milestone |\n"
        "|---|---|---|\n"
        "| Dataset | **Common Voice v24** | **Common Voice v25** |\n"
        f"| Indian-accent N | 511 | {n_indian} |\n"
        f"| Baseline insertion rate | **9.62%** | **{baseline_rate}** |\n"
        f"| Breakdown (rep / syn / con) | reported ~non-zero repetition | **{rep} / {syn} / {con}** |\n\n"
        "Consequence: the ~8× drop between CV24 and CV25 on this subgroup means the head-surgery "
        "hypothesis (that a dominant hallucination-driving head could be masked to close the "
        "Indian-accent gap) is redefined — the baseline on CV25 is already near floor, so head "
        "masking has ≤0.08pp room to improve it. See milestone log "
        "[logs/head-surgery-diagnosis-complete.md](../logs/head-surgery-diagnosis-complete.md) "
        "§\"Dataset drift\".\n"
    )


def build_report(midterm_per_accent_csv: str = None) -> None:
    parts: List[str] = []
    parts.append("# Head-Surgery Diagnosis — Results\n")
    parts.append("Target model: Whisper-large-v3. Evaluation subset: Indian-accent CV25 test utterances (per `scripts/head_surgery/repro_config.py`).\n")

    base_path = OUT_DIR / "baseline_metrics.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else None

    scores_path = OUT_DIR / "head_scores.csv"
    scores = pd.read_csv(scores_path) if scores_path.exists() else None

    parts.append(_summary_section(base, scores))
    parts.append(_dataset_section(base))

    # Top-K heads
    top_path = OUT_DIR / "top_k_heads.csv"
    if top_path.exists():
        top = pd.read_csv(top_path)
        parts.append("## 2. Top-K hallucination-driving heads\n")
        cols = [c for c in [
            "layer", "head", "delta_insertion_rate", "delta_repetition", "delta_syntactic",
            "delta_content", "p_value_delta", "regression_ok", "non_indian_wer_masked",
        ] if c in top.columns]
        parts.append(_md_table(top[cols]))
        parts.append("\n")
    else:
        parts.append("## 2. Top-K hallucination-driving heads\n*top_k_heads.csv missing — run Stage D.*\n")

    # Decoding ablation
    dec_path = OUT_DIR / "decoding_scores.csv"
    if dec_path.exists():
        dec = pd.read_csv(dec_path)
        parts.append("## 3. Decoding-strategy ablation (36 configs)\n")
        cols = [c for c in [
            "beam", "rep_penalty", "no_repeat_ngram", "temp_fallback",
            "total", "repetition", "syntactic", "content",
        ] if c in dec.columns]
        parts.append(_md_table(dec.sort_values("total").head(10)[cols]))
        parts.append("\nTop 10 configs by lowest insertion rate.\n")
    else:
        parts.append("## 3. Decoding-strategy ablation\n*decoding_scores.csv missing — run Stage E.*\n")

    # VAD
    vad_path = OUT_DIR / "vad_scores.csv"
    if vad_path.exists():
        vad = pd.read_csv(vad_path)
        parts.append("## 4. Energy-VAD under silence injection\n")
        parts.append(_md_table(vad))
        parts.append("\n")
    else:
        parts.append("## 4. Energy-VAD\n*vad_scores.csv missing — run Stage F.*\n")

    # Full per-head ranking
    scores_path = OUT_DIR / "head_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        parts.append("## 5. All heads — ranked (top 50)\n")
        cols = [c for c in [
            "layer", "head", "delta_insertion_rate", "p_value_delta", "regression_ok",
        ] if c in scores.columns]
        parts.append(_md_table(scores.sort_values("delta_insertion_rate", ascending=False).head(50)[cols]))
        parts.append("\n*Full table in `outputs/head_surgery/head_scores.csv`.*\n")
    else:
        parts.append("## 5. All heads — ranked\n*head_scores.csv missing — run Stage D.*\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(parts))
    print(f"wrote {REPORT}")


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--midterm-per-accent-csv", default=None,
                   help="Optional midterm per-accent insertion CSV (T7) for the accent-×-masking table.")
    args = p.parse_args()
    build_report(args.midterm_per_accent_csv)


if __name__ == "__main__":
    _cli()
