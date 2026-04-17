"""Stage G — assemble the final head-surgery report (T6).

Writes docs/head_surgery_report.md with:
  1. Baseline metrics (Gate G1).
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


def build_report(midterm_per_accent_csv: str = None) -> None:
    parts: List[str] = []
    parts.append("# Head-Surgery Diagnosis — Results\n")
    parts.append("Target model: Whisper-large-v3. Evaluation subset: Indian-accent CV25 test utterances (per `scripts/head_surgery/repro_config.py`).\n")

    # Baseline
    base_path = OUT_DIR / "baseline_metrics.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
        parts.append("## 1. Baseline (Gate G1)\n")
        parts.append(f"- Utterances: {base.get('n', '?')}\n")
        parts.append(f"- Insertion rate: **{_fmt_rate(base['insertion_rate_total'])}** (midterm target 9.62%, CV24).\n")
        parts.append(f"- Breakdown — repetition: {_fmt_rate(base['insertion_rate_repetition'])}, "
                     f"syntactic: {_fmt_rate(base['insertion_rate_syntactic'])}, "
                     f"content: {_fmt_rate(base['insertion_rate_content'])}.\n")
    else:
        parts.append("## 1. Baseline (Gate G1)\n*baseline_metrics.json missing — run Stage A.*\n")

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
