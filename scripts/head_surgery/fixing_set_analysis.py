"""Post-hoc fixing-set analysis on the 640-head × 484-utterance sweep.

For each Indian-accent utterance with ≥1 hallucinated token at baseline, find
every (layer, head) whose masking strictly reduces its insertion count; then
solve min-set-cover over those heads under two additional filters —
(i) the head does not introduce new hallucinations on any other utterance, and
(ii) the head passes the Stage D non-Indian-WER regression guard (regression_ok).

Inputs (all under outputs/head_surgery/):
  sweep.csv                  — Stage C
  baseline_predictions.csv   — Stage A
  head_scores.csv            — Stage D

Outputs:
  fixing_set_per_utterance.csv
  coverage_matrix.npz
  minimum_surgical_set.json

See docs/superpowers/plans/2026-04-18-head-surgery-fixing-set-analysis.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

OUT_DIR = Path("outputs/head_surgery")


def count_insertions(reference: str, hypothesis: str) -> int:
    """Number of inserted tokens in hypothesis relative to reference.

    Reference and hypothesis are normalized via Whisper's EnglishTextNormalizer
    (matching the rest of the project). Returns 0 if either side is empty.
    """
    from scripts.head_surgery.insertion_classifier import categorize_insertions
    from scripts.inference.run_inference import normalize_text

    ref = normalize_text(reference or "")
    hyp = normalize_text(hypothesis or "")
    if not ref or not hyp:
        return 0
    return len(categorize_insertions(ref, hyp))


def build_count_table(sweep_csv: Path, baseline_csv: Path) -> pd.DataFrame:
    """Long-form DataFrame: columns = [condition, layer, head, id, count].

    `condition` is "baseline" for the no-mask rows (layer=head=-1 sentinel not
    used here — baseline comes from baseline_predictions.csv), otherwise the
    string "L{L}_h{H}". Every utterance appears once per condition.
    """
    sweep = pd.read_csv(sweep_csv)
    baseline = pd.read_csv(baseline_csv)

    # Baseline rows
    baseline_counts = [
        {"condition": "baseline", "layer": -1, "head": -1,
         "id": str(r["id"]), "count": count_insertions(r["reference"], r["hypothesis"])}
        for _, r in baseline.iterrows()
    ]
    # Masked rows
    masked_counts = []
    for _, r in sweep.iterrows():
        L, h = int(r["layer"]), int(r["head"])
        if L == -1 or h == -1:
            # Pilot-baseline sentinel rows, if any — skip (baseline is in baseline_csv)
            continue
        masked_counts.append({
            "condition": f"L{L}_h{h}",
            "layer": L,
            "head": h,
            "id": str(r["id"]),
            "count": count_insertions(r["reference"], r["hypothesis"]),
        })
    return pd.DataFrame(baseline_counts + masked_counts)
