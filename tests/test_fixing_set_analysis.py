"""Tests for scripts/head_surgery/fixing_set_analysis.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "head_surgery"


def test_prerequisite_csvs_present_and_shaped():
    """If prerequisites are missing, the analysis cannot run — surface it clearly."""
    sweep = OUT_DIR / "sweep.csv"
    baseline = OUT_DIR / "baseline_predictions.csv"
    scores = OUT_DIR / "head_scores.csv"
    assert sweep.exists(), f"missing {sweep} — Stage C must have run"
    assert baseline.exists(), f"missing {baseline} — Stage A must have run"
    assert scores.exists(), f"missing {scores} — Stage D must have run"
    sweep_df = pd.read_csv(sweep, nrows=2)
    for col in ["layer", "head", "id", "reference", "hypothesis"]:
        assert col in sweep_df.columns, f"sweep.csv missing column {col}"
    baseline_df = pd.read_csv(baseline, nrows=2)
    for col in ["id", "reference", "hypothesis"]:
        assert col in baseline_df.columns, f"baseline_predictions.csv missing column {col}"
    scores_df = pd.read_csv(scores, nrows=2)
    for col in ["layer", "head", "regression_ok"]:
        assert col in scores_df.columns, f"head_scores.csv missing column {col}"
