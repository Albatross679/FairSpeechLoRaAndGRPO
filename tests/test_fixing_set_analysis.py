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


from scripts.head_surgery.fixing_set_analysis import count_insertions


def test_count_insertions_identical_is_zero():
    assert count_insertions("the cat sat", "the cat sat") == 0


def test_count_insertions_extra_word():
    # "really" is inserted — classifier treats it as content or syntactic filler
    assert count_insertions("the cat sat", "the really cat sat") == 1


def test_count_insertions_repetition_loop():
    # "thank you" repeated 3× extra times → 6 extra tokens
    assert count_insertions("thank you", "thank you thank you thank you thank you") == 6


def test_count_insertions_handles_empty():
    # Empty refs and hyps should not crash and should return 0
    assert count_insertions("", "") == 0
    assert count_insertions("", "anything") == 0
    assert count_insertions("some reference", "") == 0
